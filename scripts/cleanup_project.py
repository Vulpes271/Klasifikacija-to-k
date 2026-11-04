"""Enkraten, omejen popis/odstranitev projektnih uteži in ponovljivih predpomnilnikov."""
import argparse
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'maintenance/cleanup-florence-qwen.json'
TAGS=['qwen3.5:4b','qwen3.5:4b-q4_K_M','qwen3.5:4b-bf16','qwen3.5:9b-bf16','gemma3:4b-it-q4_K_M']
DIRECTORIES=['results/thud-extended/models/nvila','results/stage1-presentation/models/qwen3-vl',
             'results/lilocbench/cache','results/lilocbench/indices',
             'llm_object_classifier/__pycache__','scripts/__pycache__','tests/__pycache__']


def allocated(path):
    """Dejanski dodeljeni bloki; ne sešteva navideznih velikosti redkih datotek."""
    return int(subprocess.check_output(['du','-sk',str(path)],text=True).split()[0])*1024 if path.exists() else 0


def report_links(tracked):
    """Poišče lokalne povezave HTML/Markdown/LaTeX v vseh ohranjenih Git dokumentih."""
    links=[]
    for path in tracked:
        if path.suffix.lower() not in ('.html','.md','.tex'):continue
        text=path.read_text(errors='replace')
        values=re.findall(r'''(?:href|src)\s*=\s*["']([^"']+)["']''',text)
        values+=re.findall(r'\]\(([^\n)]+)\)',text)
        values+=re.findall(r'\\(?:includegraphics|input|include)(?:\[[^\]]*\])?\{([^}]+)\}',text)
        for value in values:
            value=html.unescape(value.strip().strip('<>'));url=urlsplit(value)
            if url.scheme or url.netloc or not url.path:continue
            target=(path.parent/unquote(url.path)).resolve()
            links.append((target,str(path.relative_to(ROOT))))
    return links


def audit():
    """Dovoli samo vnaprej naštete kopije, ki niso Git podatki, reference ali aktivne poti."""
    tracked=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().strip('\0').split('\0')]
    links=report_links(tracked);cfg=json.loads((ROOT/'configs/pipeline.json').read_text())
    active=[(ROOT/cfg['florence']['model_dir']).resolve(),(ROOT/cfg['input']).resolve(),(ROOT/cfg['output']).resolve(),ROOT/'reviews']
    candidates=[]
    for relative in DIRECTORIES:
        path=ROOT/relative
        if not path.exists():continue
        reasons=[]
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):reasons.append('Ni samostojna projektna mapa')
        if any(p.resolve().is_relative_to(path.resolve()) for p in tracked):reasons.append('Vsebuje Git podatke')
        if any(p.is_relative_to(path.resolve()) or path.resolve().is_relative_to(p) for p in active):reasons.append('Aktivna konfiguracija ali zasebne ocene')
        connected=[origin for target,origin in links if target.is_relative_to(path.resolve())]
        if connected:reasons.append('Povezava iz ohranjenega dokumenta')
        if any(p.is_symlink() and not p.resolve().is_relative_to(path.resolve()) for p in path.rglob('*')):reasons.append('Povezava izven projektne kopije')
        if relative.endswith('/nvila'):
            if json.loads((path/'config.json').read_text()).get('model_type')!='llava_llama':reasons.append('Nepričakovan NVILA checkpoint')
        if relative.endswith('/qwen3-vl'):
            if json.loads((path/'config.json').read_text()).get('model_type')!='qwen3_vl':reasons.append('Nepričakovan Qwen3-VL checkpoint')
        candidates.append({'path':relative,'allocated_bytes':allocated(path),'allowed':not reasons,'blocked_by':reasons,'linked_from':connected})
    for name in ('prepare.log','selection.log','florence.log','text.log','progress.md'):
        source=ROOT/'results/lilocbench'/name;copy=ROOT/'results models/lilocbench/logs'/name
        if source.exists() and copy.exists() and source.read_bytes()==copy.read_bytes():
            connected=[origin for target,origin in links if target==source.resolve()]
            candidates.append({'path':str(source.relative_to(ROOT)),'allocated_bytes':allocated(source),'allowed':not connected,
                'blocked_by':connected,'identical_preserved_copy':str(copy.relative_to(ROOT))})
    mounts=json.loads(subprocess.check_output(['docker','inspect','klasifikacija-ollama-1','--format','{{json .Mounts}}'],text=True))
    if not any(m.get('Name')=='klasifikacija_ollama-models' and m['Destination']=='/root/.ollama' for m in mounts):
        raise ValueError('Ollama ni priključena na potrjeni projektni volumen; brisanje tagov zavrnjeno')
    if set(TAGS)&set(cfg['llm']['alternatives'].values()):raise ValueError('Odstranjevanje aktivnega taga zavrnjeno')
    return {'checkpoint':'de4e087ad5fddfa1f87162f6881ffe493822a8a7','tracked_files_checked':len(tracked),'local_links_checked':len(links),
        'candidates':candidates,'exact_project_tags':TAGS,'preserved':['reviews/ (vse baze, izvozi in varnostne kopije)',
        'vsi objavljeni rezultati in izbrani podatki','Florence uteži in stara združljiva simbolna pot',
        'vse Docker slike/okolja, izvorna koda VILA in kalibracije','drugi lokalni eksperimentalni rezultati z nejasno edinstvenostjo']}


def apply(plan):
    """Odstrani samo dovoljena konkretna imena; nikoli skupnega HF/Docker predpomnilnika."""
    before=int(subprocess.check_output(['docker','exec','klasifikacija-ollama-1','du','-sk','/root/.ollama/models'],text=True).split()[0])*1024
    listing=subprocess.check_output(['docker','exec','klasifikacija-ollama-1','ollama','list'],text=True)
    plan['ollama_list_before']=listing
    installed={line.split()[0] for line in listing.splitlines()[1:]};deleted=[]
    for tag in TAGS:
        if tag not in installed:continue
        subprocess.run(['docker','exec','klasifikacija-ollama-1','ollama','rm',tag],check=True,capture_output=True)
        deleted.append(tag)
    freed=0;retired=ROOT/'models/retired-metadata';retired_before=allocated(retired)
    for item in plan['candidates']:
        if not item['allowed']:continue
        path=ROOT/item['path']
        if item['path'].endswith(('/nvila','/qwen3-vl')):
            # Konfiguracije so lahko lokalno prilagojene. Pred brisanjem uteži
            # ohranimo vse neutežne datoteke lokalno in majhne konfiguracije v Gitu.
            name=path.name;metadata=[]
            for old in path.rglob('*'):
                if not old.is_file() or '.cache' in old.parts or old.suffix in ('.safetensors','.bin','.pt','.pth','.gguf'):continue
                relative=old.relative_to(path);target=retired/name/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,target)
                if old.name.endswith(('config.json','.index.json')) and old.stat().st_size<100000:
                    public=ROOT/'maintenance/retired-model-configs'/name/relative
                    public.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,public)
                metadata.append(str(relative))
            item['preserved_metadata_local']=str((retired/name).relative_to(ROOT));item['preserved_metadata_files']=metadata
            # Prejšnji vsebniki imajo druge UID-je. Root dobi samo projektni bind,
            # brez omrežja; ukaz odstrani natanko vnaprej preverjeno mapo.
            subprocess.run(['docker','run','--rm','--network','none','-v',f'{ROOT}:/project','thud-florence:local',
                'python','-c','import shutil,sys; shutil.rmtree(sys.argv[1])','/project/'+item['path']],check=True)
        elif path.is_dir():shutil.rmtree(path)
        else:path.unlink()
        assert not path.exists();item['removed']=True;freed+=item['allocated_bytes']
    after=int(subprocess.check_output(['docker','exec','klasifikacija-ollama-1','du','-sk','/root/.ollama/models'],text=True).split()[0])*1024
    plan['ollama_list_after']=subprocess.check_output(['docker','exec','klasifikacija-ollama-1','ollama','list'],text=True)
    retained=allocated(retired)-retired_before
    plan.update(deleted_tags=deleted,ollama_allocated_before=before,ollama_allocated_after=after,
        removed_local_allocated_bytes=freed,retired_metadata_allocated_bytes=retained,
        reclaimed_allocated_bytes=freed+before-after-retained)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    plan=audit();REPORT.parent.mkdir(exist_ok=True);REPORT.write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    if args.apply:plan=apply(plan);REPORT.write_text(json.dumps(plan,indent=2,ensure_ascii=False))
    print(json.dumps({'candidates':len(plan['candidates']),'allowed':sum(c['allowed'] for c in plan['candidates']),
        'blocked':[{k:c[k] for k in ('path','blocked_by')} for c in plan['candidates'] if not c['allowed']],
        'reclaimed_allocated_bytes':plan.get('reclaimed_allocated_bytes')},ensure_ascii=False))
