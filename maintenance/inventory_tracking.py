"""Inventory tracked paths and local replay inputs; never delete anything.

Stage intended report files first, including placeholders for the inventory files,
then run this command and stage the completed inventories again.
"""
import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from maintenance.run_scan_to_map import csvwrite


def dependencies(entry):
    seen=set();pending=[entry]
    while pending:
        path=pending.pop()
        if path in seen or not path.is_file():continue
        seen.add(path);module='.'.join(path.relative_to(ROOT).with_suffix('').parts)
        package=module.split('.')[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                parent=package[:len(package)-node.level+1] if node.level else []
                name='.'.join(parent+([node.module] if node.module else []))
                names=[name]+[name+'.'+a.name for a in node.names]
            for name in names:
                p=ROOT.joinpath(*name.split('.'))
                if p.with_suffix('.py').is_file():pending.append(p.with_suffix('.py'))
                if (p/'__init__.py').is_file():pending.append(p/'__init__.py')
                for parent in p.parents:
                    if parent==ROOT:break
                    if (parent/'__init__.py').is_file():pending.append(parent/'__init__.py')
    return {p.relative_to(ROOT).as_posix() for p in seen}


def tree_bytes(path):
    if path.is_file() or path.is_symlink():return path.lstat().st_size
    total=0
    for parent,dirs,files in os.walk(path,followlinks=False):
        for name in files:
            p=Path(parent)/name
            try:total+=p.lstat().st_size
            except FileNotFoundError:pass
    return total


def main():
    cfg=json.loads((ROOT/'configs/lilocbench-tracking.json').read_text());base=json.loads((ROOT/cfg['base_config']).read_text());out=ROOT/cfg['output']
    runtime=dependencies(ROOT/'maintenance/run_sequential_scan_to_map.py')
    tracked=[p.decode() for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).split(b'\0') if p]
    expected=[out/'inventory-files.csv',out/'inventory-directories.csv',out/'local-inputs.csv',out/'INVENTORY.md']
    assert all(p.relative_to(ROOT).as_posix() in tracked for p in expected),'Stage inventory placeholders first'
    def role(path):
        if path in runtime or path in ['configs/lilocbench-tracking.json',cfg['base_config']]:return 'potrebno za zagon',False
        if path.startswith(cfg['output']+'/'):return 'rezultat in inventar tega poskusa',False
        if path.startswith('tests/test_sequential_scan_to_map') or path in ['tests/test_scan_to_map.py','tests/test_reverse_short.py']:return 'preverjanje izvedbe',False
        if path.startswith('results models/') or (path.startswith('docs/lilocbench-') and not path.startswith('docs/lilocbench-scan-to-map/')):return 'starejsi rezultati; kandidat za arhiviranje po pregledu referenc',True
        if path.startswith('docs/lilocbench-scan-to-map/'):return 'izhodi predhodnega primerjalnega poskusa',False
        if path.startswith('requirements'):return 'odvisnosti okolja; preveriti glede na uporabljene uvoze',False
        return 'druga vsebina projekta; ni ocenjena za brisanje',False
    local=[]
    inputs=[base['raw']+'/raw.npz',base['raw']+'/raw.json',base['calibration']]+[base['maps']+'/'+name+'/occupancy.npy' for name in cfg['variants']]
    for path in inputs:
        p=ROOT/path;local.append(dict(path=path,size_bytes=tree_bytes(p) if p.exists() else 0,exists=p.exists(),purpose='potrebno za polni zagon in ponovitev',cleanup_candidate=False))
    for root in ['data','results','models','.venv']:
        directory=ROOT/root
        if not directory.is_dir():continue
        children=sorted(directory.iterdir()) if root in ['data','results'] else [directory]
        for p in children:
            if p.name=='README.md':continue
            relative=p.relative_to(ROOT).as_posix()
            required=any(i==relative or i.startswith(relative+'/') for i in inputs)
            purpose='vsebuje potrebne lokalne vhode; pred ciscenjem jih lociti' if required else 'lokalni cache/rezultati; pregledati pred arhiviranjem'
            if relative==cfg['cache']:purpose='podvojeni CSV in revizije; uporablja jih --stage render, ni nujno hraniti za nov izracun'
            local.append(dict(path=relative,size_bytes=tree_bytes(p),exists=True,purpose=purpose,cleanup_candidate=not required))
    csvwrite(out/'local-inputs.csv',local)
    previous=None
    for iteration in range(10):
        files=[];directories={}
        for path in tracked:
            p=ROOT/path;size=p.lstat().st_size if p.exists() or p.is_symlink() else 0;label,candidate=role(path)
            files.append(dict(path=path,size_bytes=size,exists=p.exists() or p.is_symlink(),purpose=label,cleanup_review_candidate=candidate))
            for parent in [Path('.'),*list(Path(path).parents)[:-1]]:
                key=parent.as_posix();row=directories.setdefault(key,dict(path=key,files=0,size_bytes=0,cleanup_review_files=0))
                row['files']+=1;row['size_bytes']+=size;row['cleanup_review_files']+=int(candidate)
        csvwrite(out/'inventory-files.csv',files);csvwrite(out/'inventory-directories.csv',[directories[k] for k in sorted(directories)])
        top=[d for d in directories.values() if '/' not in d['path'] and d['path']!='.'];top.sort(key=lambda r:-r['size_bytes'])
        lines=['# Inventar repozitorija in lokalnih vhodov','',
            f"Posnetek sledenih poti iz `git ls-files` pred objavo tega poskusa; izhodišče `{cfg['source_commit']}`. Velikosti so logični bajti datotek v delovnem drevesu, ne velikost Git zgodovine ali zasedba diskovnih blokov. Vključene so nove, že dodane datoteke poskusa. Seznam vsebuje {len(files)} datotek in {len(directories)} map z agregati (nadrejene mape vključujejo podmape). Ni bilo brisanja.",
            '', '[Vsaka sledena datoteka z velikostjo in namenom](inventory-files.csv) · [Vsaka mapa z vsoto velikosti](inventory-directories.csv) · [Lokalni vhodi in cache](local-inputs.csv)',
            '', '| Mapa na vrhu | Datoteke | Velikost [MiB] |','|---|---:|---:|']
        lines += [f"| {r['path']} | {r['files']} | {r['size_bytes']/2**20:.2f} |" for r in top]
        lines += ['','## Za zagon in ponovitev','',
            'Obvezna koda je označena `potrebno za zagon` v inventarju; seznam nastane iz lokalnih Python uvozov začetne skripte. Potrebni sta obe konfiguraciji, `requirements-localization.txt` opisuje del okolja; dejansko uporabljeni paketi so NumPy, SciPy, PyYAML in Matplotlib. Nastavitve in ukazi so v [README](README.md).',
            '', 'Surovi `static_0/raw.npz`, opis laserja `raw.json`, kalibracija `transformations.yaml` ter mreži `three` in `empty` morajo biti lokalno pripravljeni. Vse natančne poti, velikosti in razpoložljivost so v `local-inputs.csv`. Ti vhodi niso sledeni v Gitu. Za branje rezultatov so dovolj README, objavljeni CSV in PNG. Za ponovni izris brez lokalizacije potrebujemo lokalne CSV po kombinacijah in revizije v `data/lilocbench-tracking/full` ter mreži; `--stage render` ne izvaja sledenja.',
            '', '## Kandidati za ločeno odločitev o čiščenju','',
            'Starejši `results models/` in starejša poročila LILocBench so označeni kot kandidati za pregled arhiviranja. Niso samodejno odvečni: vsebujejo izvirne rezultate in so lahko povezani iz drugih poročil. Pred premikom preveriti povezave in ohraniti arhiv.',
            '', 'Lokalni maskovni/LLM cache, starejši izhodi AMCL, začasne projekcije in uteži niso potrebni za ta sekvenčni test, če ohranimo oba zemljevida, testne skene in kalibracijo. `data/lilocbench-reverse` vsebuje tudi geometrijo in maske za druge poskuse, zato brisanje ni priporočilo brez pregleda. Nov `data/lilocbench-tracking` podvaja objavljene meritve in hrani tehnični preizkus; po arhiviranju revizij je kandidat za čiščenje, vendar je trenutno potreben za ukaz `--stage render`.',
            '', 'Mapi `results/lilocbench-reverse` in `data/lilocbench-filtered-map` vsebujeta potrebne vhode skupaj z drugimi izhodi. Nikoli ju ne odstraniti v celoti samo na podlagi skupne velikosti. CSV loči konkretne potrebne datoteke. O drugih vsebinah projekta ta inventar ne podaja ocene za brisanje.',
            '', 'Osvežitev po dodajanju nameravanih datotek v Git: `python3 maintenance/inventory_tracking.py`, nato ponovno dodaj posodobljene inventarje. Skripta samo bere in piše svoje poročilo, ničesar ne briše.']
        (out/'INVENTORY.md').write_text('\n'.join(lines)+'\n')
        current=tuple((p,(ROOT/p).lstat().st_size if (ROOT/p).exists() else 0) for p in tracked)
        if current==previous:break
        previous=current
    else:raise RuntimeError('Inventory size fixed point did not converge')
    print('Inventory complete:',len(files),'files;',directories['.']['size_bytes'],'tracked bytes;',len(runtime),'runtime modules')


if __name__=='__main__':main()
