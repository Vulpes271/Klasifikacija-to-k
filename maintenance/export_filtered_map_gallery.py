"""Export existing gallery/cache only. Requires beautifulsoup4, numpy; no inference or AMCL calls."""
import json,shutil,hashlib
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
root=Path(__file__).resolve().parents[1]; dest=root/'docs/lilocbench-filtered-map/gallery'
shutil.copytree(root/'results/lilocbench-filtered-map/gallery', dest, dirs_exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
def write(p,a):Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
soup=BeautifulSoup((dest/'index.html').read_text(),'html.parser')
soup.html if False else None
soup.title.string='LILocBench — galerija filtriranja zemljevida'
soup.style.string += '\n.badge{display:inline-block;padding:.3em .7em;background:#dceaf6;border-radius:4px} .room{background:#ffe2ca} table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.4em;text-align:left}figure{margin:1em 0}figcaption{font-weight:bold}nav{padding:1em;background:white} .masks img{width:120px;background:#222} .masks figure{display:inline-block;vertical-align:top;max-width:180px;font-size:12px} a{color:#174d86} .missing{color:#873a12}'
def fragment(s):return BeautifulSoup(s,'html.parser')
nav=fragment('<nav><b>Primeri lt_changes_0:</b> <a href="#lt_changes_0-04751">Hodnik, sken 4751</a> · <a href="#lt_changes_0-04946">Spodnja desna soba, sken 4946</a> · <a href="../porocilo.pdf">PDF poročilo</a><p>Razdelitev po GT: hodnik y ≥ −5 m; soba x ∈ [5, 13,8], y ∈ [−10,8, −5) m. ID skena je ničelni indeks v raw.npz. Slike, odzivi in maske so vključeni z relativnimi potmi; povezava deluje tudi prek file://.</p><p>RGB je izvirni pogled brez oznak (obstoječi JPEG galerije). SAM 2.1: rdeča = učinkovita maska movable po zaščiti prekrivanj; zelena = fixed; siva = uncertain/manjkajoče/konflikt. Binarne maske so na voljo ob vsakem primeru. Pri lt_changes_0 je maska samo diagnostična: odstranjenih zadetkov je 0. Rdeče točke so dejansko odstranjeni zadetki samo pri mapping. Položaji točk na zemljevidu uporabljajo GT za diagnostiko.</p><p>Florence in LLM besedila so shranjeni modelni izhodi, ne ročno potrjena resnica. Manjkajoč odziv je označen za točno ta RGB; odzivov bližnjih slik ne pripisujemo izbranemu skenu.</p></nav>')
soup.h1.insert_after(nav)
statuses={}
for p in ['data/lilocbench-filtered-map/mapping/mask-status.json','data/lilocbench-full/mask-status.json','results/lilocbench-rgb-rate/rgb-plan-5hz.json']:
 for item in read(p):statuses[item['frame_id']]=item
reports={};manifest=[]
raw={seq:np.load(root/'data/lilocbench-filtered-map'/seq/'raw.npz') for seq in ['mapping','lt_changes_0']}
from html import escape as e
for card in soup.find_all('article'):
 stem=card.find('img')['src'].removesuffix('-rgb.jpg');meta=read(dest/(stem+'.json'));seq=meta['sequence'];index=int(stem.rsplit('-',1)[1]);fid=meta['frame_id'];item=statuses[fid];report=Path(item['mask_report']);gt=raw[seq]['gt'][index]
 area='Gradnja zemljevida — prispevki v območju sobe' if seq=='mapping' else ('Hodnik' if gt[1]>=-5 else 'Spodnja desna soba')
 card['id']=stem;card.h2.string=f'{area} | {seq} | sken {index} | čas {meta["scan_timestamp"]:.6f}'
 badge=fragment(f'<p class="badge {"room" if seq=="lt_changes_0" and gt[1]<-5 else ""}">{area} · GT x={gt[0]:.3f}, y={gt[1]:.3f} m · RGB {meta["rgb_timestamp"]:.6f} · {fid}</p>');card.h2.insert_after(badge)
 if seq=='lt_changes_0':card.h2.insert_after(fragment('<p><a href="#mapping-00012">Primer dejansko odstranjenih zadetkov pri gradnji zemljevida: mapping, sken 12 (301 zadetkov)</a>. To je drug čas in druga vožnja; ne gre za odstranitve prikazanega lokalizacijskega skena.</p>'))
 for img,label in zip(list(card.find_all('img')),['Originalni RGB','SAM 2.1 maske, Florence okvirji in Qwen oznake','LiDAR: isti zadetki na originalnem in filtriranem zemljevidu']):
  img['alt']=f'{stem}: {label}';fig=soup.new_tag('figure');img.wrap(fig);caption=soup.new_tag('figcaption');caption.string=label;fig.append(caption)
 ev=dest/'evidence'/stem;ev.mkdir(parents=True,exist_ok=True)
 if report not in reports:reports[report]={x['frame_id']:x for x in read(report)['frames']}
 frame=reports[report][fid];sources=[s for r in frame['regions'] for s in r['sources']]
 inf=report.parent.parent/'inference';frozen=inf/'frozen'/(fid+'.json');response=inf/'responses/qwen'/(fid+'.json')
 if not response.exists():
  opts=list((root/'data').glob('lilocbench-*/**/responses/*/'+fid+'.json'));opts=[p for p in opts if '/gemma/' not in str(p)];response=opts[0] if opts else None
 flor=read(frozen) if frozen.exists() else {'frame_id':fid,'status':'native_frozen_unavailable','objects':[s['object'] for s in sources]}
 write(ev/'florence.json',flor)
 caption=flor.get('parsed',{}).get('scene_description','Opis scene v izvornem cache za ta prikaz ni na voljo; oznake in okvirji so spodaj.')
 part=f'<h3>Shranjeni odzivi in oznake</h3><p><b>Florence-2 large:</b> {e(caption)}</p><p><a href="evidence/{stem}/florence.json">Shranjeni Florence JSON</a> · <a href="{stem}.json">Časi in podatki skena</a></p>'
 models={}
 for model,path in [('Qwen3.5 27B',response),('Gemma3 27B',root/'data/lilocbench-qwen-gemma/gemma/responses/gemma'/(fid+'.json'))]:
  if path and path.exists():
   obj=read(path);outname='qwen.json' if model.startswith('Qwen') else 'gemma.json';shutil.copyfile(path,ev/outname)
   models[model]={'status':obj.get('status'),'file':f'evidence/{stem}/{outname}'}
   part+=f'<p>{model}: {e(str(obj.get("status")))} — <a href="evidence/{stem}/{outname}">celoten shranjeni odziv, validacija in podpis</a>.</p>'
  else:
   models[model]={'status':'missing'};part+=f'<p class="missing">{model}: odziv za ta RGB manjka; ni bil ponovno pridobljen.</p>'
 part+='<table><thead><tr><th>Izvorni ID</th><th>Florence oznaka</th><th>Qwen razred</th><th>Shranjena obrazložitev</th></tr></thead><tbody>'
 for s in sources:
  obj=s['object'];dec=s['llm']['models'].get('qwen35-27b-q4',{});part+='<tr>'+''.join('<td>'+e(str(v))+'</td>' for v in [obj['id'],obj['label'],dec.get('classification') or 'Odziv manjka',dec.get('reason') or 'Obrazložitev manjka'])+'</tr>'
 part+='</tbody></table><details><summary>Binarne maske SAM 2.1 po regijah</summary><div class="masks">'
 exported=[]
 for r in frame['regions']:
  src=report.parent/r['mask_path'];target=ev/f'region-{r["region_id"]:04d}.png';shutil.copyfile(src,target);rel=target.relative_to(dest).as_posix()
  part+=f'<figure><a href="{rel}"><img loading="lazy" src="{rel}" alt="SAM regija {r["region_id"]}, izvorni ID {e(str(r["source_ids"]))}"></a><figcaption>{e(str(r["source_ids"]))}: {e(r["label"])}</figcaption></figure>'
  exported.append({k:r[k] for k in ['region_id','source_ids','label','bbox_xyxy','mask_sha256','sam_score','mask_status']})
  exported[-1]['mask_path']=rel;exported[-1]['sources']=r['sources']
 part+='</div></details>'
 note=f'Dejansko odstranjenih LiDAR zadetkov: {meta["removed_hits"]}. '+('Filter je bil uporabljen med mapping.' if seq=='mapping' else 'Filter med lokalizacijo NI bil uporabljen. Rdeča SAM maska ne pomeni odstranjenih lokalizacijskih zadetkov.')
 part=f'<p><b>{note}</b> Časovna veljavnost maske: {meta["mask_valid"]}; odmik od sredine skena {meta["mask_time_offset_s"]*1000:.2f} ms.</p>'+part
 card.append(fragment(part))
 write(ev/'sam-and-decisions.json',{'frame_id':fid,'regions':exported})
 meta.update(scan_id=index,area=area,gt_xy=gt[:2].tolist(),model_responses=models,sam_evidence=f'evidence/{stem}/sam-and-decisions.json');write(dest/(stem+'.json'),meta)
 manifest.append({'example':stem,'source_mask_report':str(report.relative_to(root)),'scan_id':index,'area':area,'models':models})
soup.append(fragment('<footer><p>Vir: Trekel et al., LILocBench (IROS 2025). RGB, zemljevidi in izpeljane slike: <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>. <a href="https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/">Podatkovna zbirka</a>. Izvoz uporablja le obstoječe rezultate in cache.</p></footer>'))
(dest/'index.html').write_text(str(soup)+'\n')
write(dest/'evidence-index.json',manifest)
print('Exported',len(manifest),'cards')
