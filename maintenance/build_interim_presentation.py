"""Five simple slides in the provided LR PowerPoint template; existing cached data only.
Dependencies: python-pptx, matplotlib, numpy, Pillow. No inference or AMCL calls.
"""
import csv,json,re,shutil,sqlite3,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from PIL import Image,ImageDraw,ImageFont
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'predstavitev/vmesna predstavitev'
TEMPLATE=OUT/'predloga/PredlogaZaKoncnoPredstavitevLR.pptx'
BLACK='#222222';GRAY='#555555'
plt.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none'})
plt.rcParams['font.family']='DejaVu Sans'
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=200,facecolor='white');fig.savefig(OUT/(name+'.svg'),facecolor='white');plt.close(fig)
def schema():
 fig,ax=plt.subplots(figsize=(16,6.2));fig.subplots_adjust(0,0,1,1);ax.set(xlim=(0,16),ylim=(0,6.2));ax.axis('off');xs=[.3,4.3,8.3,12.3];w=3.35;h=1.25
 def box(x,y,t,sub):
  ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=BLACK,linewidth=1.3));ax.text(x+w/2,y+.81,t,ha='center',va='center',size=26,weight='bold',color=BLACK);ax.text(x+w/2,y+.31,sub,ha='center',va='center',size=21,color=GRAY)
 def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=22,linewidth=1.5,color=BLACK,shrinkA=6,shrinkB=6))
 for x,t,sub in zip(xs,['RGB','Florence-2','Qwen 3.5','SAM 2'],['slika','opis + okvirji','razred + razlog','maska']):box(x,4.3,t,sub)
 for x,t,sub in zip(xs,['2D LiDAR','Filter točk','Zemljevid','AMCL'],['meritve','projekcija mask','mapping','lokalizacija']):box(x,1.55,t,sub)
 for y in [4.925,2.175]:
  for x,z in zip(xs,xs[1:]):arrow((x+w,y),(z,y))
 arrow((xs[3]+w/2,4.3),(xs[1]+w/2,2.8))
 ax.text(8,3.35,'maske + razredi',ha='center',size=20,color=GRAY,bbox={'facecolor':'white','edgecolor':'none','pad':2})
 ax.text(.35,.55,'Stalno → ohranimo',size=23,color=BLACK);ax.text(5.8,.55,'Premično → filter*',size=23,color=BLACK);ax.text(11.15,.55,'Negotovo → ohranimo',size=23,color=BLACK)
 ax.text(.35,.05,'* Veljavna maska; zaščita prekrivanj.',size=16,color=GRAY)
 save(fig,'shema_pipeline')

# User-selected score: 100 - definite fixed/movable contradictions among valid outputs.
# Predicted uncertain is retained in the denominator, but is not an error.
# Unknown reference labels and invalid model responses are excluded.
c=sqlite3.connect('file:'+str(ROOT/'reviews/manual_review.sqlite3')+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
q='''select m.model,count(*) reference_regions,
 sum(case when c.computed_match is not null then 1 else 0 end) valid,
 sum(case when c.computed_match is not null and c.predicted_classification in ('fixed','movable')
 and c.predicted_classification != r.reference_class then 1 else 0 end) wrong,
 sum(case when c.computed_match is not null and c.predicted_classification='uncertain' then 1 else 0 end) uncertain
 from computed_matches c join mapping m on c.set_id=m.set_id and c.answer_id=m.answer_id
 join regions r on c.set_id=r.set_id and c.region_key=r.region_key
 where m.seed=42 and m.prompt_variant='new' and r.reference_status='rated'
 and r.reference_class in ('fixed','movable') group by m.model'''
rows=[dict(r) for r in c.execute(q)];c.close()
short={'gemma3:27b-it-q4_K_M':'Gemma 3 27B Q4','qwen3.5:27b-q4_K_M':'Qwen 3.5 27B Q4','qwen3.5:4b-bf16':'Qwen 3.5 4B BF16','qwen3.5:4b-q4_K_M':'Qwen 3.5 4B Q4','qwen3.5:9b-bf16':'Qwen 3.5 9B BF16'}
for r in rows:r.update(score_percent=100*(1-r['wrong']/r['valid']),invalid_or_missing=r['reference_regions']-r['valid'])
new=sorted(rows,key=lambda r:(-r['score_percent'],r['model']));assert len(new)==5 and all(r['reference_regions']==376 for r in new)
labels=[[short[r['model']],f'{r["score_percent"]:.2f} %'.replace('.',',')] for r in new]
with (OUT/'tabela_modelov.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['Model','%']);w.writerows(labels)
fig,ax=plt.subplots(figsize=(14,5.2));fig.subplots_adjust(.02,.03,.98,.97);ax.axis('off')
t=ax.table(cellText=labels,colLabels=['Model','%'],cellLoc='left',colLoc='left',colWidths=[.75,.25],bbox=[0,.03,1,.95]);t.auto_set_font_size(False);t.set_fontsize(26)
for (i,j),cell in t.get_celld().items():
 cell.set_facecolor('white' if i else '#EEEEEE');cell.set_edgecolor('#777777');cell.set_linewidth(.6);cell.set_text_props(color=BLACK,weight='bold' if i==0 or (i>0 and new[i-1]['model']=='qwen3.5:27b-q4_K_M') else 'normal')
save(fig,'tabela_modelov');schema()

report=ROOT/'results/lilocbench-box-segmentation-test2/segmentation.json';seg=json.loads(report.read_text());frame=next(f for f in seg['frames'] if f['frame_id']=='view-000044');reg=next(r for r in frame['regions'] if r['region_id']==1)
orig=Image.open(frame['image_path']).convert('RGB');a=np.array(orig);mask=np.asarray(Image.open(report.parent/reg['mask_path']))>0
shutil.copyfile(frame['image_path'],OUT/'okolje_rgb.png');shutil.copyfile(report.parent/reg['mask_path'],OUT/'sam2_maska_skatle.png');shutil.copyfile(ROOT/'docs/lilocbench-filtered-map/maps_comparison.png',OUT/'okolje_zemljevid.png')
outline=orig.copy();ImageDraw.Draw(outline).rectangle(tuple(reg['bbox_xyxy']),outline='black',width=5)
overlay=a.copy();overlay[mask]=(a[mask]*.45+np.array([240,145,45])*.55).astype(np.uint8);overlay=Image.fromarray(overlay)
outline.save(OUT/'florence_okvir.png');overlay.save(OUT/'sam2_prekrivanje.png')
pw,ph=896,672;canvas=Image.new('RGB',(3*pw+120,ph+125),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',59)
for i,(im,title) in enumerate([(orig,'RGB'),(outline,'Florence okvir'),(overlay,'Maska SAM 2')]):
 x=20+i*(pw+40);canvas.paste(im.resize((pw,ph),Image.Resampling.LANCZOS),(x,0));draw.text((x,ph+22),title,font=font,fill='black')
canvas.save(OUT/'primer_segmentacije.png')

prs=Presentation(TEMPLATE);layout=prs.slides[2].slide_layout
# Keep original template masters, layout, logos and 16:9 dimensions.
for sid in list(prs.slides._sldIdLst):prs.part.drop_rel(sid.rId);prs.slides._sldIdLst.remove(sid)
speech=(OUT/'besedilo_5_prosojnic.txt').read_text();talks=re.findall(r'GOVOR:\n(.*?)(?=\n\nPROSOJNICA |\Z)',speech,re.S);assert len(talks)==5

def text(s,x,y,w,h,content,size=21,bold=False,color=BLACK):
 box=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame;tf.word_wrap=True
 for i,line in enumerate(content.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color.lstrip('#'));p.space_after=Pt(12)
 return box

def slide(title,n):
 s=prs.slides.add_slide(layout)
 for sh in list(s.shapes):
  if sh!=s.shapes.title:sh._element.getparent().remove(sh._element)
 titlebox=s.shapes.title;titlebox.text=title
 # Orange title style from the source template.
 for p in titlebox.text_frame.paragraphs:p.font.name='Arial';p.font.size=Pt(28);p.font.bold=True;p.font.color.rgb=RGBColor.from_string('FF9933')
 text(s,9.2,5.15,.4,.25,str(n),11,color=GRAY);s.notes_slide.notes_text_frame.text=talks[n-1].strip();return s

def pic(s,name,x,y,w):return s.shapes.add_picture(str(OUT/name),Inches(x),Inches(y),width=Inches(w))
s=slide('Naloga in podatki',1)
text(s,.5,1.3,4.6,.5,'LILocBench',23,True)
text(s,.5,1.85,4.6,.45,'RGB + 2D LiDAR',21)
text(s,.5,2.5,4.5,1.7,'• Statično okolje\n• Ljudje\n• Prestavljeni predmeti',22)
text(s,.5,4.18,4.5,.55,'Cilj: stabilen zemljevid',23,True)
pic(s,'okolje_rgb.png',5.28,1.42,4.1)
text(s,5.28,4.58,4.1,.35,'Hodnik · kartonske škatle',15,color=GRAY)
s=slide('Izbira modela',2)
table=s.shapes.add_table(6,2,Inches(.5),Inches(1.5),Inches(9),Inches(3.2)).table
for col,w in zip(table.columns,[6.7,2.3]):col.width=Inches(w)
for i,data in enumerate([['Model','%']]+labels):
 for j,val in enumerate(data):
  cell=table.cell(i,j);cell.text=val;cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string('EEEEEE' if i==0 else 'FFFFFF');cell.margin_left=Inches(.13);cell.margin_top=Inches(.06);cell.margin_bottom=Inches(.06)
  for p in cell.text_frame.paragraphs:p.font.name='Arial';p.font.size=Pt(23);p.font.color.rgb=RGBColor.from_string('222222');p.font.bold=(i==0 or (i>0 and new[i-1]['model']=='qwen3.5:27b-q4_K_M'))
s=slide('Pipeline in odločanje',3);pic(s,'shema_pipeline.png',.4,1.25,9.1)
s=slide('Primer segmentacije',4);pic(s,'primer_segmentacije.png',.45,1.62,9.0)
text(s,.5,1.12,9,.4,'Škatla → premično',22,True)
text(s,.5,4.48,9,.4,'lt_changes_0 · view-000044',17,color=GRAY)
text(s,.5,4.91,9,.3,'Barvna maska ≠ odstranjene LiDAR točke',15,color=GRAY)
s=slide('Trenutno stanje in nadaljevanje',5)
text(s,.5,1.35,4.3,.5,'Že deluje',24,True)
text(s,.5,2.03,4.3,2.4,'• Izločanje točk\n• Gradnja zemljevida\n• Lokalizacija z AMCL',21)
text(s,5.2,1.35,4.3,.5,'Naslednji koraki',24,True)
text(s,5.2,2.03,4.3,2.4,'• Maske in projekcija\n• Časovna pokritost\n• Primerjava lokalizacije',21)
text(s,.5,4.6,9,.45,'Rezultati: odvisnost od sekvence',20)
prs.save(OUT/'vmesna_predstavitev.pptx')
(OUT/'sledljivost.json').write_text(json.dumps({'template':{'branch':'feature/configurable-vision-pipeline','path':'templateLaTeX_202627-ppt-doc-teX/PredlogaZaKoncnoPredstavitevLR.pptx','revision':'e7dce765','sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()},'segmentation':{'source_report':str(report.relative_to(ROOT)),'sequence':'lt_changes_0','frame_id':'view-000044','region_id':reg['region_id'],'source_ids':reg['source_ids'],'mask_sha256':reg['mask_sha256']},'manual_evaluation':{'source':'reviews/manual_review.sqlite3','reference_regions':376,'seed':42,'main_prompt':'new','score_formula':'100 * (1 - definite_fixed_movable_errors / valid_responses)','predicted_uncertain':'Not an error; retained in denominator','reference_uncertain':'Excluded','invalid_predictions':'Excluded from score denominator; counts retained in results','metric_note':'Complement of definite error rate, not exact class accuracy','results':rows},'inference_rerun':False,'amcl_rerun':False},ensure_ascii=False,indent=2)+'\n')
(OUT/'PREBERI_ME.txt').write_text('''5 prosojnic; približno 5 minut govora.

vmesna_predstavitev.pptx — glavna predstavitev; govor v opombah.
vmesna_predstavitev.pdf — pregled prosojnic.
besedilo_5_prosojnic.txt — kratke alineje za slide in ločeno govor.
shema_pipeline.png / SVG — osnovna črno-bela shema.
primer_segmentacije.png — isti RGB, Florence okvir in shranjena maska SAM 2.
tabela_modelov.png / SVG / CSV — dejanske vrednosti iz tvojih oznak.
predloga/ — nespremenjena izvirna predloga LR z veje feature/configurable-vision-pipeline.

Slog: izvirna predloga LR, 16:9, logotipi, oranžni naslovi; črno besedilo, osnovne tabele in črno-bela shema. Qwen 3.5 27B je poudarjen kot izbrani model. Tabela uporablja uporabnikov izračun 100 % − delež napačnih odločitev. Napačne so samo zamenjave fixed/movable na veljavnih odgovorih z jasno referenco. Napoved uncertain ni napaka; ostane v imenovalcu. Referenčne uncertain in neocenljive regije izpustimo; neveljavne odgovore izpustimo iz imenovalca. Qwen 27B ima 2 napaki med 364 veljavnimi odgovori (81 uncertain; 12 neveljavnih izpuščenih), kar da 99,45 %. Ta kazalnik ni običajna točnost razvrščanja. Podrobni števci so v sledljivost.json.

Primer: LILocBench, lt_changes_0, view-000044, SAM regija 1, izvorna ID 1 in 4. Barva maske je samo vizualizacija enega shranjenega rezultata, ne odstranitev LiDAR zadetkov.
Sledljivost: sledljivost.json. Brez novih modelnih ali AMCL zagonov.

Vir podatkov/slik: Trekel et al., LILocBench, CC BY-NC-SA 4.0.
https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/
https://creativecommons.org/licenses/by-nc-sa/4.0/

Ponovna izdelava: python3 maintenance/build_interim_presentation.py
Odvisnosti: python-pptx, Matplotlib, NumPy, Pillow.
''')
print('Built 5 slides using original LR template.')
