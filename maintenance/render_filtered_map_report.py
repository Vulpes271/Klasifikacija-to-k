"""Render the checked-in Markdown report; requires reportlab and Pillow. No experiments run."""
import re
from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether

BASE=Path(__file__).resolve().parents[1]/'docs/lilocbench-filtered-map'
FONT=Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DejaVu',str(FONT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(FONT/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
styles=getSampleStyleSheet()
for style in styles.byName.values():style.fontName='DejaVu'
styles['BodyText'].fontSize=9;styles['BodyText'].leading=12;styles['BodyText'].spaceAfter=8
styles['Heading1'].fontSize=17;styles['Heading1'].leading=21
styles['Heading2'].fontSize=13;styles['Heading2'].leading=17;styles['Heading2'].spaceBefore=10
styles['Heading3'].fontSize=10;styles['Heading3'].leading=13
small=ParagraphStyle('small',parent=styles['BodyText'],fontSize=7,leading=9,spaceAfter=0)
caption=ParagraphStyle('caption',parent=styles['BodyText'],fontSize=8,leading=10,spaceAfter=4)
W=A4[0]-80

def para(s,style='BodyText'):
 s=escape(s)
 s=re.sub(r'https://[^\s]+',lambda m:f'<link href="{m[0]}" color="#174d86">{m[0]}</link>',s)
 return Paragraph(s,styles[style])
def image(path,width):
 im=Image(str(BASE/path));im.drawHeight*=width/im.drawWidth;im.drawWidth=width;return im

def render():
 lines=(BASE/'porocilo.md').read_text().splitlines();story=[];i=0
 while i<len(lines):
  line=lines[i].strip();i+=1
  if not line:continue
  if line=='<!-- pagebreak -->':story.append(PageBreak());continue
  if line.startswith('#'):
   level=len(line)-len(line.lstrip('#'));story.append(para(line[level:].strip(),f'Heading{min(level,3)}'));continue
  if line.startswith('|'):
   rows=[]
   while True:
    cells=[s.strip() for s in line.strip('|').split('|')]
    if not all(re.fullmatch('[-: ]+',s) for s in cells):rows.append(cells)
    if i>=len(lines) or not lines[i].strip().startswith('|'):break
    line=lines[i].strip();i+=1
   n=len(rows[0]);widths=[W/n]*n
   if n==6:widths=[36,40]+[(W-76)/4]*4
   if n==3:widths=[W*.28,W*.30,W*.42]
   t=Table([[Paragraph(escape(c),small) for c in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
   t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2eaf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c5cdd4')),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
   story.extend([t,Spacer(1,8)]);continue
  if line.startswith('!['):
   pictures=[re.match(r'!\[(.*?)\]\((.*?)\)',line).groups()]
   while i<len(lines) and lines[i].startswith('!['):pictures.append(re.match(r'!\[(.*?)\]\((.*?)\)',lines[i]).groups());i+=1
   width=(W-10)/len(pictures);cells=[]
   for label,path in pictures:cells.append([image(path,width),Paragraph(escape(label),caption)])
   story.extend([Table([cells],colWidths=[W/len(cells)]*len(cells),style=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),5)]),Spacer(1,8)]);continue
  paragraph=[line]
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','<!--')):paragraph.append(lines[i].strip());i+=1
  story.append(para(' '.join(paragraph)))
 def footer(canvas,doc):
  canvas.setFont('DejaVu',7);canvas.setFillColor(colors.gray);canvas.drawString(40,25,'LILocBench · shranjeni rezultati · 12. 11. 2026');canvas.drawRightString(A4[0]-40,25,str(doc.page))
 doc=SimpleDocTemplate(str(BASE/'porocilo.pdf'),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=32,bottomMargin=40,title='LILocBench: semantično filtriranje LiDAR',author='Klasifikacija-to-k',pageCompression=1)
 doc.build(story,onFirstPage=footer,onLaterPages=footer)
 print(BASE/'porocilo.pdf')
if __name__=='__main__':render()
