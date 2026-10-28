"""Thin LOCO orchestration; native runners and text reasoning remain shared."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .experiment import read_json, write_json, sha256
from .image_dataset import load_images
from .thud_experiment import immutable_json
from .thud_extended_common import ROOT, native_selection

OUT = ROOT / 'results/loco'
PUBLIC = ROOT / 'analiza rezultatov vmesna predstavitev/industrija LOCO'
VLM = PUBLIC / '01_vlm'
LLM = PUBLIC / '02_llm'


def prepare():
    # Deliberately do NOT read objects.jsonl/COCO in input preparation.
    images = load_images(ROOT / 'data/loco_test30/discovery.jsonl', limit=30)
    assert len(images) == 30
    selected = []
    for i, image in enumerate(images):
        assert image['regions'] is None and 'objects' not in image['record']
        selected.append({'dataset': 'loco_test30', 'index': i,
            'frame_id': image['record']['frame_id'], 'source_frame_id': image['record']['frame_id'],
            'image_path': image['path'], 'image_sha256': image['image_sha256'],
            'image_size': image['image_size'], 'subset20': False})
    immutable_json(OUT/'inputs.json', selected)
    native_selection(OUT/'inputs.json')
    preservation = OUT/'setup/prior-results-sha.json'
    if not preservation.exists():
        files = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        immutable_json(preservation, {p:sha256((ROOT/p).read_bytes()) for p in files
            if p and (p.startswith('analiza rezultatov vmesna predstavitev/') or p.startswith('reports/') or p.startswith('data/'))
            and not p.startswith(str(PUBLIC.relative_to(ROOT))+'/')})
    immutable_json(OUT/'setup/class-map.json',read_json(ROOT/'configs/loco-class-map.json'))
    write_json(OUT/'setup/preparation.json',{'images':len(selected),'input_keys':sorted(selected[0]),
        'annotations_read':False,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'discovery_sha256':sha256((ROOT/'data/loco_test30/discovery.jsonl').read_bytes()),
        'class_map_sha256':sha256((ROOT/'configs/loco-class-map.json').read_bytes())})
    PUBLIC.mkdir(parents=True,exist_ok=True)
    journal=PUBLIC/'NAPREDEK.md'
    if not journal.exists():journal.write_text('# LOCO: dnevnik napredka\n\n'
        '- Prebrana CODEX_LOCO.md in opis podatkov; uporabljenih vseh 30 RGB slik.\n'
        '- Referenčne oznake niso vhod modelov. Preslikava petih razredov zamrznjena pred inferenco.\n'
        '- Uporaba lokalnih uteži in obstoječih izoliranih slik; brez THUD inferenc ali prepisovanja rezultatov.\n'
        '- Piloti so merjeni r1 položaji; brez dodatnih ogrevalnih generacij. Nalaganje je zabeleženo posebej.\n')
    print('LOCO RGB selection frozen:',len(selected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','vlm-export','freeze','pilot','run','export','runtime','report','verify'))
    args=parser.parse_args()
    os.environ['THUD_PROGRESS_FILE']=str(OUT/'progress.md')
    if args.phase=='prepare':prepare();return
    if args.phase=='vlm-export':
        from .loco_report import export_vlm
        export_vlm();return
    if args.phase in ('runtime','export'):
        from .loco_report import capture_runtime
        capture_runtime()
        if args.phase=='runtime':return
    if args.phase in ('report','verify'):
        from .loco_report import report, verify
        (report if args.phase=='report' else verify)();return
    from . import thud_stage2_presentation as stage
    stage.configure_paths(source=VLM,target=LLM,out=OUT/'llm')
    if args.phase=='freeze':
        stage.freeze()
        from .loco_report import context_policy
        context_policy();return
    stage.execute(args.phase)


if __name__=='__main__':main()
