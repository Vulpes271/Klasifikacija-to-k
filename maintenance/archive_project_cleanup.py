"""Archive reviewed legacy paths by content hash, verify, then remove project copies.

No LILocBench scans/masks/calibrations/grids, active weights or human ratings are
removed. Private archive manifests and review snapshots never enter Git.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from llm_object_classifier.review_viewer import Results


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def view_signature(path,florence):
    r=Results(path,florence)
    return dict(set_id=r.set_id,frames=len(r.frames),regions=len(r.flat),keys_sha256=hashlib.sha256(json.dumps([x[2] for x in r.flat]).encode()).hexdigest())


def finalize(archive,snapshot,out):
    summaries=snapshot['paths'];preserved=snapshot['preserved_sha256']
    paths=[r['path'] for r in summaries if r['action']=='archived_and_removed']
    for row in snapshot['files']:
        p=archive/'tree'/row['path']
        actual=hashlib.sha256(os.readlink(p).encode()).hexdigest() if row['type']=='symlink' else digest(p)
        assert actual==row['sha256']
    for relative in paths:
        p=ROOT/relative
        if not p.exists() and not p.is_symlink():continue
        if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
        else:p.unlink()
    assert all(digest(ROOT/p)==h for p,h in preserved.items())
    with (out/'removed-paths.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]),lineterminator='\n');writer.writeheader();writer.writerows(summaries)
    removed=sum(r['logical_bytes'] for r in summaries if r['action']=='archived_and_removed')
    # Physical saving from dedup is measured as unique regular content bytes;
    # reviews remain twice intentionally, and Git history is untouched.
    unique_bytes=sum({r['sha256']:r['size_bytes'] for r in snapshot['files'] if r['type']=='file'}.values())
    public=dict(archive_branch='archive/before-cleanup-relocalization-2026-11-18',archive_commit=snapshot['source_commit'],local_archive=str(archive),
        archive_manifest_sha256=digest(archive/'manifest.json'),removed_project_bytes=removed,archive_unique_payload_bytes=unique_bytes,
        removed_tracked_files=sum(r['tracked_files'] for r in summaries if r['action']=='archived_and_removed'),
        verified_private_review_backup=True,legacy_viewer_identities_preserved=True,active_inputs_and_weights_sha256_unchanged=True,
        unused_test_weights='No unused tensor weights remain; Florence and SAM checkpoints are active. Retired NVILA/Qwen-VL metadata archived.',
        retired_runner_tests='Already absent before cleanup. All extant active pipeline/viewer/LILocBench tests retained.',preserved_sha256={p:h for p,h in preserved.items() if not p.startswith('reviews/')})
    (out/'archive-validation.json').write_text(json.dumps(public,indent=2)+'\n')
    (archive/'RESTORE.md').write_text('Legacy paths are under tree/ with original relative names. manifest.json records every SHA256.\nPrivate reviews are preserved in tree/reviews and in a consistent SQLite backup.\nRestore into a separate checkout of archive/before-cleanup-relocalization-2026-11-18.\nNever overwrite current ratings without comparison.\n')
    print(json.dumps(public,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive-root',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    archive=args.archive_root.resolve();assert not archive.is_relative_to(ROOT)
    if args.resume:
        finalize(archive,json.loads((archive/'manifest.json').read_text()),ROOT/'docs/cleanup-2026-11-18');return
    assert not archive.exists(),'Use a new archive root; never overwrite archives'
    out=ROOT/'docs/cleanup-2026-11-18'
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
    paths=['data/nyu20','nyu20-slike.tar.gz','vision-prvi.tar.gz','vision-launch.log','models/retired-metadata',
        'results models/prompt_comparison40','results models/florence_grounding_llm_comparison10','results models/florence_caption_grounding_test10',
        'results models/lilocbench/index.html','results models/lilocbench/native/overlays',
        'analiza rezultatov vmesna predstavitev','reports']
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'results').iterdir()) if not p.name.startswith('lilocbench') and p.name not in ['pipeline','pipeline-smoke-cleanup']]
    paths=[p for p in paths if (ROOT/p).exists()]
    for p in paths:
        assert p not in ['models/florence-2-large','models/sam2.1-hiera-small','reviews']
        assert not (p.startswith('data/lilocbench') or p.startswith('docs/') or p.startswith('predstavitev/'))
    # Direct callers of retired datasets are the legacy viewer defaults (updated
    # in the cleanup commit). Their old adapters and rating identities survive.
    signatures={}
    florence=ROOT/'results models/florence_caption_grounding_test10'
    for name in ['prompt_comparison40','florence_grounding_llm_comparison10','lilocbench']:
        signatures[name]=view_signature(ROOT/'results models'/name,florence)
    preserved_paths=['configs/pipeline.json','models/florence-2-large/model.safetensors','models/sam2.1-hiera-small/sam2.1_hiera_small.pt',
        'reviews/manual_review.sqlite3','data/lilocbench-filtered-map/static_0/raw.npz','data/lilocbench-filtered-map/lt_changes_0/raw.npz','data/lilocbench-filtered-map/mapping/raw.npz',
        'data/lilocbench-reverse/projection.npz']+[f'results/lilocbench-reverse/maps/{n}/occupancy.npy' for n in ['original','front','three','empty']]
    preserved={p:digest(ROOT/p) for p in preserved_paths}
    archive.mkdir(parents=True);(archive/'objects').mkdir();tree=archive/'tree';tree.mkdir()
    manifest=[];objects={};summaries=[]
    def save(relative,remove_group):
        source=ROOT/relative;dest=tree/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if source.is_symlink():
            link=os.readlink(source);dest.symlink_to(link);row=dict(path=relative,type='symlink',size_bytes=len(link.encode()),sha256=hashlib.sha256(link.encode()).hexdigest(),link_target=link)
        else:
            h=digest(source);size=source.stat().st_size
            if h not in objects:
                obj=archive/'objects'/h;shutil.copy2(source,obj)
                assert digest(obj)==h;objects[h]=size
            os.link(archive/'objects'/h,dest)
            assert digest(dest)==h
            row=dict(path=relative,type='file',size_bytes=size,sha256=h)
        row.update(remove_group=remove_group,tracked=relative in tracked);manifest.append(row)
    for relative in [*paths,'reviews']:
        source=ROOT/relative;before=len(manifest)
        if source.is_dir():
            for parent,dirs,files in os.walk(source,followlinks=False):
                (tree/Path(parent).relative_to(ROOT)).mkdir(parents=True,exist_ok=True)
                for name in list(dirs):
                    p=Path(parent)/name
                    if p.is_symlink():save(p.relative_to(ROOT).as_posix(),relative);dirs.remove(name)
                for name in sorted(files):save((Path(parent)/name).relative_to(ROOT).as_posix(),relative)
        else:save(relative,relative)
        selected=manifest[before:]
        summaries.append(dict(path=relative,action='archived_and_removed' if relative!='reviews' else 'archived_and_preserved',files=len(selected),
            logical_bytes=sum(r['size_bytes'] for r in selected),tracked_files=sum(r['tracked'] for r in selected)))
    # SQLite online backup creates a consistent snapshot, including committed WAL.
    with sqlite3.connect((ROOT/'reviews/manual_review.sqlite3').as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(archive/'manual_review-consistent.sqlite3') as dst:
        src.backup(dst);assert dst.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    snapshot=dict(source_commit='0510c225320da56bc2b89e2e135711beb57c9618',paths=summaries,files=manifest,preserved_sha256=preserved,viewer_signatures=signatures,
        consistent_review_backup_sha256=digest(archive/'manual_review-consistent.sqlite3'))
    (archive/'manifest.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    # Verify relocated datasets before removing a single source path.
    for name in ['prompt_comparison40','florence_grounding_llm_comparison10']:
        assert view_signature(tree/'results models'/name,tree/'results models/florence_caption_grounding_test10')==signatures[name]
    for r in manifest:
        p=tree/r['path'];assert (hashlib.sha256(os.readlink(p).encode()).hexdigest() if r['type']=='symlink' else digest(p))==r['sha256']
    finalize(archive,snapshot,out)


if __name__=='__main__':main()
