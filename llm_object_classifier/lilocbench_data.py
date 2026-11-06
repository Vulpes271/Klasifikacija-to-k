"""Bounded, resumable HTTP-range extraction of necessary LILocBench ZIP members."""
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

BASE='https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/'
LIMIT=12_000_000_000


class Downloads:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.ledger=self.folder/'downloads.json'
        self.rows=json.loads(self.ledger.read_text()) if self.ledger.exists() else []

    def get(self,url,start=None,count=None):
        import hashlib
        key=hashlib.sha256(f'{url}:{start}:{count}'.encode()).hexdigest()
        target=self.folder/key
        if target.exists():return target.read_bytes()
        used=sum(r['bytes'] for r in self.rows)
        budget=LIMIT-used
        if budget<=0 or (count is not None and count>budget):raise RuntimeError('12 GB download cap exceeded')
        headers={'User-Agent':'LILocBench-pilot/1.0'}
        if start is not None:headers['Range']=f'bytes={start}-{start+count-1}'
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:
            if start is not None and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):
                raise RuntimeError('Server does not honor exact byte ranges; refused whole archive')
            data=response.read(min(budget,count if count is not None else 50_000_000)+1)
            if len(data)>budget or (count is not None and len(data)!=count):raise RuntimeError('Unexpected download size')
            self.rows.append({'url':url,'start':start,'bytes':len(data),'etag':response.headers.get('ETag')})
            self.ledger.write_text(json.dumps(self.rows,indent=2)+'\n')
        temporary=target.with_suffix('.part');temporary.write_bytes(data);temporary.replace(target)
        return data


class RemoteZip(io.RawIOBase):
    def __init__(self,downloads,url,size):self.downloads=downloads;self.url=url;self.size=size;self.position=0
    def seekable(self):return True
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def tell(self):return self.position
    def read(self,size=-1):
        size=min(self.size-self.position,size if size>=0 else self.size-self.position)
        if size<=0:return b''
        data=self.downloads.get(self.url,self.position,size);self.position+=len(data);return data


def prepare_data(data,sequence,stored):
    """Reuse local GT/calibration; extract no mapping/depth/rear-camera archive members."""
    data=Path(data);stored=Path(stored)
    if sequence!='lt_changes_0':raise ValueError('This pilot only downloads the authorized lt_changes_0 sequence')
    seq=data/sequence;seq.mkdir(parents=True,exist_ok=True)
    required=['transformations.yaml','camera_front/color/intrinsics.yaml','laser_scan_front/laser_specs.yaml',
              'laser_scan_front/ranges.txt','dingo_velocity_controller/odom.txt']
    downloads=Downloads(data/'http-cache')
    local=stored/'metadata'/sequence
    for name in ['gt_poses.txt','transformations.yaml','camera_front/color/intrinsics.yaml']:
        target=seq/name;existing=local/name.replace('/','--')
        if not target.exists() and existing.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(existing,target)
    if not (seq/'gt_poses.txt').exists():
        (seq/'gt_poses.txt').write_bytes(downloads.get(BASE+f'data/gt_poses_public/{sequence}/gt_poses.txt'))
    missing=[name for name in required if not (seq/name).exists()]
    if missing:
        archive_size=10_008_762_359
        url=BASE+f'data/data_files_zipped/{sequence}.zip'
        with zipfile.ZipFile(RemoteZip(downloads,url,archive_size)) as archive:
            # Keep a small index locally for auditable membership and extraction size.
            members={i.filename:i for i in archive.infolist()}
            (data/'archive-index.json').write_text(json.dumps({n:{'bytes':i.file_size,'compressed_bytes':i.compress_size} for n,i in members.items()},indent=2))
            missing_paths=[name for name in missing if name not in members]
            if missing_paths:raise FileNotFoundError(f'Archive members missing: {missing_paths}')
            needed=sum(members[name].file_size for name in missing)
            if shutil.disk_usage(data).free < needed+1_000_000_000:raise RuntimeError('Insufficient extraction space')
            for name in missing:
                target=seq/name;target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix(target.suffix+'.part')
                with archive.open(name) as incoming,temporary.open('wb') as outgoing:shutil.copyfileobj(incoming,outgoing)
                temporary.replace(target);print(f'Extracted {name}: {target.stat().st_size} bytes',flush=True)
    mapdir=data/'map';mapdir.mkdir(exist_ok=True)
    if not list(mapdir.glob('*.yaml')):
        with zipfile.ZipFile(io.BytesIO(downloads.get(BASE+'data/map_office.zip'))) as archive:
            for info in archive.infolist():
                if Path(info.filename).suffix in {'.yaml','.pgm','.png'}:
                    (mapdir/Path(info.filename).name).write_bytes(archive.read(info))
    evaluator=data/'reference-code/evaluate.py'
    if not evaluator.exists():
        evaluator.parent.mkdir(exist_ok=True)
        evaluator.write_bytes(downloads.get('https://raw.githubusercontent.com/PRBonn/LILocBench/main/src/evaluate.py'))
    return seq
