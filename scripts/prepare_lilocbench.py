"""Selective ZIP/ZIP64 RGB pairing; ground-truth poses never enter model prompts."""
import argparse
import bisect
import io
import json
import math
from pathlib import Path
import struct
import urllib.request
import urllib.parse
import zipfile
import zlib
import hashlib
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results models/lilocbench'
CACHE=ROOT/'results/lilocbench/cache'
SITE='https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/'


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


class Transfers:
    def __init__(self):
        self.path=OUT/'download-ledger.json'
        self.rows=json.loads(self.path.read_text()) if self.path.exists() else []

    def get(self,url,start=None,length=None,limit=6*1024**2):
        from hashlib import sha256
        key=sha256(f'{url}|{start}|{length}'.encode()).hexdigest();path=CACHE/key
        if path.exists():return path.read_bytes()
        headers={'Accept-Encoding':'identity'}
        if start is not None:headers['Range']=f'bytes={start}-{start+length-1}'
        row={'url':url,'start':start,'requested_bytes':length,'payload_bytes':0}
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
                row.update(status=response.status,content_range=response.headers.get('Content-Range'),etag=response.headers.get('ETag'))
                if start is not None:
                    expected=f'bytes {start}-{start+length-1}/'
                    if response.status!=206 or not response.headers.get('Content-Range','').startswith(expected):
                        raise ValueError('Range not honored; aborted before reading archive body')
                data=response.read((length if start is not None else limit)+1);row['payload_bytes']=len(data)
                if start is not None and len(data)!=length:raise ValueError('Incomplete/oversized range')
                if start is None and len(data)>limit:raise ValueError('Small metadata download exceeded limit')
        except Exception as exc:
            row['error']=str(exc);self.rows.append(row);write(self.path,self.rows);raise
        self.rows.append(row);write(self.path,self.rows)
        CACHE.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return data


class RemoteZip(io.RawIOBase):
    """Same range-read approach as prepare_loco_sample, parameterized and ZIP64-safe."""
    def __init__(self,url,transfers):
        self.url=url;self.transfers=transfers;self.position=0
        transfers.get(url,0,1)
        record=next(r for r in transfers.rows if r['url']==url and r['start']==0 and r['requested_bytes']==1 and 'error' not in r)
        self.size=int(record['content_range'].split('/')[-1])

    def tell(self):return self.position
    def seekable(self):return True
    def readable(self):return True
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def read(self,length=-1):
        length=self.size-self.position if length<0 else min(length,self.size-self.position)
        if length<=0:return b''
        if length>128*1024**2:raise ValueError('Refusing unexpectedly large ZIP index read')
        data=self.transfers.get(self.url,self.position,length);self.position+=length;return data


def member(stream,info):
    header=stream.transfers.get(stream.url,info.header_offset,30)
    values=struct.unpack('<4s5H3I2H',header)
    if values[0]!=b'PK\x03\x04':raise ValueError('Bad local ZIP header')
    start=info.header_offset+30+values[-2]+values[-1]
    packed=stream.transfers.get(stream.url,start,info.compress_size)
    raw=zlib.decompress(packed,-15) if info.compress_type==8 else packed if info.compress_type==0 else None
    if raw is None or len(raw)!=info.file_size or zlib.crc32(raw)&0xffffffff!=info.CRC:raise ValueError('Member checksum mismatch')
    return raw


def inspect():
    from html.parser import HTMLParser
    OUT.mkdir(parents=True,exist_ok=True);transfers=Transfers()
    page=transfers.get(SITE)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.links.append(urllib.parse.urljoin(SITE,dict(attrs).get('href','')))
    parser=Links();parser.feed(page.decode());(OUT/'source.html').write_bytes(page)
    sources={}
    for seq in ('mapping','lt_changes_0'):
        archive=next(u for u in parser.links if u.endswith('/data_files_zipped/'+seq+'.zip'))
        gt=next(u for u in parser.links if u.endswith('/gt_poses_public/'+seq+'/gt_poses.txt'))
        folder=OUT/'metadata'/seq;folder.mkdir(parents=True,exist_ok=True)
        poses=transfers.get(gt);(folder/'gt_poses.txt').write_bytes(poses)
        stream=RemoteZip(archive,transfers)
        with zipfile.ZipFile(stream) as z:
            infos=z.infolist()
            index=[{'path':i.filename,'offset':i.header_offset,'compressed_bytes':i.compress_size,'bytes':i.file_size,'crc32':i.CRC,'compression':i.compress_type} for i in infos]
            write(CACHE.parent/'indices'/(seq+'.json'),index)
            write(folder/'archive-index-summary.json',{'entries':len(index),'archive_bytes':stream.size,
                'zip64_offsets_present':any(i.header_offset>2**32 for i in infos)})
            calibration=[i for i in infos if i.filename.endswith('transformations.yaml') or '/color/intrinsics.yaml' in i.filename or i.filename.endswith('extrinsics_depth_to_color.yaml')]
            for i in calibration:
                (folder/Path(i.filename).name if i.filename.endswith('transformations.yaml') else folder/('/'.join(i.filename.split('/')[-3:]).replace('/','--'))).write_bytes(member(stream,i))
            counts=Counter('/'.join(i.filename.split('/')[:-3]) for i in infos if '/color/images/' in i.filename and i.filename.endswith('.png'))
            print(seq,'archive bytes',stream.size,'RGB cameras',dict(counts),'calibrations',[i.filename for i in calibration],flush=True)
        sources[seq]={'archive':archive,'ground_truth':gt,'archive_bytes':stream.size}
    write(OUT/'sources.json',{'official_site':SITE,'sequences':sources,'license':'https://creativecommons.org/licenses/by-nc-sa/4.0/'})


def archive_index(seq):
    local=CACHE.parent/'indices'/(seq+'.json');old=OUT/'metadata'/seq/'archive-index.json'
    if old.exists():
        local.parent.mkdir(parents=True,exist_ok=True);shutil.move(old,local)
    return json.loads(local.read_text())


def camera_transform(seq,camera='camera_front'):
    from scipy.spatial.transform import Rotation
    edges=yaml.safe_load((OUT/'metadata'/seq/'transformations.yaml').read_text())
    by_child={e['child']:e for e in edges};node=camera+'_color_optical_frame';chain=[]
    while node!='base_link':
        edge=by_child[node];chain.append(edge);node=edge['parent']
    position=np.zeros(3);rotation=Rotation.identity()
    for edge in reversed(chain):
        t=np.array([edge['translation'][k] for k in ('x','y','z')]);q=[edge['rotation'][k] for k in ('x','y','z','w')]
        position+=rotation.apply(t);rotation=rotation*Rotation.from_quat(q)
    forward=rotation.apply([0,0,1])
    if camera=='camera_front' and forward[0]<.9:raise ValueError('Front RGB orientation not confirmed by calibration')
    return position,rotation,{'frame':camera+'_color_optical_frame','translation_xyz':position.tolist(),
        'quaternion_xyzw':rotation.as_quat().tolist(),'optical_forward_in_base':forward.tolist(),
        'chain':list(reversed(chain))}


def views(seq,index,camera='camera_front'):
    from scipy.spatial.transform import Rotation,Slerp
    poses=np.loadtxt(OUT/'metadata'/seq/'gt_poses.txt')
    if poses.shape[1]!=8 or not np.all(np.isfinite(poses)) or not np.all(np.diff(poses[:,0])>0):raise ValueError('Invalid TUM ground truth')
    times=poses[:,0];tbc,rbc,cal=camera_transform(seq,camera)
    slerp=Slerp(times,Rotation.from_quat(poses[:,4:8]));arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(poses[:,1:4],axis=0),axis=1))]
    names=sorted((r for r in index if f'{camera}/color/images/' in r['path'] and r['path'].endswith('.png')),
                 key=lambda r:float(Path(r['path']).stem))
    selected=[];last=-float('inf');excluded=Counter()
    for member in names:
        t=float(Path(member['path']).stem)
        if t-last<.5:continue # Select by poses from timestamp-thinned RGB metadata, not image contents.
        last=t;i=int(np.searchsorted(times,t))
        if i==0 or i==len(times):excluded['outside_gt_time_range']+=1;continue
        a,b=i-1,i;gap=times[b]-times[a];nearest=a if t-times[a]<=times[b]-t else b
        delta=abs(t-times[nearest])
        if gap>.10001 or delta>.05001:excluded['pose_gap_or_time_offset']+=1;continue
        alpha=(t-times[a])/gap;position=(1-alpha)*poses[a,1:4]+alpha*poses[b,1:4]
        rotation=slerp([t])[0];camera_position=position+rotation.apply(tbc);camera_rotation=rotation*rbc
        selected.append({'sequence':seq,'camera':camera,'timestamp_text':Path(member['path']).stem,'timestamp':t,
            'archive_path':member['path'],'zip_entry':member,'robot_position_xyz':position.tolist(),
            'robot_quaternion_xyzw':rotation.as_quat().tolist(),'camera_position_xyz':camera_position.tolist(),
            'camera_quaternion_xyzw':camera_rotation.as_quat().tolist(),'camera_forward_world':camera_rotation.apply([0,0,1]).tolist(),
            'nearest_gt_timestamp':float(times[nearest]),'gt_time_offset_seconds':float(delta),
            'interpolation_gt_timestamps':[float(times[a]),float(times[b])],'interpolation_alpha':float(alpha),
            'mapping_or_sequence_arc_m':float((1-alpha)*arc[a]+alpha*arc[b])})
    return selected,{'camera_transform':cal,'candidate_views':len(selected),'excluded':dict(excluded),'ground_truth_pose_count':len(poses)}


def angle(a,b):
    return math.degrees(2*math.acos(min(1,abs(float(np.dot(a,b))))))


def select():
    if (OUT/'pairs.json').exists():
        print('Pair manifest already frozen; no re-selection.',flush=True);return
    sources=json.loads((OUT/'sources.json').read_text());indices={s:archive_index(s) for s in ('mapping','lt_changes_0')}
    candidates={};audit={}
    for seq in indices:
        candidates[seq],audit[seq]=views(seq,indices[seq])
        write(OUT/'metadata'/seq/'archive-index-summary.json',{'entries':len(indices[seq]),
            'archive_bytes':sources['sequences'][seq]['archive_bytes'],'zip64_offsets_present':any(r['offset']>2**32 for r in indices[seq])})
    left=candidates['mapping'];right=candidates['lt_changes_0'];positions=np.array([v['camera_position_xyz'] for v in left]);pool=[]
    for b in right:
        distances=np.linalg.norm(positions-np.array(b['camera_position_xyz']),axis=1)
        for i in np.flatnonzero(distances<=.5):
            a=left[int(i)];degrees=angle(a['camera_quaternion_xyzw'],b['camera_quaternion_xyzw'])
            if degrees<=20:
                forward=math.degrees(math.acos(float(np.clip(np.dot(a['camera_forward_world'],b['camera_forward_world']),-1,1))))
                pool.append({'a':a,'b':b,'distance_m':float(distances[i]),'orientation_difference_degrees':degrees,
                    'forward_direction_difference_degrees':forward,'strict':bool(distances[i]<=.25 and degrees<=10)})
    selected=[];used_a=set();used_b=set();cells=set()
    def novelty(pair):
        if not selected:return 10.
        a=pair['a'];return min(math.sqrt(np.linalg.norm(np.array(a['camera_position_xyz'])-s['a']['camera_position_xyz'])**2+
            (angle(a['camera_quaternion_xyzw'],s['a']['camera_quaternion_xyzw'])/30)**2) for s in selected)
    for strict in (True,False):
        while len(selected)<50:
            available=[]
            for pair in pool:
                if pair['strict']!=strict or pair['a']['archive_path'] in used_a or pair['b']['archive_path'] in used_b:continue
                # Reject near-identical viewpoints on BOTH trajectories, independently of timing.
                near=False
                for old in selected:
                    for side in ('a','b'):
                        dist=np.linalg.norm(np.array(pair[side]['camera_position_xyz'])-old[side]['camera_position_xyz'])
                        deg=angle(pair[side]['camera_quaternion_xyzw'],old[side]['camera_quaternion_xyzw'])
                        if dist<.75 and deg<25:near=True;break
                    if near:break
                if near:continue
                xyz=pair['a']['camera_position_xyz'];cell=(math.floor(xyz[0]/3),math.floor(xyz[1]/3))
                score=(cell not in cells,novelty(pair),-(pair['distance_m']+pair['orientation_difference_degrees']/100),-pair['a']['timestamp'])
                available.append((score,pair,cell))
            if not available:break
            _,pair,cell=max(available,key=lambda x:x[0]);selected.append(pair);cells.add(cell)
            used_a.add(pair['a']['archive_path']);used_b.add(pair['b']['archive_path'])
    selected.sort(key=lambda p:p['a']['timestamp'])
    for n,pair in enumerate(selected,1):
        pair['pair_id']=f'pair-{n:03d}'
        for side,k in [('a',2*n-1),('b',2*n)]:pair[side]['frame_id']=f'view-{k:06d}'
        pair['spatial_cell_3m']=[math.floor(pair['a']['camera_position_xyz'][i]/3) for i in (0,1)]
    write(OUT/'pairs.json',{'protocol':'lilocbench-pairs-v1','pairs':selected,'selection':{
        'target_pairs':50,'camera':'camera_front','initial_thresholds':{'distance_m':.25,'orientation_degrees':10},
        'maximum_thresholds':{'distance_m':.5,'orientation_degrees':20},'pose_interpolation':'linear translation and quaternion SLERP in global GT frame, then calibrated RGB optical extrinsics',
        'max_nearest_gt_time_offset_seconds':.05,'max_bracketing_gt_gap_seconds':.1,'rgb_candidate_interval_seconds':.5,
        'viewpoint_exclusion':'Reject when either sequence has another selected center within 0.75m AND rotation within 25deg',
        'ranking':'Strict candidates first; new 3m spatial cells, then farthest calibrated camera pose; deterministic tie-breaking.',
        'strict_pairs':sum(p['strict'] for p in selected),'spatial_cells':len(cells),
        'shortfall_reason':None if len(selected)==50 else 'Insufficient distinct GT-consistent camera viewpoints under the fixed thresholds and novelty requirement.',
        'room_labels':'No room ground truth provided; spatial cells are coverage aids, not claimed room identities.'},'candidate_audit':audit})
    print('Frozen pairs:',len(selected),'strict:',sum(p['strict'] for p in selected),'spatial cells:',len(cells),flush=True)


def download():
    manifest=json.loads((OUT/'pairs.json').read_text());transfers=Transfers();sources=json.loads((OUT/'sources.json').read_text())
    streams={s:RemoteZip(sources['sequences'][s]['archive'],transfers) for s in sources['sequences']}
    images=OUT/'images';images.mkdir(exist_ok=True)
    seen=set();neutral=[]
    for pair in manifest['pairs']:
        for side in ('a','b'):
            view=pair[side];target=images/(view['frame_id']+'.png')
            if not target.exists():
                row=view['zip_entry'];info=zipfile.ZipInfo(row['path']);info.header_offset=row['offset'];info.compress_size=row['compressed_bytes'];info.file_size=row['bytes'];info.CRC=row['crc32'];info.compress_type=row['compression']
                target.write_bytes(member(streams[view['sequence']],info))
            digest=hashlib.sha256(target.read_bytes()).hexdigest();view['sha256']=digest
            if digest in seen:raise ValueError('Duplicate selected RGB bytes; review frozen selection, do not fabricate replacements')
            seen.add(digest)
            with Image.open(target) as image:size=list(image.size)
            view['local_path']=str(target.relative_to(ROOT));view['image_size']=size
            neutral.append({'dataset':'lilocbench','index':len(neutral),'frame_id':view['frame_id'],
                'image_path':str(target),'image_sha256':digest,'image_size':size,'subset20':False,'source_frame_id':view['frame_id']})
    write(OUT/'pairs.json',manifest);write(OUT/'native-selection.json',neutral)
    for start in range(0,len(manifest['pairs']),10):
        group=manifest['pairs'][start:start+10];sheet=Image.new('RGB',(650,len(group)*265),'white');draw=ImageDraw.Draw(sheet)
        for i,pair in enumerate(group):
            for col,side in enumerate(('a','b')):
                with Image.open(ROOT/pair[side]['local_path']) as image:
                    image.thumbnail((320,240));sheet.paste(image,(col*325,i*265))
            draw.text((4,i*265+242),f"{pair['pair_id']}  {pair['distance_m']:.3f}m  {pair['orientation_difference_degrees']:.1f}deg",fill='black')
        sheet.save(OUT/f'contact-{start//10+1}.jpg')
    write(OUT/'transfer-summary.json',{'payload_bytes':sum(r['payload_bytes'] for r in transfers.rows),
        'successful_requests':sum('error' not in r for r in transfers.rows),'failed_requests':sum('error' in r for r in transfers.rows),
        'archive_range_requests':sum(r['start'] is not None for r in transfers.rows),
        'whole_archive_downloaded':False,'downloaded_rgb_images':len(neutral),'depth_images_downloaded':0,
        'scope':'Actual HTTP response-body bytes for these dataset/metadata requests, including retries; excludes HTTP/TLS headers and documentation browsing.'})
    print('RGB saved:',len(neutral),'downloaded bytes:',sum(r['payload_bytes'] for r in transfers.rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('inspect','select','download'))
    args=parser.parse_args();globals()[args.phase]()
