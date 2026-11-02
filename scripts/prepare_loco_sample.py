"""Select 6 annotated photographs per LOCO environment and fetch only 30 JPEGs.

COCO annotations are pinned and verified. ZIP HTTP ranges avoid downloading the
whole image archive. Discovery inputs contain no object labels or boxes.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

REVISION = 'b460ab8c37f09162f0613c7546948c496e1b1628'
ANNOTATIONS = f'https://raw.githubusercontent.com/tum-fml/loco/{REVISION}/rgb/loco-all-v1.json'
ANNOTATION_SHA = '5a794d2e61f308150039e3e1953bc57e8fad208bb3e2b9117245230ec1c22054'
ARCHIVE = 'https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=73e976ba&path=LOCO%5Cv1%5Cdataset.zip'
ARCHIVE_SIZE = 769055500


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch_range(start, length):
    request = urllib.request.Request(ARCHIVE, headers={'Range': f'bytes={start}-{start+length-1}'})
    with urllib.request.urlopen(request, timeout=90) as response:
        # Refuse a server ignoring Range rather than silently fetching 770 MB.
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
            raise ValueError('Server did not honor the requested ZIP byte range.')
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError('Incomplete or oversized ZIP byte range.')
    return data


class RemoteZip(io.RawIOBase):
    """Only used to read the ZIP central directory; images use one request each."""
    def __init__(self):
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else ARCHIVE_SIZE + offset
        return self.position

    def read(self, length=-1):
        length = ARCHIVE_SIZE - self.position if length < 0 else min(length, ARCHIVE_SIZE - self.position)
        if length <= 0:
            return b''
        data = fetch_range(self.position, length)
        self.position += len(data)
        return data


def timestamp(image):
    number = float(Path(image['file_name']).stem.replace(',', '.'))
    return number / 1000 if number > 100_000_000_000 else number


def valid_boxes(image, annotations):
    if not annotations:
        return False
    for a in annotations:
        x, y, w, h = a['bbox']
        if not (w > 0 and h > 0 and x >= 0 and y >= 0 and x+w <= image['width'] and y+h <= image['height']):
            return False
    return True


def member_name(image, members):
    # Subset 1 is flattened in the ZIP; the other subsets retain capture paths.
    source = image['path'].lstrip('/')
    if source in members:
        return source
    subset = source.split('/')[1]
    flat = f"dataset/{subset}/{image['file_name']}"
    return flat if flat in members else None


def select(coco, members):
    """Six numeric-filename bins per subset; prefer class diversity and rare classes."""
    annotations = defaultdict(list)
    for a in coco['annotations']:
        annotations[a['image_id']].append(a)
    counts = Counter(a['category_id'] for a in coco['annotations'])
    chosen = []
    for subset in range(1, 6):
        candidates = [i for i in coco['images'] if f'/subset-{subset}/' in i['path'] and member_name(i, members) and valid_boxes(i, annotations[i['id']])]
        candidates.sort(key=lambda i: (timestamp(i), i['id']))
        if len(candidates) < 6:
            raise ValueError(f'Not enough valid images in subset {subset}.')
        for bin_id in range(6):
            group = candidates[len(candidates)*bin_id//6:len(candidates)*(bin_id+1)//6]
            def score(i):
                labels = {a['category_id'] for a in annotations[i['id']]}
                return len(labels), sum(1/counts[c] for c in labels), min(len(annotations[i['id']]), 20), -i['id']
            image = max(group, key=score)
            chosen.append((subset, bin_id+1, image, annotations[image['id']]))
    return chosen


def image_bytes(info):
    # Enough room for the maximum ZIP filename+extra header, plus image bytes.
    length = min(30 + 65535*2 + info.compress_size, ARCHIVE_SIZE-info.header_offset)
    block = fetch_range(info.header_offset, length)
    if block[:4] != b'PK\x03\x04':
        raise ValueError('Invalid local ZIP header.')
    fields = struct.unpack('<4s5H3I2H', block[:30])
    offset = 30 + fields[-2] + fields[-1]
    packed = block[offset:offset+info.compress_size]
    if len(packed) != info.compress_size:
        raise ValueError('Incomplete compressed image.')
    data = zlib.decompress(packed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else packed if info.compress_type == zipfile.ZIP_STORED else None
    if data is None or len(data) != info.file_size or zlib.crc32(data) & 0xffffffff != info.CRC:
        raise ValueError('ZIP image checksum failed.')
    return data


def extend100(output):
    """Keep the old 30; add maximin timestamp views, 20 per subset, resumably."""
    from PIL import Image
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True)
    previous=Path('data/loco_test30')
    selection=json.loads((previous/'selection.json').read_text())
    old={r['frame_id']:r for r in selection}
    records=[]
    for line in (previous/'discovery.jsonl').read_text().splitlines():
        r=json.loads(line);s=old[r['frame_id']]
        records.append({'frame_id':r['frame_id'],'dataset':'loco','environment':f"loco-subset-{s['subset']}",
            'image_path':str(previous/r['image_path']),'image_size':r['image_size'],'sha256':r['image_sha256'],
            'source_image_id':s['source_image_id'],'timestamp':timestamp({'file_name':s['source_file_name']})})
    target=output/'selection.json'
    if target.exists():records=json.loads(target.read_text())
    def save():
        temp=target.with_suffix('.tmp');temp.write_text(json.dumps(records,indent=2));temp.replace(target)
    for r in records:assert sha(Path(r['image_path']).read_bytes())==r['sha256']
    save()
    if len(records)==100:return
    cache=Path('results/experiment200/source-coco.json');cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(ANNOTATIONS,timeout=120) as response:data=response.read()
        assert sha(data)==ANNOTATION_SHA;cache.write_bytes(data)
    original=cache.read_bytes();assert sha(original)==ANNOTATION_SHA
    coco=json.loads(original)
    with zipfile.ZipFile(RemoteZip()) as remote:infos={i.filename:i for i in remote.infolist()}
    used={r['sha256'] for r in records}
    for dataset in ('thud_agtea_test50','thud_dense_test50'):
        for path in (Path('data')/dataset/'images').iterdir():
            if path.is_file():used.add(sha(path.read_bytes()))
    try:
        for subset in range(1,6):
            env=f'loco-subset-{subset}'
            candidates=[i for i in coco['images'] if f'/subset-{subset}/' in i['path'] and member_name(i,infos)]
            while sum(r['environment']==env for r in records)<20:
                ids={r['source_image_id'] for r in records};times=[r['timestamp'] for r in records if r['environment']==env]
                available=[i for i in candidates if i['id'] not in ids]
                if not available:raise ValueError('No unique candidates left in '+env)
                image=max(available,key=lambda i:(min(abs(timestamp(i)-t) for t in times),-i['id']))
                candidates.remove(image)
                data=image_bytes(infos[member_name(image,infos)]);digest=sha(data)
                if digest in used:continue
                with Image.open(io.BytesIO(data)) as decoded:
                    decoded.load();assert decoded.size==(image['width'],image['height'])
                fid=f"loco-s{subset}-{image['id']}";path=output/'images'/(fid+'.jpg');path.write_bytes(data)
                records.append({'frame_id':fid,'dataset':'loco','environment':env,'image_path':str(path),
                    'image_size':[image['width'],image['height']],'sha256':digest,'source_image_id':image['id'],
                    'timestamp':timestamp(image)})
                used.add(digest);save()
                print(f'LOCO selected {len(records)}/100',flush=True)
    except Exception as exc:
        (output/'download-error.json').write_text(json.dumps({'error':str(exc),'selected':len(records),'missing':100-len(records)}))
        raise
    (output/'source.json').write_text(json.dumps({'revision':REVISION,'annotations_sha256':ANNOTATION_SHA,
        'archive_url':ARCHIVE,'selection':'Keep 30; maximin timestamp distance within each subset, no class-based selection; SHA256 deduplication',
        'images':len(records),'per_environment':dict(Counter(r['environment'] for r in records))},indent=2))


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, help='Optional local copy of the pinned COCO file.')
    parser.add_argument('--output', type=Path, default=Path('data/loco_test30'))
    parser.add_argument('--extend100',action='store_true',help='Keep existing 30 and fetch only 70 more selected photographs')
    args = parser.parse_args()
    if args.extend100:
        extend100(args.output);return
    if args.output.exists():
        raise ValueError('Output exists. Keep it and use a new output path for a new selection.')
    if args.annotations:
        original = args.annotations.read_bytes()
    else:
        with urllib.request.urlopen(ANNOTATIONS, timeout=120) as response:
            original = response.read()
    if sha(original) != ANNOTATION_SHA:
        raise ValueError('Annotations differ from the pinned source.')
    coco = json.loads(original)
    categories = {c['id']: c['name'] for c in coco['categories']}
    with zipfile.ZipFile(RemoteZip()) as remote:
        infos = {i.filename: i for i in remote.infolist()}
    selected = select(coco, infos)
    jobs = []
    for subset, bin_id, image, anns in selected:
        member = member_name(image, infos)
        jobs.append((subset, bin_id, image, anns, infos[member]))
    # These are independent reads; writes happen below in deterministic order.
    with ThreadPoolExecutor(max_workers=4) as executor:
        payloads = list(executor.map(lambda j: image_bytes(j[4]), jobs))
    hashes = [sha(data) for data in payloads]
    if len(set(hashes)) != 30:
        raise ValueError('Selected photographs contain byte-identical duplicates.')
    for job, data in zip(jobs, payloads):
        image = job[2]
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            if decoded.size != (image['width'], image['height']):
                raise ValueError('COCO dimensions do not match the photograph.')
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir()
    records, discovery, sources, selected_anns, selected_images = [], [], [], [], []
    for (subset, bin_id, image, anns, info), data in zip(jobs, payloads):
        fid = f"loco-s{subset}-{image['id']}"
        relative = f'images/{fid}.jpg'
        (args.output/relative).write_bytes(data)
        base = {'frame_id': fid, 'image_path': relative, 'image_size': [image['width'], image['height']], 'image_sha256': sha(data)}
        discovery.append(base)
        objects = []
        for a in sorted(anns, key=lambda a: a['id']):
            x, y, w, h = a['bbox']
            objects.append({'id': a['id'], 'label': categories[a['category_id']], 'bbox_xyxy': [x,y,x+w,y+h], 'region_kind': 'object_instance', 'source_category_id': a['category_id']})
        records.append({**base, 'scene': f'loco-subset-{subset}', 'objects': objects})
        sources.append({'frame_id': fid, 'subset': subset, 'filename_bin': bin_id, 'source_image_id': image['id'], 'source_file_name': image['file_name'], 'source_path': image['path'], 'archive_member': info.filename, 'zip_crc32': f'{info.CRC:08x}', 'sha256': sha(data), 'annotations': len(anns)})
        selected_images.append({**image, 'file_name': relative})
        selected_anns.extend(anns)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    for name, rows in [('objects.jsonl', records), ('discovery.jsonl', discovery)]:
        (args.output/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    save('annotations.coco.json', {'images': selected_images, 'annotations': selected_anns, 'categories': coco['categories']})
    save('selection.json', sources)
    save('source.json', {'dataset': 'LOCO: Logistics Objects in Context', 'repository': 'https://github.com/tum-fml/loco', 'revision': REVISION, 'annotations_url': ANNOTATIONS, 'annotations_sha256': ANNOTATION_SHA, 'archive_url': ARCHIVE, 'archive_size_bytes': ARCHIVE_SIZE, 'license': 'CC0-1.0', 'source_file_images': len(coco['images']), 'source_file_annotations': len(coco['annotations']), 'selection': '6 numeric-filename bins in each of 5 subsets; maximize category diversity, then rare-category coverage, then capped region count; no model outputs used', 'selected_images': 30, 'selected_annotations': len(selected_anns), 'category_counts': dict(Counter(categories[a['category_id']] for a in selected_anns)), 'stability_ground_truth': False})
    (args.output/'SHA256SUMS.txt').write_text(''.join(f"{r['image_sha256']}  {r['image_path']}\n" for r in records))
    print(f'Ready: {args.output}, 30 photographs, {len(selected_anns)} annotations.')


if __name__ == '__main__':
    main()
