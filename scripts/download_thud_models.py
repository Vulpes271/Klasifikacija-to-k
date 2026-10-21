#!/usr/bin/env python3
"""Download pinned public model files locally; curl resumes interrupted files.

No credentials are needed. Large weights stay below the ignored results folder.
The saved Hub metadata fixes the exact revision used by both GPU runners.
"""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/thud-extended'
MODELS = {'nvila': ('Efficient-Large-Model/NVILA-8B', '049743cc51bdd8872cdf696a1a58b03fef2c367e'),
          'florence': ('microsoft/Florence-2-large', '21a599d414c4d928c9032694c424fb94458e3594')}


def download(item):
    model, revision, name, target = item
    if target.exists():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    url = f'https://huggingface.co/{model}/resolve/{revision}/{name}?download=true'
    subprocess.run(['curl', '--fail', '--location', '--retry', '5',
                    '--connect-timeout', '30', '--continue-at', '-',
                    '--output', str(partial), url], check=True)
    partial.rename(target)
    print('Downloaded', target.relative_to(ROOT), flush=True)
    return str(target)


def main():
    files = []
    for key, (model, revision) in MODELS.items():
        meta = OUT / 'setup' / f'{key}-api.txt'
        if not meta.exists():
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_bytes(urllib.request.urlopen('https://huggingface.co/api/models/' + model + '/revision/' + revision).read())
        info = json.loads(meta.read_text())
        if info['sha'] != revision:
            raise ValueError('Model revision changed: ' + model)
        for record in info['siblings']:
            name = record['rfilename']
            if name.endswith(('.bin', '.ipynb')) or name == 'trainer_state.json':
                continue
            files.append((model, info['sha'], name, OUT / 'models' / key / name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))


if __name__ == '__main__':
    main()
