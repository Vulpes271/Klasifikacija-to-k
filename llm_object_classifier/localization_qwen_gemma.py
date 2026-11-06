"""One local batch: shared Florence/RGB/SAM, Qwen versus Gemma classification only."""
import argparse
import copy
import csv
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np
from PIL import Image
import yaml

from .box_segmentation import digest,save_preview
from .experiment import options
from .llm_checks import validate_response,make_signature,check_signature,installed_digest,typed_id,fingerprint
from .localization_pilot import ROOT,write,docker_command,pose_table,project_scan,transform_tree,mask_layers,evaluate
from .localization_comparison import load_estimates
from .localization_full import plot_full
from .pipeline import load_settings,run_text
from .vision_backends import Backend


def read(path):return json.loads(Path(path).read_text())


def source_for(item,pilot,cfg):
    old=item['origin']=='stored-lilocbench'
    folder=ROOT/pilot['stored_results'] if old else ROOT/cfg['full_cache']/'inference'
    model=pilot['classification_model'] if old else 'qwen'
    return folder,model


def reclassify(frame,decisions,model):
    """Copy class metadata only; preserve shared region grouping, masks and IDs."""
    result=copy.deepcopy(frame)
    for region in result['regions']:
        for source in region['sources']:
            key=typed_id(source['object']['id'])
            source['llm']['models'][model]=decisions.get(key,{'classification':None})
    return result


def run(cfg_path):
    cfg=read(cfg_path);pilot=read(ROOT/cfg['pilot_config']);cache=ROOT/cfg['cache'];output=ROOT/cfg['output']
    cache.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    if cfg['seed']!=42 or pilot['seed']!=42:raise ValueError('Pilot seed must stay 42')
    source=ROOT/cfg['full_results'];all_arrays=np.load(source/'changed-replay/scans.npz');all_eval=np.load(source/'changed-replay/evaluation.npz')
    selected=(all_arrays['times']>=cfg['start'])&(all_arrays['times']<=cfg['end'])
    arrays={key:value[selected] if key in ['times','ranges','odom','odom_end','removed'] else value for key,value in all_arrays.items()}
    times=arrays['times'];gt=all_eval['gt'][selected];has_gt=all_eval['has_gt'][selected]
    if not len(times) or not has_gt[0]:raise ValueError('Segment lacks scans/initial GT')
    x0,y0,x1,y1=cfg['plot_bounds_xyxy']
    if np.any((gt[has_gt,0]<x0)|(gt[has_gt,0]>x1)|(gt[has_gt,1]<y0)|(gt[has_gt,1]>y1)):
        raise ValueError('Chosen crop omits selected GT route')
    items=[item for item in read(ROOT/cfg['full_cache']/'mask-status.json') if cfg['start']-.1<=item['timestamp']<=cfg['end']+.1]
    protocol={'configuration':cfg,'scans':len(times),'rgb_frames':len(items),'amcl':pilot['amcl'],
              'selected_before_segment_runs':True,'only_change':'classification model; exact same frozen inputs, system/user messages, output schema, SAM masks, filtered-region groups, RGB timestamps, projection, AMCL and seed',
              'initial_pose':gt[0].tolist(),'rgb_ids':[i['frame_id'] for i in items]}
    path=output/'protocol.json'
    if path.exists() and read(path)!=protocol:raise ValueError('Changed protocol; choose a new output')
    if not path.exists():write(path,protocol)
    cfg_llm=load_settings(ROOT/cfg['pipeline_config']);cfg_llm['llm']['selected']='gemma'
    if cfg_llm['llm']['alternatives']['gemma']!=cfg['gemma']:raise ValueError('Unexpected Gemma alternative')
    frozen=cache/'gemma';(frozen/'frozen').mkdir(parents=True,exist_ok=True);(frozen/'prompts/users').mkdir(parents=True,exist_ok=True)
    ip=subprocess.check_output(['docker','inspect',pilot['llm_container'] if 'llm_container' in pilot else cfg_llm['llm']['container'],
        '--format','{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'],text=True).strip()
    backend=Backend({'backend':'ollama','name':'gemma','model':cfg['gemma'],'base_url':f'http://{ip}:11434','precision':'Q4_K_M','require_gpu':True},cfg_llm['llm']['timeout'])
    metadata=backend.metadata();gemma_digest=installed_digest(backend)
    opts=options(cfg_llm['llm'],{'temperature':cfg_llm['llm']['temperature'],'seed':cfg_llm['llm']['seed']})
    manifest={'config':{'models':[{'name':'gemma','model':cfg['gemma'],'backend':'ollama','precision':'Q4_K_M','require_gpu':True}]},'inputs':[]}
    local_docs={};prompts={};raw_qwen={};cached_gemma=0;report_index={};shared_frames={};input_hashes={}
    for item in items:
        fid=item['frame_id'];folder,qname=source_for(item,pilot,cfg)
        fp=folder/'frozen'/(fid+'.json');up=folder/'prompts/users'/(fid+'.txt');qp=folder/'responses'/qname/(fid+'.json')
        for p in [fp,up,qp,Path(item['path'])]:
            if not p.exists():raise FileNotFoundError(f'Missing shared section input: {p}')
            input_hashes[str(p.relative_to(ROOT))]=digest(p)
        if digest(Path(item['path']))!=item['image_sha256']:raise ValueError('RGB identity changed')
        doc=read(fp);user=up.read_text();qraw=read(qp)
        qmessages=(qraw.get('signature') or {}).get('payload',{}).get('messages') or (qraw.get('request') or {}).get('messages')
        expected=[{'role':'system','content':cfg_llm['llm']['system_file_text']},{'role':'user','content':user}]
        if qmessages!=expected:raise ValueError(f'Qwen input differs from exact shared messages: {fid}')
        qpayload=(qraw.get('signature') or {}).get('payload',{});qrequest=qraw.get('request') or {}
        if fingerprint(qpayload.get('options',qrequest.get('options')))!=fingerprint(opts):raise ValueError('Classification sampling options differ')
        if qpayload.get('format',qrequest.get('format'))!='json' or qpayload.get('think',qrequest.get('think')) is not False:
            raise ValueError('Classification JSON/think settings differ')
        local_docs[fid]=doc;prompts[fid]=user;raw_qwen[fid]=qraw
        target=frozen/'frozen'/(fid+'.json');shutil.copyfile(fp,target)
        (frozen/'prompts/users'/(fid+'.txt')).write_text(user)
        manifest['inputs'].append({'frame_id':fid,'frozen':f'frozen/{fid}.json','input_sha256':digest(fp),'user_prompt':f'prompts/users/{fid}.txt'})
        # Reuse legacy Gemma only when its signed inputs/options/digest exactly match.
        legacy=ROOT/pilot['stored_results']/'responses/gemma3-27b-q4'/(fid+'.json')
        destination=frozen/'responses/gemma'/(fid+'.json')
        if not destination.exists() and legacy.exists():
            old=read(legacy);signature=make_signature(doc,digest(fp),cfg['gemma'],gemma_digest,cfg_llm['llm']['system_file_text'],user,opts)
            if check_signature(old,signature,doc,digest(fp))['verified']:
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(legacy,destination)
        if legacy.exists() and destination.exists() and digest(legacy)==digest(destination):cached_gemma+=1
        if doc['status']=='ok':
            if not item.get('mask_report'):raise FileNotFoundError(f'Missing shared SAM masks for {fid}; prepare with localization_full --stage filter')
            report=Path(item['mask_report'])
            if report not in report_index:report_index[report]={f['frame_id']:f for f in read(report)['frames']}
            frame=report_index[report][fid];shared_frames[fid]=(report.parent,frame)
            for region in frame['regions']:
                mp=report.parent/region['mask_path']
                if digest(mp)!=region['mask_sha256']:raise ValueError('Shared SAM mask changed')
                input_hashes[str(mp.relative_to(ROOT))]=digest(mp)
    write(frozen/'manifest.json',manifest)
    write(frozen/'run.json',{'config':cfg_llm,'provenance':'Exact frozen Florence documents and user prompts copied byte-for-byte from Qwen sources; model selection alone differs'})
    # The existing runner handles GPU checks, token budget, schema and terminal skips.
    if not all((frozen/'responses/gemma'/(f['frame_id']+'.json')).exists() for f in manifest['inputs']):
        run_text(frozen,{'config':cfg_llm},manifest)
    class_data={};layers={'qwen':{},'gemma':{}};class_counts={};previews=output/'previews'
    for model in ['qwen','gemma']:
        records={};valid_count=0;counts={'fixed':0,'movable':0,'uncertain':0};errors=[]
        for item in items:
            fid=item['frame_id'];doc=local_docs[fid]
            raw=raw_qwen[fid] if model=='qwen' else read(frozen/'responses/gemma'/(fid+'.json'))
            check=validate_response(raw,doc['parsed']['objects']);valid=doc['status']=='ok' and check['schema_valid_response']
            decisions={typed_id(i['id']):i for i in check['usable_items']};records[fid]={'valid':bool(valid),'decisions':decisions}
            if valid:
                valid_count+=1
                for value in decisions.values():counts[value['classification']]+=1
            else:errors.append({'frame_id':fid,'status':raw.get('status'),'errors':check['errors']})
            if fid not in shared_frames:continue
            folder,original=shared_frames[fid];frame=reclassify(original,decisions,model)
            if valid:layers[model][fid]=mask_layers(folder,frame,model)
            image=Image.open(item['path']).convert('RGB');dest=previews/model/fid;dest.mkdir(parents=True,exist_ok=True)
            for region in frame['regions']:
                binary=np.asarray(Image.open(folder/region['mask_path']))>0
                save_preview(image,binary,region,dest/f'region-{region["region_id"]:04d}.jpg',model)
        class_data[model]=records;class_counts[model]={'valid_rgb':valid_count,'classes_on_valid_rgb':counts,'invalid_or_skipped':errors}
    data=ROOT/cfg['data'];seq=data/cfg['sequence'];ot,op,_=pose_table(seq/'dingo_velocity_controller/odom.txt')
    transform=transform_tree(yaml.safe_load((seq/'transformations.yaml').read_text()))
    intrinsic=yaml.safe_load((seq/'camera_front/color/intrinsics.yaml').read_text());base_scan=transform('base_link','scan_front_link');base_camera=transform('base_link','camera_front_color_optical_frame')
    baseline=read(source/'changed-replay/input.json');baseline['initial_pose']=gt[0].tolist()
    removals={};coverages={}
    for model in ['qwen','gemma']:
        candidates=[item for item in items if item['frame_id'] in layers[model]]
        stamps=np.array([i['timestamp'] for i in candidates]);removed=np.zeros(arrays['ranges'].shape,bool);valid=np.zeros(len(times),bool)
        for i,t in enumerate(times):
            if not len(stamps):continue
            nearest=int(np.argmin(abs(stamps-(t+baseline['laser']['scan_time']/2))))
            if abs(stamps[nearest]-(t+baseline['laser']['scan_time']/2))>pilot['mask_max_dt']:continue
            item=candidates[nearest];uv,visible=project_scan(arrays['ranges'][i],t,item['timestamp'],baseline['laser'],base_scan,base_camera,intrinsic,ot,op,pilot['pose_max_gap'])
            pixels=np.floor(uv[visible]).astype(int);indices=np.flatnonzero(visible)
            removed[i,indices]=layers[model][item['frame_id']][pixels[:,1],pixels[:,0]];valid[i]=True
        removals[model]=removed;coverages[model]=valid
    if not np.array_equal(removals['qwen'],arrays['removed']):raise ValueError('Reused Qwen section filter differs from the original full-run filter')
    metrics={};branches={};success={};titles={'A':'AMCL brez filtra','B':'Filter Qwen','C':'Filter Gemma'}
    for branch,model in [('A',None),('B','qwen'),('C','gemma')]:
        replay=output/f'replay-{branch}';replay.mkdir(exist_ok=True)
        selected_arrays={**arrays,'removed':np.zeros(arrays['ranges'].shape,bool) if model is None else removals[model]}
        np.savez_compressed(replay/'scans.npz',**selected_arrays);write(replay/'input.json',baseline)
        (replay/'amcl.yaml').write_text(yaml.safe_dump({'amcl':{'ros__parameters':pilot['amcl']}}))
        if not (output/branch/'poses.json').exists():
            command=docker_command(pilot['ros_image']);command[3:3]=['--network','none','-e','ROS_DOMAIN_ID=91','-e','ROS_LOCALHOST_ONLY=1']
            command+=['bash','-c','source /opt/ros/jazzy/setup.bash && exec python3 -m llm_object_classifier.amcl_replay "$@"','comparison',
                      '--input',str(replay/'input.json'),'--branch','A' if model is None else 'B','--output',str(output/branch)]
            subprocess.run(command,check=True)
        estimates=load_estimates(output/branch/'poses.json',times)
        summary,pos,yaw,success[branch]=evaluate(gt,estimates,has_gt,data/'reference-code/evaluate.py')
        summary.update(mask_valid_scans=int(coverages[model].sum()) if model else 0,
                       mask_coverage_percent=float(coverages[model].mean()*100) if model else 0,
                       removed_hits=int(removals[model].sum()) if model else 0)
        metrics['unfiltered' if model is None else model]=summary
        branches[branch]={'times':times,'gt':gt,'has_gt':has_gt,'est':estimates}
    plot_full(output,data,branches,success,bounds=cfg['plot_bounds_xyxy'],titles=titles,
              filename='trajectories_qwen_gemma.png',heading='Same RGB / Florence / SAM; classifier only differs — seed 42, no voting')
    comparable=disagree=0
    for item in items:
        fid=item['frame_id']
        if all(class_data[m][fid]['valid'] for m in ['qwen','gemma']):
            for key,value in class_data['qwen'][fid]['decisions'].items():
                comparable+=1;disagree+=value['classification']!=class_data['gemma'][fid]['decisions'][key]['classification']
    result={'configuration':cfg,'models':{'qwen':cfg['qwen'],'gemma':cfg['gemma'],'gemma_digest':gemma_digest},
            'scans':len(times),'rgb':len(items),'legacy_gemma_reused':cached_gemma,'metrics':metrics,
            'classification':class_counts,'comparable_region_labels':comparable,'disagreeing_region_labels':disagree,
            'shared_inputs_verified':True,'same_qwen_filter_verified':True,'new_florence_calls':0,'new_sam_calls':0,
            'previews':str(previews.relative_to(ROOT))}
    write(output/'metrics.json',result)
    for path in (frozen/'responses/gemma').glob('*.json'):input_hashes[str(path.relative_to(ROOT))]=digest(path)
    write(output/'provenance.json',{'sha256':input_hashes,'system_sha256':digest(ROOT/cfg_llm['llm']['system_file']),
                                   'gemma_digest':gemma_digest,'same_inputs_and_masks':True})
    public=ROOT/'docs/lilocbench-qwen-gemma';public.mkdir(exist_ok=True)
    shutil.copyfile(output/'trajectories_qwen_gemma.png',public/'trajectories_qwen_gemma.png');shutil.copyfile(output/'metrics.json',public/'metrics.json')
    lines=['# Qwen / Gemma: spodnja desna soba in prihod po hodniku','',
           f'Odsek `{cfg["sequence"]}`: **{cfg["start"]:.0f}–{cfg["end"]:.0f}**, {len(times)} skenov in {len(items)} istih RGB slik. Izbran na podlagi GT poti pred klasifikacijo Gemma in novimi AMCL zagoni.','',
           f'Gemma **`{cfg["gemma"]}`**, digest `{gemma_digest}`; Qwen `{cfg["qwen"]}`. Gemma nadomesti **samo klasifikacijo** `fixed/movable/uncertain`. Isti zamrznjeni Florence JSON in bajtno enaki sistemski/uporabniški prompti; ista JSON shema, RGB, skupine regij, izvorni ID-ji in originalne SAM maske. Brez glasovanja.','',
           '![AMCL brez filtra / Qwen / Gemma](trajectories_qwen_gemma.png)','',
           '| Različica | Položaj <10 cm [%] | Yaw <4° [%] | Oba praga [%] | Uspešne / ocenljive | RMSE [cm / °] | Pokritost mask [%] | Odstranjeni zadetki |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in [('unfiltered','AMCL brez filtra'),('qwen','Qwen'),('gemma','Gemma')]:
        s=metrics[name];lines.append(f'| {label} | {s["position_below_10cm_percent"]:.2f} | {s["yaw_below_4deg_percent"]:.2f} | {s["success_percent"]:.2f} | {s["successful_poses"]}/{s["evaluable_poses"]} | {s["position_rmse_cm"]:.3f} / {s["yaw_rmse_deg"]:.3f} | {s["mask_coverage_percent"]:.2f} | {s["removed_hits"]} |')
    lines+=['',f'Veljavne klasifikacije slik: Qwen {class_counts["qwen"]["valid_rgb"]}/{len(items)}, Gemma {class_counts["gemma"]["valid_rgb"]}/{len(items)}. Različnih razredov je {disagree}/{comparable} primerljivih izvornih regij. Neveljaven/preskočen odgovor ohrani sken; ničesar ne popravljamo ali krajšamo. Razlika v pokritosti je posledica veljavnosti klasifikacij, ne različnih mask.',
            '', 'Vsi trije AMCL zagoni so sveži na istem odseku: ista prva GT inicializacija in kovarianca, odometrija, časovni koraki, nastavitve pilota in seme 42. Prejšnje celozaporedne poze niso izrezane kot nadomestilo za nov zagon. Poznejši GT samo ocenjuje in riše, brez poravnave trajektorij.',
            'Ocena AMCL je modra; GT je zelen samo pri hkratnih napakah <10 cm in <4°, drugod rdeč. Isti izrez in merilo v vseh panelih. Filter obdrži vse skene in uporablja samo posamezno RGB masko v oknu ±50 ms; odstranjeni zadetki so NaN.','',
            f'Ponovno uporabljene so vse Florence/RGB/SAM datoteke; novih Florence/SAM klicev ni. Veljavnih starih podpisanih Gemma odgovorov je ponovno uporabljenih {cached_gemma}. Surovi izpisi in predogledi ostanejo lokalno: `{previews.relative_to(ROOT)}`.','',
            '```bash','python3 -m llm_object_classifier.localization_qwen_gemma --config configs/lilocbench-qwen-gemma.json','```','',
            '[Konfiguracija](../../configs/lilocbench-qwen-gemma.json), [metrike in neveljavni odgovori](metrics.json). Zemljevid: [LILocBench, Trekel et al.](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); sestavljena slika pod isto licenco.']
    (public/'README.md').write_text('\n'.join(lines)+'\n')
    compact={k:result[k] for k in ['models','scans','rgb','legacy_gemma_reused','metrics','comparable_region_labels','disagreeing_region_labels','shared_inputs_verified','same_qwen_filter_verified','new_florence_calls','new_sam_calls','previews']}
    write(output/'summary.json',compact);print(json.dumps(compact,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/lilocbench-qwen-gemma.json')
    args=p.parse_args();output=ROOT/read(args.config)['output']
    try:run(args.config.resolve())
    except Exception as exc:
        write(output/'batch-error.json',{'error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
