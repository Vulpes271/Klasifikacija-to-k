"""Dva preverjena native klica Florence; ta modul teče v obstoječem CUDA okolju."""
import argparse
from pathlib import Path
import time
import traceback
import uuid

from .experiment import read_json,write_json,sha256
from .grounding_common import CAPTION,GROUND,freeze,weights_identity
from .gpu_runtime import environment,begin_measure,end_measure,validate_boxes,validate_eos
from .llm_checks import fingerprint


def task_signature(run, frame, task, caption=None):
    """Podpis poveže sliko, uteži, nastavitve in pri groundingu celoten izvorni opis."""
    payload={'image_sha256':frame['image_sha256'],'settings':run['config']['florence'],
             'weights':run['florence_weights']['sha256'],'task':task,
             'seed':run['config']['llm']['seed'],'implementation':run['source_sha256']['llm_object_classifier/florence_runtime.py']}
    if caption is not None:payload['caption_sha256']=sha256(caption.read_bytes())
    return fingerprint(payload)


def check_native(path, signature):
    """Napake in preskoki so tudi zaključeni zapisi; spremenjenega rezultata ne prepiše."""
    if not path.exists():return False
    if read_json(path).get('signature')!=signature:raise ValueError(f'Neujemanje Florence podpisa: {path}')
    return True


def run_florence(output):
    """Samostojno obdela RGB slike; nobenih referenc ali privzetega eksperimentalnega nabora."""
    run=read_json(output/'run.json');cfg=run['config']['florence'];frames=run['frames']
    if sha256(Path(__file__).read_bytes())!=run['source_sha256']['llm_object_classifier/florence_runtime.py']:
        raise ValueError('Spremenjena koda Florence izvajalnika')
    # Pred nalaganjem GPU preverimo tudi podpise že zaključenih klicev.
    pending=[]
    for frame in frames:
        folder=output/'native'/frame['frame_id'];cp=folder/'caption.json';gp=folder/'grounding.json'
        caption_ok=check_native(cp,task_signature(run,frame,CAPTION))
        grounding_ok=gp.exists() and caption_ok and check_native(gp,task_signature(run,frame,GROUND,cp))
        if not caption_ok or not grounding_ok:pending.append(frame)
    if weights_identity(cfg['model_dir'])!=run['florence_weights']:raise ValueError('Spremenjene lokalne uteži/procesor Florence')
    if not pending:
        print('Florence: vsi podpisi ustrezajo; 0 novih inferenc.',flush=True);return
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM,AutoProcessor
    if not torch.cuda.is_available():raise RuntimeError('Florence potrebuje CUDA; CPU ni dovoljen')
    model=AutoModelForCausalLM.from_pretrained(cfg['model_dir'],trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.float16,attn_implementation=cfg['attention']).cuda().eval()
    processor=AutoProcessor.from_pretrained(cfg['model_dir'],trust_remote_code=True,local_files_only=True)
    freeze(output/'florence-environment.json',{**environment(torch,model),'weights':run['florence_weights']['sha256'],
        'image_processor':processor.image_processor.to_dict(),'generation_config':model.generation_config.to_dict()})

    def infer(frame,task,text='',caption_path=None):
        """Procesor prejme originalno RGB sliko; postprocessor vrne originalne slikovne pike."""
        torch.manual_seed(run['config']['llm']['seed']);prompt=task+text
        row={'task':task,'user_prompt':prompt,'frame_id':frame['frame_id'],'image_sha256':frame['image_sha256'],
             'model':cfg['model'],'model_revision':cfg['revision'],'signature':task_signature(run,frame,task,caption_path),
             'generation':{k:cfg[k] for k in ('max_new_tokens','num_beams','do_sample')},
             'seed':run['config']['llm']['seed'],'call_id':uuid.uuid4().hex,'generation_attempted':True}
        if caption_path:row['source_caption']={'sha256':sha256(caption_path.read_bytes()),'text':text}
        tick=begin_measure(torch)
        try:
            with Image.open(output/frame['image_path']) as original:image=original.convert('RGB')
            inputs=processor(text=prompt,images=image,return_tensors='pt').to('cuda',torch.float16)
            row['input_token_ids']=inputs['input_ids'][0].tolist()
            row['effective_text_prompt']=processor._construct_prompts([prompt])[0]
            with torch.inference_mode():
                ids=model.generate(input_ids=inputs['input_ids'],pixel_values=inputs['pixel_values'],
                    max_new_tokens=cfg['max_new_tokens'],num_beams=cfg['num_beams'],do_sample=False)
            text=processor.batch_decode(ids,skip_special_tokens=False)[0]
            row.update(content=text,token_ids=ids[0].tolist(),output_tokens=int(ids.shape[-1]-1),
                       input_tensor_shapes={k:list(v.shape) for k,v in inputs.items()})
            row['parsed']=processor.post_process_generation(text,task=task,image_size=tuple(frame['image_size']))
            validate_boxes(row['parsed'],task,frame['image_size'])
            if ids.shape[-1]-1>=cfg['max_new_tokens']:raise ValueError('Dosežena meja izhoda; možen odrezan odgovor')
            row['last_token_id']=int(ids[0,-1])
            row['eos_token_ids']=validate_eos(row['last_token_id'],model.generation_config.eos_token_id,processor.tokenizer.eos_token_id)
            row['status']='ok'
        except Exception as exc:row.update(status='error',error=str(exc),traceback=traceback.format_exc())
        row.update(end_measure(torch,tick),finished_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        return row

    for frame in pending:
        if sha256((output/frame['image_path']).read_bytes())!=frame['image_sha256']:raise ValueError('Spremenjena shranjena RGB slika')
        folder=output/'native'/frame['frame_id'];cp=folder/'caption.json';gp=folder/'grounding.json'
        if not check_native(cp,task_signature(run,frame,CAPTION)):write_json(cp,infer(frame,CAPTION))
        if not check_native(gp,task_signature(run,frame,GROUND,cp)):
            caption=read_json(cp)
            if caption['status']=='ok':
                # Ni izbiranja fraz ali krajšanja: grounding dobi natanko CELOTEN opis.
                write_json(gp,infer(frame,GROUND,caption['parsed'][CAPTION],cp))
            else:write_json(gp,{'status':'skipped','error':'Ni veljavnega celotnega opisa','generation_attempted':False,
                               'signature':task_signature(run,frame,GROUND,cp)})
        print(f"Florence: {frame['frame_id']} shranjeno.",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run_florence(parser.parse_args().output.resolve())
