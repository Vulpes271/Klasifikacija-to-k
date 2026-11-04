"""Nalaganje celotnega LLM na GPU in dostop do njegovega obstoječega tokenizerja."""
import http.client
import io
import json
import re
import subprocess
from .vision_backends import request

def runner_http(port,endpoint,payload,container='klasifikacija-ollama-1'):
    """Uporabi tokenizer naloženega modela, brez nove generacije ali namestitve.

    Besedilo potuje po stdin, ne kot lupinski ukaz. HTTPResponse obravnava tudi
    deljen HTTP odgovor; dolžina besedila ni približek števila tokenov.
    """
    if endpoint!='/tokenize' or not 1<=int(port)<=65535:raise ValueError('Unexpected runner endpoint')
    command='''export LC_ALL=C
body=$(cat)
exec 3<>/dev/tcp/127.0.0.1/$1
printf 'POST %s HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: %s\r\n\r\n' "$2" "${#body}" >&3
printf '%s' "$body" >&3
cat <&3'''
    proc=subprocess.run(['docker','exec','-i',container,'/bin/bash','-c',command,'--',str(port),endpoint],
        input=json.dumps(payload,ensure_ascii=False).encode(),capture_output=True,timeout=120)
    if proc.returncode:raise RuntimeError('Runner tokenizer transport failed: '+proc.stderr.decode()[:300])
    class Socket:
        def makefile(self,*args,**kwargs):return io.BytesIO(proc.stdout)
    response=http.client.HTTPResponse(Socket());response.begin();body=response.read()
    if response.status!=200:raise RuntimeError('Runner tokenizer HTTP '+str(response.status)+': '+body[:300].decode())
    return json.loads(body)

def load_model(backend,opts,digest,container='klasifikacija-ollama-1'):
    """Naloži model in potrdi uteži, celoten GPU ter zahtevano velikost konteksta."""
    # Zahteva brez prompta samo naloži uteži; ne ustvarja dodatnega testnega odgovora.
    loaded=request(backend.url+'/api/generate',{'model':backend.spec['model'],'keep_alive':'30m','options':opts},backend.timeout)
    if loaded.get('eval_count',0):raise ValueError('Unexpected generation during load-only request')
    state=request(backend.url+'/api/ps');gpu=next(r for r in state['models'] if r['name']==backend.spec['model'])
    if gpu['digest']!=digest or gpu['size_vram']<gpu['size'] or gpu.get('context_length')!=opts['num_ctx']:
        raise ValueError('Full GPU/digest/context verification failed; CPU fallback forbidden')
    ps=subprocess.check_output(['docker','exec',container,'ps','-eo','args='],text=True)
    ports=[]
    for line in ps.splitlines():
        if ('runner' in line or 'llama-server' in line) and '--port' in line:
            match=re.search(r'--port[ =]+(\d+)',line)
            if match:ports.append(int(match.group(1)))
    if len(ports)!=1:raise ValueError('Cannot identify exactly one sequential runner tokenizer')
    # Preverjanje transporta tokenizerja ne izvaja inferenc.
    runner_http(ports[0],'/tokenize',{'content':'','add_special':False},container)
    return {'load_response':loaded,'gpu':gpu,'tokenizer_port':ports[0],'generation_performed':False}
