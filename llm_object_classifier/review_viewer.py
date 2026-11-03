"""Offline human review of saved results. Standard library only; no model calls."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime,timezone
import io
import json
import mimetypes
from pathlib import Path
import secrets
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import uuid
import webbrowser

from .experiment import read_json,strict_json,sha256

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RESULTS=ROOT/'results models/prompt_comparison40'
DEFAULT_FLORENCE=ROOT/'results models/florence_caption_grounding_test10'
DEFAULT_DB=ROOT/'reviews/manual_review.sqlite3'
RATINGS={None,'correct','incorrect','unassessable'}
ISSUES={'wrong_label','wrong_box','duplicate','group'}


def now():return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def id_key(value):
    if type(value) not in (str,int):raise ValueError('Invalid region ID')
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


class Results:
    """Adapters for the 40-image comparison and existing 200-image baseline."""
    def __init__(self,path,florence=DEFAULT_FLORENCE):
        self.path=Path(path).resolve();self.florence=Path(florence).resolve()
        if (self.path/'experiment200-manifest.json').is_file():
            self.kind='baseline200';self.manifest_path=self.path/'experiment200-manifest.json'
        else:
            self.kind='comparison40';self.manifest_path=self.path/'manifest.json'
        manifest=read_json(self.manifest_path)
        if self.kind=='comparison40' and manifest.get('protocol')!='prompt40-v1':
            raise ValueError('Expected prompt40-v1 or experiment200-manifest.json')
        self.manifest_sha=sha256(self.manifest_path.read_bytes())
        self.set_id=sha256((self.kind+'\0'+self.manifest_sha).encode())
        self.models=manifest['config']['models'];self.frames=[];self.flat=[]
        self.identities={};self.raw_cache={};self.cache_frame=None
        for i,entry in enumerate(manifest['inputs']):
            frozen=(self.path/entry['frozen']).resolve()
            if not frozen.is_relative_to(self.path):raise ValueError('Frozen path outside results')
            digest=entry.get('input_sha256',entry.get('sha256'))
            if sha256(frozen.read_bytes())!=digest:raise ValueError('Frozen input SHA256 mismatch: '+str(frozen))
            saved=read_json(frozen);fid=entry['frame_id'];objects=saved['parsed']['objects']
            if len({id_key(o['id']) for o in objects})!=len(objects):raise ValueError('Duplicate region IDs')
            originals=[self.florence/fid/name for name in ('original.jpg','original.png','original.jpeg')]
            original=next((p for p in originals if p.is_file()),None)
            image=original or self.path/'images'/(fid+'.jpg')
            if not image.is_file():raise ValueError('Missing saved image: '+str(image))
            variants=[('old',42)]
            if self.kind=='comparison40':
                variants+=[('new',s) for s in ([42,43,44] if entry['repeatability'] else [42])]
            descriptors=[]
            for spec in self.models:
                for prompt,seed in variants:
                    identity=json.dumps([spec['model'],prompt,seed],separators=(',',':'))
                    if self.kind=='baseline200':response=self.path/'responses'/spec['name']/(fid+'.json')
                    elif prompt=='old':response=self.path/'baseline'/spec['name']/(fid+'.json')
                    else:response=self.path/'responses'/spec['name']/str(seed)/(fid+'.json')
                    self.identities[identity]={'model':spec['model'],'prompt_variant':prompt,'seed':seed}
                    descriptors.append({'identity':identity,'path':response,'repeat':seed!=42})
            frame={'frame_id':fid,'dataset':entry.get('dataset','loco'),'environment':entry.get('environment',''),
                'input_sha256':digest,'description':saved['parsed']['scene_description'],'objects':objects,
                'image':image,'original':original is not None,'answers':descriptors}
            self.frames.append(frame)
            for j,obj in enumerate(objects):
                key=sha256(json.dumps([frame['dataset'],fid,id_key(obj['id']),digest],ensure_ascii=False).encode())
                self.flat.append((i,j,key));obj['_review_key']=key
        if not self.flat:raise ValueError('No saved regions to review')

    def region(self,i,j):
        if type(i) is not int or type(j) is not int or not 0<=i<len(self.frames):raise ValueError('Invalid image index')
        f=self.frames[i]
        if not 0<=j<len(f['objects']):raise ValueError('Invalid region index')
        return f,f['objects'][j]

    def extract(self,i,descriptor,obj):
        # Read only this image's saved answers, never all model responses.
        if self.cache_frame!=i:self.raw_cache={};self.cache_frame=i
        path=descriptor['path']
        try:
            if path not in self.raw_cache:self.raw_cache[path]=read_json(path)
            row=self.raw_cache[path]
            if row.get('input_sha256',row.get('caption_sha256'))!=self.frames[i]['input_sha256']:
                raise ValueError('Response input SHA256 does not match this image')
            if row.get('status')!='response_received':raise ValueError(row.get('error','No completed response'))
            value=strict_json(row['content'])
            if not isinstance(value,dict) or not isinstance(value.get('objects'),list):raise ValueError('Missing objects list')
            matches=[r for r in value['objects'] if isinstance(r,dict) and type(r.get('id')) is type(obj['id']) and r.get('id')==obj['id']]
            if len(matches)!=1:raise ValueError('Missing or duplicate response ID for this region')
            prediction=matches[0];classification=prediction.get('classification');reason=prediction.get('reason')
            issues=[]
            if classification not in ('fixed','movable','uncertain'):issues.append('Razred ni veljavno vrnjen.')
            if not isinstance(reason,str) or not reason.strip():issues.append('Utemeljitev ni vrnjena.')
            return {'classification':classification if isinstance(classification,str) else None,
                'reason':reason if isinstance(reason,str) else None,'warning':' '.join(issues)}
        except (OSError,ValueError,KeyError,TypeError):
            return {'classification':None,'reason':None,'warning':'Shranjeni odgovor za ta ID ni uporaben; napaka ostaja v izvornih rezultatih.'}


class Store:
    """Each edit and its audit event commit together; SQLite handles crash recovery."""
    def __init__(self,path,results):
        self.path=Path(path).resolve();self.results=results;self.sid=results.set_id
        if self.path.is_relative_to(results.path) or self.path.is_relative_to(results.florence):
            raise ValueError('Review database must be outside source results directories')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.executescript('''
            CREATE TABLE IF NOT EXISTS sets(id TEXT PRIMARY KEY, manifest_sha TEXT, source TEXT);
            CREATE TABLE IF NOT EXISTS mapping(set_id TEXT, identity TEXT, answer_id TEXT, label TEXT,
                model TEXT, prompt_variant TEXT, seed INTEGER, PRIMARY KEY(set_id,identity), UNIQUE(set_id,answer_id));
            CREATE TABLE IF NOT EXISTS regions(set_id TEXT, region_key TEXT, dataset TEXT, image_id TEXT,
                region_id_json TEXT, input_sha256 TEXT, reference_status TEXT DEFAULT 'unrated', reference_class TEXT,
                issues TEXT DEFAULT '[]', note TEXT DEFAULT '', modified_at TEXT,
                answers_seen_at TEXT, identities_revealed_at TEXT, PRIMARY KEY(set_id,region_key));
            CREATE TABLE IF NOT EXISTS answers(set_id TEXT, region_key TEXT, answer_id TEXT,
                classification_rating TEXT, reason_rating TEXT, modified_at TEXT,
                PRIMARY KEY(set_id,region_key,answer_id));
            CREATE TABLE IF NOT EXISTS cursors(set_id TEXT PRIMARY KEY, state TEXT, modified_at TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, set_id TEXT, region_key TEXT,
                answer_id TEXT, action TEXT, payload TEXT, changed_at TEXT);
            ''')
            c.execute('INSERT OR IGNORE INTO sets VALUES(?,?,?)',(self.sid,results.manifest_sha,str(results.path)))
            existing={r['identity'] for r in c.execute('SELECT identity FROM mapping WHERE set_id=?',(self.sid,))}
            label_index=len(existing)
            for repeat in (False,True):
                identities=[(k,v) for k,v in results.identities.items() if (v['seed']!=42)==repeat and k not in existing]
                secrets.SystemRandom().shuffle(identities)
                for identity,value in identities:
                    label=chr(65+label_index);label_index+=1
                    c.execute('INSERT INTO mapping VALUES(?,?,?,?,?,?,?)',
                        (self.sid,identity,uuid.uuid4().hex,label,value['model'],value['prompt_variant'],value['seed']))
        with self.connect() as c:self.mapping={r['identity']:dict(r) for r in c.execute('SELECT * FROM mapping WHERE set_id=?',(self.sid,))}

    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path,timeout=10);c.row_factory=sqlite3.Row
        c.execute('PRAGMA synchronous=FULL');c.execute('PRAGMA busy_timeout=10000')
        try:
            with c:yield c
        finally:c.close()

    def snapshot(self):
        with self.connect() as c:
            c.execute('BEGIN')
            regions={r['region_key']:dict(r) for r in c.execute('SELECT * FROM regions WHERE set_id=?',(self.sid,))}
            answers={(r['region_key'],r['answer_id']):dict(r) for r in c.execute('SELECT * FROM answers WHERE set_id=?',(self.sid,))}
        return regions,answers

    def cursor(self):
        with self.connect() as c:r=c.execute('SELECT state FROM cursors WHERE set_id=?',(self.sid,)).fetchone()
        i,j,_=self.results.flat[0]
        return json.loads(r[0]) if r else {'frame':i,'region':j,'only_unrated':False,'show_answers':False,'tab':'main'}

    def save_cursor(self,state):
        self.results.region(state['frame'],state['region'])
        with self.connect() as c:c.execute('INSERT INTO cursors VALUES(?,?,?) ON CONFLICT(set_id) DO UPDATE SET state=excluded.state,modified_at=excluded.modified_at',
            (self.sid,json.dumps(state),now()))

    def ensure_region(self,c,frame,obj):
        c.execute('INSERT OR IGNORE INTO regions(set_id,region_key,dataset,image_id,region_id_json,input_sha256) VALUES(?,?,?,?,?,?)',
            (self.sid,obj['_review_key'],frame['dataset'],frame['frame_id'],id_key(obj['id']),frame['input_sha256']))

    def rate(self,i,j,patch):
        frame,obj=self.results.region(i,j);key=obj['_review_key'];stamp=now();values={};answer_id=None
        if set(patch)=={'reference'}:
            value=patch['reference']
            if value not in (None,'fixed','movable','uncertain','unassessable'):raise ValueError('Invalid reference')
            values={'reference_status':'unrated' if value is None else ('unassessable' if value=='unassessable' else 'rated'),
                    'reference_class':value if value in ('fixed','movable','uncertain') else None}
        elif set(patch)=={'issues'}:
            if not isinstance(patch['issues'],list) or not set(patch['issues'])<=ISSUES:raise ValueError('Invalid issue flags')
            values={'issues':json.dumps(sorted(set(patch['issues'])))}
        elif set(patch)=={'note'}:
            if not isinstance(patch['note'],str) or len(patch['note'])>4000:raise ValueError('Note too long')
            values={'note':patch['note']}
        elif set(patch)=={'answer_id','part','value'}:
            allowed={self.mapping[d['identity']]['answer_id'] for d in frame['answers']}
            if patch['answer_id'] not in allowed or patch['part'] not in ('classification','reason') or patch['value'] not in RATINGS:
                raise ValueError('Invalid answer rating')
            answer_id=patch['answer_id']
        else:raise ValueError('Unknown edit')
        with self.connect() as c:
            self.ensure_region(c,frame,obj)
            if answer_id:
                row=c.execute('SELECT reference_status FROM regions WHERE set_id=? AND region_key=?',(self.sid,key)).fetchone()
                if row['reference_status']=='unrated':raise ValueError('Najprej podajte referenčno odločitev.')
                c.execute('INSERT OR IGNORE INTO answers(set_id,region_key,answer_id) VALUES(?,?,?)',(self.sid,key,answer_id))
                # Column is selected from the fixed allow-list above.
                column=patch['part']+'_rating'
                c.execute(f'UPDATE answers SET {column}=?,modified_at=? WHERE set_id=? AND region_key=? AND answer_id=?',
                    (patch['value'],stamp,self.sid,key,answer_id))
            else:
                assignments=','.join(k+'=?' for k in values)
                c.execute(f'UPDATE regions SET {assignments},modified_at=? WHERE set_id=? AND region_key=?',
                    (*values.values(),stamp,self.sid,key))
            c.execute('INSERT INTO events(set_id,region_key,answer_id,action,payload,changed_at) VALUES(?,?,?,?,?,?)',
                (self.sid,key,answer_id,'answer' if answer_id else next(iter(patch)),json.dumps(patch,ensure_ascii=False),stamp))

    def seen(self,i,j,reveal=False):
        frame,obj=self.results.region(i,j);column='identities_revealed_at' if reveal else 'answers_seen_at'
        with self.connect() as c:
            row=c.execute('SELECT reference_status FROM regions WHERE set_id=? AND region_key=?',(self.sid,obj['_review_key'])).fetchone()
            if not row or row['reference_status']=='unrated':raise ValueError('Najprej podajte referenco ali NE MOREM OCENITI.')
            c.execute(f'UPDATE regions SET {column}=coalesce({column},?) WHERE set_id=? AND region_key=?',(now(),self.sid,obj['_review_key']))

    def export(self):
        regions,answers=self.snapshot()
        by_answer={r['answer_id']:r for r in self.mapping.values()};records=[]
        for key,r in regions.items():
            common={'result_set_id':self.sid,'dataset':r['dataset'],'image_id':r['image_id'],'region_id_json':r['region_id_json'],
                'input_sha256':r['input_sha256'],'reference_status':r['reference_status'],'reference_class':r['reference_class'],
                'florence_issues':json.loads(r['issues']),'note':r['note'],'answers_seen_at':r['answers_seen_at'],
                'identities_revealed_at':r['identities_revealed_at'],'reference_modified_at':r['modified_at']}
            records.append({**common,'row_type':'region','modified_at':r['modified_at']})
            for (rkey,answer_id),rating in answers.items():
                if rkey!=key:continue
                identity=by_answer[answer_id]
                records.append({**common,'row_type':'answer','model':identity['model'],'prompt_variant':identity['prompt_variant'],
                    'seed':identity['seed'],'anonymous_label':identity['label'],'classification_rating':rating['classification_rating'],
                    'reason_rating':rating['reason_rating'],'modified_at':rating['modified_at']})
        with self.connect() as c:events=[dict(r) for r in c.execute('SELECT * FROM events WHERE set_id=? ORDER BY id',(self.sid,))]
        return {'schema_version':1,'result_set_id':self.sid,'source_manifest_sha256':self.results.manifest_sha,
            'mapping':list(self.mapping.values()),'reviews':records,'events':events,'cursor':self.cursor()}


class App:
    def __init__(self,results,store):self.results=results;self.store=store;self.token=secrets.token_urlsafe(32)

    def completion(self):
        regions,answers=self.store.snapshot();done={};refs=0;ratings=0;total=0
        for i,j,key in self.results.flat:
            region=regions.get(key,{});has_ref=region.get('reference_status','unrated')!='unrated';refs+=int(has_ref)
            complete=has_ref
            for d in self.results.frames[i]['answers']:
                a=answers.get((key,self.store.mapping[d['identity']]['answer_id']),{})
                for part in ('classification_rating','reason_rating'):
                    total+=1;ratings+=int(a.get(part) is not None);complete=complete and a.get(part) is not None
            done[key]=complete
        return done,{'regions':len(done),'with_reference':refs,'complete_regions':sum(done.values()),'answer_ratings':ratings,'answer_ratings_total':total}

    def view(self,state=None):
        state=state or self.store.cursor();i,j=state['frame'],state['region'];f,o=self.results.region(i,j)
        regions,answers=self.store.snapshot();done,progress=self.completion();review=regions.get(o['_review_key'],{})
        return {'cursor':state,'image_count':len(self.results.frames),'image_id':f['frame_id'],'dataset':f['dataset'],
            'environment':f['environment'],'description':f['description'],'image_url':'/image/'+str(i),'original_image':f['original'],
            'regions':[{'index':n,'id':r['id'],'label':r['label'],'bbox':r.get('bbox_xyxy'),
                'complete':done[r['_review_key']]} for n,r in enumerate(f['objects'])],
            'selected':{'index':j,'id':o['id'],'label':o['label'],'bbox':o.get('bbox_xyxy'),'observation':o.get('observation','')},
            'review':{'reference_status':review.get('reference_status','unrated'),'reference_class':review.get('reference_class'),
                'issues':json.loads(review.get('issues','[]')),'note':review.get('note',''),'modified_at':review.get('modified_at')},
            'progress':progress,'has_repeats':any(d['repeat'] for d in f['answers'])}

    def navigate(self,data):
        state=self.store.cursor();action=data.get('action');only=bool(data.get('only_unrated',state['only_unrated']))
        done,_=self.completion();flat=self.results.flat;position=next(n for n,(i,j,_) in enumerate(flat) if (i,j)==(state['frame'],state['region']))
        target=(state['frame'],state['region'])
        if action=='select':self.results.region(data['frame'],data['region']);target=(data['frame'],data['region'])
        elif action in ('next_region','previous_region','next_image','previous_image'):
            forward=action.startswith('next');indices=range(position+1,len(flat)) if forward else range(position-1,-1,-1)
            for n in indices:
                i,j,key=flat[n]
                if action.endswith('image') and i==state['frame']:continue
                if only and done[key]:continue
                target=(i,j);break
        elif action=='filter':
            if only and done[flat[position][2]]:
                target=next(((i,j) for i,j,key in flat if not done[key]),target)
        else:raise ValueError('Invalid navigation')
        new={'frame':target[0],'region':target[1],'only_unrated':only,'show_answers':False,'tab':'main'}
        self.store.save_cursor(new);return self.view(new)

    def answers(self,repeat=False,reveal=False,seed=43):
        if seed not in (43,44):raise ValueError('Invalid repetition seed')
        state=self.store.cursor();i,j=state['frame'],state['region'];f,o=self.results.region(i,j)
        self.store.seen(i,j,reveal=reveal)
        _,ratings=self.store.snapshot();rows=[]
        for descriptor in f['answers']:
            if descriptor['repeat']!=repeat:continue
            identity=self.store.mapping[descriptor['identity']]
            if repeat and identity['seed']!=seed:continue
            prediction=self.results.extract(i,descriptor,o)
            rating=ratings.get((o['_review_key'],identity['answer_id']),{})
            item={'answer_id':identity['answer_id'],'label':identity['label'],**prediction,
                'classification_rating':rating.get('classification_rating'),'reason_rating':rating.get('reason_rating')}
            if reveal:item['identity']={k:identity[k] for k in ('model','prompt_variant','seed')}
            rows.append(item)
        state.update(show_answers=True,tab='repeat' if repeat else 'main',repeat_seed=seed);self.store.save_cursor(state)
        return {'answers':sorted(rows,key=lambda r:r['label']),'repeat':repeat,'seed':seed if repeat else 42,'revealed':reveal}

    def hide_answers(self):
        state=self.store.cursor();state['show_answers']=False;self.store.save_cursor(state)
        return {'hidden':True}


def make_server(app,host='127.0.0.1',port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,data,mime='application/json; charset=utf-8',status=200,download=None):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if download:self.send_header('Content-Disposition','attachment; filename="'+download+'"')
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                path=urlparse(self.path).path
                if path=='/':
                    text=Path(__file__).with_suffix('.html').read_text().replace('__TOKEN__',app.token)
                    return self.send(text.encode(),'text/html; charset=utf-8')
                if path=='/api/state':return self.send(app.view())
                if path.startswith('/image/'):
                    i=int(path.split('/')[-1]);f=app.results.frames[i]
                    if i<0:raise ValueError('Invalid image')
                    return self.send(f['image'].read_bytes(),mimetypes.guess_type(f['image'].name)[0] or 'image/jpeg')
                if path in ('/export.csv','/export.json'):
                    data=app.store.export();name='manual-review-'+app.results.set_id[:10]
                    if path.endswith('json'):return self.send(data,download=name+'.json')
                    fields=['result_set_id','row_type','dataset','image_id','region_id_json','input_sha256','anonymous_label',
                        'model','prompt_variant','seed','reference_status','reference_class','classification_rating','reason_rating',
                        'florence_issues','note','modified_at','reference_modified_at','answers_seen_at','identities_revealed_at']
                    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader()
                    for r in data['reviews']:writer.writerow({**r,'florence_issues':json.dumps(r['florence_issues'],ensure_ascii=False)})
                    return self.send(output.getvalue().encode(),'text/csv; charset=utf-8',download=name+'.csv')
                self.send({'error':'Not found'},status=404)
            except (ValueError,KeyError,IndexError,OSError) as exc:self.send({'error':str(exc)},status=400)
        def do_POST(self):
            try:
                if self.headers.get('X-Review-Token')!=app.token:return self.send({'error':'Invalid local session'},status=403)
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('Invalid request size')
                data=strict_json(self.rfile.read(length).decode());path=urlparse(self.path).path
                if path=='/api/navigate':return self.send(app.navigate(data))
                if path=='/api/rate':
                    app.store.rate(data['frame'],data['region'],data['patch']);return self.send({'saved':True,'progress':app.completion()[1]})
                if path=='/api/answers':return self.send(app.answers(bool(data.get('repeat')),bool(data.get('reveal')),data.get('seed',43)))
                if path=='/api/hide':return self.send(app.hide_answers())
                self.send({'error':'Not found'},status=404)
            except (ValueError,KeyError,TypeError,IndexError) as exc:self.send({'error':str(exc)},status=400)
            except sqlite3.Error as exc:self.send({'error':'Shranjevanje ni uspelo: '+str(exc)},status=500)
    return ThreadingHTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=DEFAULT_RESULTS)
    parser.add_argument('--florence-dir',type=Path,default=DEFAULT_FLORENCE)
    parser.add_argument('--db',type=Path,default=DEFAULT_DB)
    parser.add_argument('--host',choices=('127.0.0.1','localhost'),default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765);parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args();results=Results(args.results_dir,args.florence_dir);store=Store(args.db,results)
    server=make_server(App(results,store),args.host,args.port);url=f'http://{args.host}:{server.server_address[1]}/'
    print(f'Pregledovalnik: {url}\nOcene: {store.path}\nSlike: {len(results.frames)}. Ustavitev: Ctrl+C.',flush=True)
    if not args.no_browser:threading.Timer(.3,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
