"""Offline fixtures only: never create or modify a user's review database."""
import base64
import ctypes
import ctypes.util
import csv
import io
import json
import sqlite3
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_object_classifier.review_viewer import App, Results, Store, make_server, sha256


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.florence=self.root/'florence';self.florence.mkdir()

    def tearDown(self):self.temp.cleanup()

    def fixture(self,kind='comparison40'):
        folder=self.root/kind;folder.mkdir()
        models=[{'name':f'm{i}','model':f'private-model-{i}'} for i in range(3)]
        inputs=[]
        for n in range(2):
            fid=f'image-{n}';source=self.florence/fid;source.mkdir(exist_ok=True)
            image=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
            (source/'original.png').write_bytes(image)
            frozen=folder/f'{fid}.json'
            objects=[{'id':7,'label':'shelf','bbox_xyxy':[0,0,1,1],'observation':'Mounted shelf'},
                     {'id':'7','label':'box','bbox_xyxy':[0,0,1,1],'observation':'A box'}]
            frozen.write_text(json.dumps({'parsed':{'objects':objects,'scene_description':'Full scene description.'}}))
            digest=sha256(frozen.read_bytes())
            inputs.append({'frame_id':fid,'dataset':'fixture','environment':'s1','frozen':frozen.name,
                'input_sha256':digest,'sha256':digest,'repeatability':n==0})
            for model in models:
                destinations=[folder/'responses'/model['name']/(fid+'.json')] if kind=='baseline200' else [
                    folder/'baseline'/model['name']/(fid+'.json'),
                    *[folder/'responses'/model['name']/str(seed)/(fid+'.json') for seed in ([42,43,44] if n==0 else [42])]]
                for dest in destinations:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_text(json.dumps({'status':'response_received','input_sha256':digest,'content':json.dumps({
                        'objects':[{'id':7,'classification':'fixed','reason':'Entire unchanged explanation.\nSecond line.'},
                                   {'id':'7','classification':'movable','reason':'Different typed ID.'}]})}))
        manifest={'protocol':'prompt40-v1','config':{'models':models},'inputs':inputs}
        (folder/('manifest.json' if kind=='comparison40' else 'experiment200-manifest.json')).write_text(json.dumps(manifest))
        results=Results(folder,self.florence);store=Store(self.root/(kind+'.sqlite3'),results)
        return results,store,App(results,store)

    def test_adapters_blind_gate_and_separate_repetitions(self):
        for kind,count,repeats in [('comparison40',6,3),('baseline200',3,0)]:
            with self.subTest(kind=kind):
                r,s,a=self.fixture(kind)
                self.assertEqual(len(r.frames),2);self.assertEqual(len(r.flat),4);self.assertFalse(r.raw_cache)
                self.assertEqual(a.view()['review']['reference_status'],'unrated')
                self.assertNotIn('private-model',json.dumps(a.view()))
                self.assertEqual(s.export()['reviews'],[])
                a.navigate({'action':'mode','mode':'reasons'})
                with self.assertRaises(ValueError):a.answers()
                s.rate(0,0,{'reference':'uncertain'})
                data=a.answers();self.assertEqual(len(data['answers']),count)
                self.assertNotIn('private-model',json.dumps(data));self.assertNotIn('prompt_variant',json.dumps(data))
                self.assertTrue(all(x['classification']=='fixed' for x in data['answers']))
                self.assertTrue(all(x['reason'].endswith('Second line.') for x in data['answers']))
                self.assertEqual(len(a.answers(repeat=True)['answers']),repeats)
                self.assertEqual(len(a.answers(repeat=True,seed=44)['answers']),repeats)
                self.assertIn('private-model',json.dumps(a.answers(reveal=True)))
                a.hide_answers();self.assertFalse(Store(s.path,r).cursor()['show_answers'])
                a.navigate({'action':'next_region'})
                s.rate(0,1,{'reference':'unassessable'})
                self.assertTrue(all(x['classification']=='movable' for x in a.answers()['answers']))

    def test_save_reload_edit_and_source_preservation(self):
        r,s,a=self.fixture()
        a.navigate({'action':'mode','mode':'reasons'})
        before={p:p.read_bytes() for p in r.path.rglob('*') if p.is_file()}
        s.rate(0,0,{'reference':'fixed'})
        answer=a.answers()['answers'][0]['answer_id']
        s.rate(0,0,{'answer_id':answer,'part':'classification','value':'correct'})
        s.rate(0,0,{'answer_id':answer,'part':'reason','value':'incorrect'})
        s.rate(0,0,{'issues':['wrong_box','group']});s.rate(0,0,{'note':'Prva opomba'})
        s.rate(0,0,{'reference':'uncertain'});s.rate(0,0,{'note':'Popravljena opomba'})
        a.navigate({'action':'next_region'})
        again=Store(s.path,r);app=App(r,again)
        self.assertEqual(again.mapping,s.mapping);self.assertEqual(again.cursor()['region'],1)
        app.navigate({'action':'previous_region'});view=app.view()
        self.assertEqual(view['review']['reference_class'],'uncertain')
        self.assertEqual(view['review']['note'],'Popravljena opomba')
        rows=app.answers()['answers'];card=next(x for x in rows if x['answer_id']==answer)
        self.assertEqual(card['classification_rating'],'correct');self.assertEqual(card['reason_rating'],'incorrect')
        again.rate(0,0,{'reference':'unassessable'})
        self.assertIsNone(app.view()['review']['reference_class'])
        self.assertEqual(app.view()['review']['reference_status'],'unassessable')
        again.rate(0,0,{'answer_id':answer,'part':'reason','value':None})
        self.assertIsNone(next(x for x in app.answers()['answers'] if x['answer_id']==answer)['reason_rating'])
        export=again.export();self.assertGreaterEqual(len(export['events']),9)
        row=next(x for x in export['reviews'] if x['row_type']=='answer')
        for key in ('input_sha256','region_id_json','model','prompt_variant','seed','modified_at'):
            self.assertIsNotNone(row[key])
        self.assertEqual(before,{p:p.read_bytes() for p in r.path.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError):Store(r.path/'bad.sqlite3',r)

    def test_completion_filter_and_mismatched_input(self):
        r,s,a=self.fixture();s.rate(0,0,{'reference':'unassessable'})
        for d in r.frames[0]['answers']:
            answer=s.mapping[d['identity']]['answer_id']
            for part in ('classification','reason'):s.rate(0,0,{'answer_id':answer,'part':part,'value':'unassessable'})
        self.assertEqual(a.completion()[1]['complete_regions'],1)
        view=a.navigate({'action':'filter','only_unrated':True})
        self.assertEqual(view['cursor']['region'],1)
        self.assertFalse(view['cursor']['show_answers'])
        s.rate(0,1,{'reference':'fixed'})
        d=r.frames[0]['answers'][0];row=json.loads(d['path'].read_text());row['input_sha256']='wrong'
        d['path'].write_text(json.dumps(row));r.raw_cache={}
        broken=r.extract(0,d,r.frames[0]['objects'][1])
        self.assertIsNone(broken['classification']);self.assertTrue(broken['warning'])
        view=a.navigate({'action':'next_image'});self.assertEqual(view['cursor']['frame'],1)
        self.assertEqual(a.navigate({'action':'previous_image','only_unrated':False})['cursor']['frame'],0)

    def test_http_and_exports(self):
        r,s,a=self.fixture();server=make_server(a,port=0)
        a.navigate({'action':'mode','mode':'reasons'})
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_address[1]}'
        def post(path,data,token=None):
            return urlopen(Request(base+path,data=json.dumps(data).encode(),headers={
                'Content-Type':'application/json','X-Review-Token':token or a.token}),timeout=5)
        try:
            with urlopen(base) as response:html=response.read().decode()
            self.assertIn(a.token,html);self.assertNotIn('__TOKEN__',html)
            with urlopen(base+'/api/state') as response:self.assertEqual(json.load(response)['image_count'],2)
            with urlopen(base+'/image/0') as response:self.assertEqual(response.headers['Content-Type'],'image/png')
            with self.assertRaises(HTTPError) as error:post('/api/rate',{},token='bad')
            self.assertEqual(error.exception.code,403)
            error.exception.close()
            with post('/api/rate',{'frame':0,'region':0,'patch':{'reference':'uncertain'}}) as response:
                self.assertTrue(json.load(response)['saved'])
            with post('/api/answers',{}) as response:card=json.load(response)['answers'][0]
            with post('/api/rate',{'frame':0,'region':0,'patch':{
                'answer_id':card['answer_id'],'part':'reason','value':'incorrect'}}) as response:self.assertEqual(response.status,200)
            with urlopen(base+'/export.json') as response:export=json.load(response)
            self.assertEqual(len(export['reviews']),13);self.assertTrue(export['mapping'])
            with urlopen(base+'/export.csv') as response:rows=list(csv.DictReader(io.StringIO(response.read().decode())))
            self.assertEqual(rows[0]['reference_class'],'uncertain')
            rated=next(row for row in rows if row.get('reason_rating')=='incorrect')
            self.assertEqual(rated['classification_rating'],'')
            self.assertEqual(rated['predicted_classification'],'fixed');self.assertEqual(rated['computed_match'],'False')
            self.assertEqual(rows[1]['input_sha256'],r.frames[0]['input_sha256'])
        finally:server.shutdown();server.server_close();thread.join(timeout=5)

    def test_additive_v1_migration_and_changed_reference(self):
        r,s,a=self.fixture();s.rate(0,0,{'reference':'fixed'})
        aid=next(iter(s.mapping.values()))['answer_id']
        s.rate(0,0,{'answer_id':aid,'part':'classification','value':'incorrect'})
        s.rate(0,0,{'answer_id':aid,'part':'reason','value':'correct'})
        s.rate(0,0,{'note':'Keep my previous note'})
        tables=['regions','answers','events','mapping','cursors']
        def human_rows(store):
            with store.connect() as c:return {t:[tuple(row) for row in c.execute('SELECT * FROM '+t)] for t in tables}
        # Recreate the old database structure by removing only the additive v2 table.
        with s.connect() as c:
            c.execute('DROP TABLE computed_matches')
            c.execute('INSERT OR REPLACE INTO cursors VALUES(?,?,?)',(s.sid,json.dumps({
                'frame':0,'region':0,'only_unrated':True,'show_answers':True,'tab':'main'}),'old-time'))
        before=human_rows(s);upgraded=Store(s.path,r)
        self.assertEqual(before,human_rows(upgraded));self.assertEqual(upgraded.mapping,s.mapping)
        self.assertEqual(upgraded.cursor()['mode'],'reference');self.assertFalse(upgraded.cursor()['show_answers'])
        self.assertTrue(all(row['computed_match']==1 for row in upgraded.computed().values()))
        upgraded.rate(0,0,{'reference':'movable'})
        self.assertTrue(all(row['computed_match']==0 for row in upgraded.computed().values()))
        upgraded.rate(0,0,{'reference':'unassessable'})
        self.assertTrue(all(row['computed_match'] is None and row['computed_match_status']=='no_reference' for row in upgraded.computed().values()))
        upgraded.rate(0,0,{'reference':'uncertain'})
        self.assertTrue(all(row['computed_match']==0 for row in upgraded.computed().values()))
        manual=upgraded.snapshot()[1][(r.flat[0][2],aid)]
        self.assertEqual(manual['classification_rating'],'incorrect');self.assertEqual(manual['reason_rating'],'correct')
        self.assertEqual(Store(s.path,r).snapshot()[1],upgraded.snapshot()[1])

    def test_exact_groups_preserve_conflicts_and_apply_atomically(self):
        r,s,a=self.fixture();a.navigate({'action':'mode','mode':'reasons'});s.rate(0,0,{'reference':'fixed'})
        groups=a.answers()['groups'];self.assertEqual(len(groups),1);group=groups[0]
        self.assertEqual(len(group['members']),12);self.assertEqual(len(group['labels']),6)
        aid=group['members'][0]['answer_id']
        s.rate(0,0,{'answer_id':aid,'part':'reason','value':'incorrect'})
        self.assertTrue(a.answers()['groups'][0]['mixed_ratings'])
        self.assertEqual(len(s.snapshot()[1]),1) # viewing never merges old human scores
        s.rate_group(0,0,group['group_id'],'correct')
        self.assertTrue(all(row['reason_rating']=='correct' for row in s.snapshot()[1].values()))
        self.assertEqual(len(s.snapshot()[1]),12)
        self.assertFalse(a.answers(repeat=True,seed=44)['groups'][0]['mixed_ratings'])
        self.assertEqual(a.answers()['groups'][0]['reason_rating'],'correct')
        with self.assertRaises(ValueError):s.rate_group(0,1,group['group_id'],'incorrect')
        before=s.snapshot();events=s.export()['events']
        with s.connect() as c:c.execute("CREATE TRIGGER reject_reason BEFORE UPDATE ON answers WHEN NEW.answer_id='"+aid+"' BEGIN SELECT RAISE(ABORT,'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):s.rate_group(0,0,group['group_id'],'incorrect')
        self.assertEqual(before,s.snapshot());self.assertEqual(events,s.export()['events'])

    def test_groups_require_literal_class_and_reason_and_same_input(self):
        r,s,a=self.fixture();descriptors=r.frames[0]['answers']
        changes=[('fixed','Literal.'),('fixed','Literal. '),('movable','Literal.'),('fixed',None),('fixed',None)]
        for d,(classification,reason) in zip(descriptors,changes):
            row=json.loads(d['path'].read_text());content=json.loads(row['content'])
            content['objects'][0].update(classification=classification,reason=reason)
            row['content']=json.dumps(content);d['path'].write_text(json.dumps(row))
        r.raw_cache={};groups=s.groups(0,0)
        self.assertEqual(len(groups),6) # three different literals, two missing reasons, remaining original
        self.assertNotEqual(s.groups(0,1)[0]['group_id'],groups[0]['group_id'])
        # A mismatched input is not linked even when its original text was identical.
        d=descriptors[-1];row=json.loads(d['path'].read_text());row['input_sha256']='different'
        d['path'].write_text(json.dumps(row));r.raw_cache={}
        self.assertEqual(len(s.groups(0,0)),7)

    def test_modes_filter_wrap_resume_and_atomic_reference_failure(self):
        r,s,a=self.fixture();self.assertEqual(s.cursor()['mode'],'reference')
        s.rate(0,0,{'reference':'fixed'})
        done,p=a.completion();self.assertTrue(done[r.flat[0][2]]);self.assertEqual(p['with_reference'],1)
        self.assertEqual(p['reason_ratings'],0)
        with self.assertRaises(ValueError):a.answers()
        self.assertEqual(a.navigate({'action':'next_unrated'})['cursor']['region'],1)
        a.navigate({'action':'mode','mode':'reasons'})
        self.assertEqual(a.completion()[1]['complete_regions'],0)
        self.assertEqual(a.navigate({'action':'filter','only_unrated':True})['cursor']['region'],1)
        self.assertEqual(a.navigate({'action':'previous_region'})['cursor']['region'],0)
        group=a.answers()['groups'][0];s.rate_group(0,0,group['group_id'],'unassessable')
        self.assertEqual(a.completion()[1]['reason_regions'],1)
        a.navigate({'action':'mode','mode':'reference','only_unrated':False})
        for i,j,_ in r.flat:s.rate(i,j,{'reference':'unassessable'})
        self.assertEqual(a.completion()[1]['complete_regions'],4)
        a.navigate({'action':'select','frame':1,'region':1})
        self.assertEqual(a.navigate({'action':'next_unrated'})['cursor']['frame'],1)
        self.assertEqual(Store(s.path,r).cursor(),s.cursor())
        before=s.snapshot();events=s.export()['events']
        with patch.object(r,'extract',side_effect=OSError('temporary read failure')):
            with self.assertRaises(OSError):s.rate(1,1,{'reference':'fixed'})
        self.assertEqual(before,s.snapshot());self.assertEqual(events,s.export()['events'])

    def test_shortcuts_execute_in_javascript(self):
        # Use an existing system JS engine; viewer itself has no JS/Python dependencies.
        libname=ctypes.util.find_library('javascriptcoregtk-4.1')
        if not libname:self.skipTest('System JavaScriptCore unavailable')
        js=ctypes.CDLL(libname);ptr=ctypes.c_void_p
        js.JSGlobalContextCreate.argtypes=[ptr];js.JSGlobalContextCreate.restype=ptr
        js.JSStringCreateWithUTF8CString.argtypes=[ctypes.c_char_p];js.JSStringCreateWithUTF8CString.restype=ptr
        js.JSEvaluateScript.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.POINTER(ptr)];js.JSEvaluateScript.restype=ptr
        js.JSStringRelease.argtypes=[ptr];js.JSGlobalContextRelease.argtypes=[ptr]
        ctx=js.JSGlobalContextCreate(None)
        html=Path('llm_object_classifier/review_viewer.html').read_text();script=html.split('<script>')[1].split('</script>')[0]
        logic=script.split('// SHORTCUT_LOGIC_START:')[1].split('// SHORTCUT_LOGIC_END')[0].split('\n',1)[1]
        harness='''
        function check(ok){if(!ok)throw Error('Shortcut assertion failed');}
        function event(key,extra={}){return {key,target:{isContentEditable:false,closest:()=>null},...extra};}
        for(const [key,expected] of [['1','fixed'],['2','movable'],['3','uncertain'],['4','unassessable']])check(shortcutAction(event(key)).reference===expected);
        check(shortcutAction(event('ArrowLeft')).navigate==='previous_region');
        check(shortcutAction(event('ArrowRight',{shiftKey:true})).navigate==='next_image');
        check(shortcutAction(event('ArrowLeft',{shiftKey:true})).navigate==='previous_image');
        check(shortcutAction(event('ArrowRight')).navigate==='next_region');
        check(shortcutAction(event(' ')).toggle);
        for(const key of ['1','ArrowRight',' ']){
          check(shortcutAction(event(key,{repeat:true}))===null);
          check(shortcutAction(event(key),true)===null);
          check(shortcutAction(event(key,{target:{isContentEditable:true}}))===null);
          check(shortcutAction(event(key,{target:{isContentEditable:false,closest:selector=>selector.includes('textarea')?{}:null}}))===null);
          for(const flag of ['ctrlKey','altKey','metaKey','isComposing'])check(shortcutAction(event(key,{[flag]:true}))===null);
        }
        '''
        # Parse the complete UI script too, catching syntax errors outside the shortcut helper.
        code='new Function('+json.dumps(script)+');\n'+logic+harness
        def execute(code):
            text=js.JSStringCreateWithUTF8CString(code.encode());exception=ptr()
            try:
                js.JSEvaluateScript(ctx,text,None,None,1,ctypes.byref(exception))
                self.assertFalse(exception.value,'JavaScript syntax/interaction test failed')
            finally:js.JSStringRelease(text)
        try:
            execute(code)
            # Exercise the actual UI reference handler with a delayed successful save.
            handler=script.split('async function setReference(')[1].split('function renderReference(){')[0]
            execute('''
                let state={review:{},cursor:{mode:'reference'}},pending=0,moving=false,failures=[],answerData=null;
                let trace=[],resolveSave;
                function $(id){return {hidden:true};}
                function renderReference(){}
                function save(){return new Promise(resolve=>{resolveSave=()=>{trace.push('saved');resolve();};});}
                async function navigate(action){trace.push(action);}
                async function loadAnswers(){trace.push('reveal');}
                async function setReference('''+handler+'''
                setReference('fixed');check(trace.length===0);resolveSave();
            ''')
            execute("check(JSON.stringify(trace)===JSON.stringify(['saved','next_unrated']));")
            execute("trace=[];save=async()=>{failures.push('disk error');};setReference('movable');")
            execute("check(trace.length===0);check(failures.length===1);check(state.review.reference_class==='movable');")
            execute("failures=[];state.cursor.mode='reasons';save=async()=>{trace.push('saved');};setReference('uncertain');")
            execute("check(JSON.stringify(trace)===JSON.stringify(['saved']));")
        finally:js.JSGlobalContextRelease(ctx)


if __name__=='__main__':unittest.main()
