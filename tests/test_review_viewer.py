"""Offline fixtures only: never create or modify a user's review database."""
import base64
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
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
        self.assertEqual(a.navigate({'action':'previous_image'})['cursor']['frame'],0)

    def test_http_and_exports(self):
        r,s,a=self.fixture();server=make_server(a,port=0)
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
            self.assertEqual(len(export['reviews']),2);self.assertTrue(export['mapping'])
            with urlopen(base+'/export.csv') as response:rows=list(csv.DictReader(io.StringIO(response.read().decode())))
            self.assertEqual(rows[0]['reference_class'],'uncertain')
            self.assertEqual(rows[1]['reason_rating'],'incorrect');self.assertEqual(rows[1]['classification_rating'],'')
            self.assertEqual(rows[1]['input_sha256'],r.frames[0]['input_sha256'])
        finally:server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':unittest.main()
