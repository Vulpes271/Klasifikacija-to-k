"""Only the new LILocBench adapter; no old experiments or model calls."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import zipfile

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from scripts import prepare_lilocbench as prepare
from llm_object_classifier import lilocbench
from llm_object_classifier.review_viewer import Results,Store,App
from llm_object_classifier.experiment import sha256


class LILocBenchTests(unittest.TestCase):
    def test_range_200_is_aborted_before_body_read(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(prepare,'OUT',Path(tmp)),patch.object(prepare,'CACHE',Path(tmp)/'cache'):
            response=Mock(status=200,headers={});response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
            with patch('urllib.request.urlopen',return_value=response):
                transfers=prepare.Transfers()
                with self.assertRaisesRegex(ValueError,'Range not honored'):transfers.get('https://example.invalid/a.zip',0,1)
            response.read.assert_not_called();self.assertEqual(transfers.rows[0]['payload_bytes'],0)

    def test_selective_zip64_index_and_member_crc(self):
        buffer=io.BytesIO()
        with patch.object(zipfile,'ZIP64_LIMIT',1):
            with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:z.writestr('camera_front/color/images/1.png',b'only selected RGB bytes')
        archive=buffer.getvalue()
        def serve(request,timeout=None):
            start,end=map(int,request.get_header('Range').removeprefix('bytes=').split('-'))
            response=Mock(status=206,headers={'Content-Range':f'bytes {start}-{end}/{len(archive)}'})
            response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
            response.read=Mock(return_value=archive[start:end+1]);return response
        with tempfile.TemporaryDirectory() as tmp,patch.object(prepare,'OUT',Path(tmp)),patch.object(prepare,'CACHE',Path(tmp)/'cache'),patch('urllib.request.urlopen',side_effect=serve):
            transfers=prepare.Transfers();stream=prepare.RemoteZip('https://example.invalid/a.zip',transfers)
            with zipfile.ZipFile(stream) as z:info=z.infolist()[0]
            self.assertEqual(prepare.member(stream,info),b'only selected RGB bytes')
            self.assertTrue(all(r['status']==206 for r in transfers.rows))
            self.assertTrue(all(r['payload_bytes']==r['requested_bytes'] for r in transfers.rows))
            count=len(transfers.rows);prepare.member(stream,info);self.assertEqual(len(transfers.rows),count)

    def test_pose_slerp_and_rgb_extrinsic_are_composed_at_image_time(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(prepare,'OUT',Path(tmp)):
            folder=Path(tmp)/'metadata/mapping';folder.mkdir(parents=True)
            def edge(parent,child,t,q):return {'parent':parent,'child':child,'translation':dict(zip('xyz',t)),'rotation':dict(zip('xyzw',q))}
            edges=[edge('base_link','camera_front_link',[.1,0,.6],[0,0,0,1]),
                edge('camera_front_link','camera_front_color_frame',[0,0,0],[0,0,0,1]),
                edge('camera_front_color_frame','camera_front_color_optical_frame',[0,0,0],[-.5,.5,-.5,.5])]
            (folder/'transformations.yaml').write_text(yaml.safe_dump(edges))
            end=Rotation.from_euler('z',90,degrees=True).as_quat()
            np.savetxt(folder/'gt_poses.txt',np.array([[0,0,0,0,0,0,0,1],[.05,.2,0,0,*end]]))
            views,audit=prepare.views('mapping',[{'path':'camera_front/color/images/0.012500000.png'}])
            self.assertEqual(len(views),1);v=views[0];angle=np.deg2rad(22.5)
            np.testing.assert_allclose(v['camera_position_xyz'],[.05+.1*np.cos(angle),.1*np.sin(angle),.6],atol=1e-8)
            np.testing.assert_allclose(v['camera_forward_world'],[np.cos(angle),np.sin(angle),0],atol=1e-8)
            self.assertAlmostEqual(v['gt_time_offset_seconds'],.0125);self.assertAlmostEqual(v['interpolation_alpha'],.25)
            self.assertEqual(audit['camera_transform']['frame'],'camera_front_color_optical_frame')
            self.assertAlmostEqual(prepare.angle([0,0,0,1],[0,0,0,-1]),0)

    def test_native_tokenizer_http_chunking_without_inference(self):
        body=b'{"tokens":[1,2,3]}'
        response=b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n'+f'{len(body):x}'.encode()+b'\r\n'+body+b'\r\n0\r\n\r\n'
        fake=Mock(returncode=0,stdout=response,stderr=b'')
        with patch.object(lilocbench.subprocess,'run',return_value=fake) as process:
            self.assertEqual(lilocbench.runner_http(12345,'/tokenize',{'content':'text','add_special':False})['tokens'],[1,2,3])
            call=process.call_args
            self.assertNotIn('text',str(call.args[0]));self.assertIn(b'"text"',call.kwargs['input'])

    def test_separate_single_prompt_review_namespace_and_shared_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'dataset';(base/'images').mkdir(parents=True)
            image=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
            (base/'images/view-000001.png').write_bytes(image)
            frozen=base/'frozen.json';frozen.write_text(json.dumps({'parsed':{'scene_description':'A room.','objects':[{'id':1,'label':'chair','bbox_xyxy':[0,0,1,1]}]}}))
            digest=sha256(frozen.read_bytes());models=[{'name':str(i),'model':f'model-{i}'} for i in range(5)]
            manifest={'protocol':'lilocbench-v1','config':{'models':models},'inputs':[{'frame_id':'view-000001','dataset':'lilocbench','frozen':frozen.name,'input_sha256':digest}]}
            (base/'manifest.json').write_text(json.dumps(manifest))
            for spec in models:
                path=base/'responses'/spec['name']/'view-000001.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'status':'response_received','input_sha256':digest,'finish_reason':'stop','response':{'done':True,'done_reason':'stop'},
                    'content':json.dumps({'objects':[{'id':1,'classification':'movable','reason':'Transportable chair.'}]})}))
            results=Results(base);self.assertEqual(results.kind,'lilocbench');self.assertEqual(len(results.identities),5)
            self.assertTrue(all(v['prompt_variant']=='new' for v in results.identities.values()))
            store=Store(root/'separate-review.sqlite3',results);app=App(results,store)
            store.rate(0,0,{'reference':'movable'})
            self.assertEqual(app.completion('reference')[1]['complete_regions'],1)
            self.assertEqual(app.completion('reasons')[1]['complete_regions'],0)
            self.assertTrue(all(v['computed_match']==1 for v in store.computed().values()))
            app.navigate({'action':'mode','mode':'reasons'});self.assertEqual(len(app.answers()['groups']),1)
            self.assertEqual(len(app.answers()['groups'][0]['members']),5)
