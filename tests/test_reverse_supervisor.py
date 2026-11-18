import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from maintenance import run_reverse_experiment as supervisor


class SupervisorTests(unittest.TestCase):
    def fixture(self,root):
        config=root/'config.json';config.write_text(json.dumps({'output':'results','cache':'cache','seeds':[42]}))
        output=root/'results';output.mkdir();(output/'progress.json').write_text(json.dumps({'camera':'front','batch':1}))
        return config,output

    def run_supervisor(self,root,config,fake):
        with patch.object(supervisor,'ROOT',root),patch.object(sys,'argv',['supervisor','--config',str(config)]),\
             patch.object(supervisor.subprocess,'run',side_effect=fake),patch.object(supervisor.time,'sleep'):
            return supervisor.main()

    def test_native_sam_retry_archives_partial_and_preserves_accepted_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,out=self.fixture(root);calls=[]
            partial=root/'cache/models/front/full/0001/work/masks';partial.mkdir(parents=True);(partial/'partial.png').write_bytes(b'partial')
            accepted=root/'cache/models/front/full/0000/work/masks';accepted.mkdir(parents=True);(accepted/'segmentation.json').write_text('{}')
            response=root/'cache/received-llm.json';response.write_text('{"status":"response_received"}')
            def fake(command,**kwargs):
                calls.append(command)
                if len(calls)==1:
                    (out/'error.json').write_text(json.dumps({'error':'klasifikacija-sam2:local returned non-zero exit status 139.'}))
                    return SimpleNamespace(returncode=1)
                self.assertFalse(partial.exists());self.assertTrue((accepted/'segmentation.json').exists())
                self.assertEqual(response.read_text(),'{"status":"response_received"}')
                (out/'metrics.json').write_text(json.dumps({'complete':True,'runs':{'all':[{}]*28}}))
                return SimpleNamespace(returncode=0)
            self.assertEqual(self.run_supervisor(root,config,fake),0)
            state=json.loads((out/'job.json').read_text());self.assertEqual(state['status'],'complete')
            self.assertEqual(len(state['recoveries']),1);self.assertEqual(len(calls),2)
            self.assertEqual(len(list((out/'runtime-recoveries').glob('**/partial.png'))),1)

    def test_semantic_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,out=self.fixture(root);calls=[]
            def fake(command,**kwargs):
                calls.append(command);(out/'error.json').write_text(json.dumps({'error':'LLM model digest changed'}))
                return SimpleNamespace(returncode=1)
            self.assertEqual(self.run_supervisor(root,config,fake),1)
            self.assertEqual(len(calls),1);self.assertEqual(json.loads((out/'job.json').read_text())['status'],'failed')

    def test_same_native_failure_stops_after_three_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,out=self.fixture(root);calls=[]
            def fake(command,**kwargs):
                calls.append(command);(out/'error.json').write_text(json.dumps({'error':'klasifikacija-sam2:local returned non-zero exit status 139.'}))
                return SimpleNamespace(returncode=1)
            self.assertEqual(self.run_supervisor(root,config,fake),1)
            self.assertEqual(len(calls),4);self.assertIn('Three retries exhausted',json.loads((out/'job.json').read_text())['reason'])


if __name__=='__main__':unittest.main()
