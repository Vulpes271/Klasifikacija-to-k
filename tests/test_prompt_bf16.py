"""BF16 context estimation uses saved counts only with matching text provenance."""
import json
from pathlib import Path
import tempfile
import unittest

from llm_object_classifier.prompt_comparison_bf16 import recorded_budget


class BF16BudgetTests(unittest.TestCase):
    def test_recorded_count_is_estimate_and_rejects_changed_request_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);path=base/'baseline/qwen35-4b-q4/f.json';path.parent.mkdir(parents=True)
            opts={'num_ctx':8192,'num_predict':2048,'seed':42}
            row={'input_sha256':'frozen','request':{'messages':[
                {'role':'system','content':'exact system'},{'role':'user','content':'exact user'}],'options':opts},
                'response':{'prompt_eval_count':4000}}
            path.write_text(json.dumps(row));frame={'frame_id':'f','input_sha256':'frozen'}
            profile={'text_bpe_sha256':'bpe','template_sha256':'template','template':'template','special_tokens':['<unused>']}
            budget=recorded_budget(base,frame,'old','exact system','exact user',opts,profile,profile)
            self.assertFalse(budget['exact']);self.assertEqual(budget['estimated_prompt_tokens'],4128)
            self.assertEqual(budget['status'],'estimated_sufficient')
            for system,target in [('changed',profile),('exact system',{**profile,'text_bpe_sha256':'different'})]:
                fallback=recorded_budget(base,frame,'old',system,'exact user',opts,target,profile)
                self.assertEqual(fallback['method'],'conservative_utf8_bytes_plus_template_plus_512')
            row['request']['options']={**opts,'seed':43};path.write_text(json.dumps(row))
            fallback=recorded_budget(base,frame,'old','exact system','exact user',opts,profile,profile)
            self.assertEqual(fallback['method'],'conservative_utf8_bytes_plus_template_plus_512')
