from types import SimpleNamespace
import unittest
from unittest.mock import patch

from maintenance.guard_reverse_continuation import GPU_LOAD_FAILURE,lost_container_gpu,metadata_order_equivalent


class LostGpuRecoveryTests(unittest.TestCase):
    def test_only_confirmed_nvml_loss_qualifies(self):
        with patch('maintenance.guard_reverse_continuation.subprocess.run',return_value=SimpleNamespace(
            returncode=255,stdout='Failed to initialize NVML: Unknown Error\n',stderr='')) as command:
            self.assertIn('NVML',lost_container_gpu(GPU_LOAD_FAILURE,'test-ollama'))
            self.assertEqual(command.call_args.args[0],['docker','exec','test-ollama','nvidia-smi'])

    def test_healthy_device_does_not_bypass_model_context_verification(self):
        with patch('maintenance.guard_reverse_continuation.subprocess.run',return_value=SimpleNamespace(
            returncode=0,stdout='GPU healthy',stderr='')):
            self.assertIsNone(lost_container_gpu(GPU_LOAD_FAILURE,'test-ollama'))

    def test_other_model_failures_do_not_restart_service(self):
        with patch('maintenance.guard_reverse_continuation.subprocess.run') as command:
            self.assertIsNone(lost_container_gpu('Model digest changed','test-ollama'))
            command.assert_not_called()

    def test_other_device_errors_do_not_trigger_recovery(self):
        with patch('maintenance.guard_reverse_continuation.subprocess.run',return_value=SimpleNamespace(
            returncode=1,stdout='',stderr='Permission denied')):
            self.assertIsNone(lost_container_gpu(GPU_LOAD_FAILURE,'test-ollama'))

    def test_equivalent_parameter_order_preserves_other_metadata(self):
        a={'digest':'same','show':{'parameters':'temperature 1\ntop_k 20','modelfile':'FROM same\nPARAMETER temperature 1\nPARAMETER top_k 20','template':'fixed'}}
        b={'digest':'same','show':{'parameters':'top_k 20\ntemperature 1','modelfile':'FROM same\nPARAMETER top_k 20\nPARAMETER temperature 1','template':'fixed'}}
        self.assertTrue(metadata_order_equivalent(a,b))
        b['show']['template']='changed';self.assertFalse(metadata_order_equivalent(a,b))

    def test_parameter_value_or_weight_change_is_not_normalized(self):
        a={'digest':'same','show':{'parameters':'top_k 20'}}
        self.assertFalse(metadata_order_equivalent(a,{'digest':'same','show':{'parameters':'top_k 21'}}))
        self.assertFalse(metadata_order_equivalent(a,{'digest':'different','show':{'parameters':'top_k 20'}}))


if __name__=='__main__':unittest.main()
