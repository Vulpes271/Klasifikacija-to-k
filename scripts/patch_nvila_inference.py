#!/usr/bin/env python3
"""Small, explicit inference-only compatibility patch for pinned NVIDIA VILA.

Keep Qwen2 + SigLIP + dynamic S2 and all weights unchanged. Avoid importing FP8,
training or other vision architectures. SDPA replaces the CUDA-12.2 FlashAttn
binary that cannot execute on RTX 5090. Save the resulting git diff separately.
"""
from pathlib import Path
import sys


def patch(root):
    replacements = {
        'llava/model/__init__.py': [
            ('from .language_model.fp8linearqwen2 import FP8LinearQwen2Config, FP8LinearQwen2Model\n', ''),
            ('from .language_model.qllava_qllama import QLlavaLlamaConfig, QLlavaLlamaModel\n', '')],
        'llava/train/sequence_parallel/globals.py': [
            ('import deepspeed.comm as dist', 'import torch.distributed as dist')],
        'llava/model/multimodal_encoder/builder.py': [
            ('from .intern_encoder import InternVisionTower, InternVisionTowerS2', ''),
            ('from .radio_encoder import RADIOVisionTower', '')],
        'llava/model/multimodal_encoder/siglip_encoder.py': [
            ('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')],
        'llava/model/multimodal_encoder/vision_encoder.py': [
            ('from s2wrapper import forward as multiscale_forward',
             '# Static S2 is unused; dynamic S2 below has its own implementation.')],
    }
    for name, changes in replacements.items():
        path = root / name
        content = path.read_text()
        for old, new in changes:
            if old not in content:
                if new and new in content:
                    continue
                if not new:
                    continue
                raise ValueError('Pinned upstream code changed: ' + name)
            content = content.replace(old, new)
        path.write_text(content)


if __name__ == '__main__':
    patch(Path(sys.argv[1]))
