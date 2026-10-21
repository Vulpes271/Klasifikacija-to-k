FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# The CUDA 12.8 wheels include Blackwell/sm_120; never replace them with old pins.
RUN pip install --no-cache-dir transformers==4.46.0 accelerate==0.34.2 \
    timm==0.9.12 einops==0.8.1 pillow==11.1.0 numpy==1.26.4
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
