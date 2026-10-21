FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# Only inference dependencies; do not install VILA's torch 2.3/training stack.
RUN pip install --no-cache-dir transformers==4.46.0 accelerate==0.34.2 \
    timm==0.9.12 einops==0.8.1 pillow==11.1.0 numpy==1.26.4 \
    hydra-core==1.3.2 loguru==0.7.3 opencv-python-headless==4.8.0.76 \
    sentencepiece==0.2.0 protobuf==3.20.3
COPY scripts/patch_nvila_inference.py /opt/patch_nvila_inference.py
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
