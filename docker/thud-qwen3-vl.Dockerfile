FROM thud-florence:local
# Reuse the working Blackwell CUDA/PyTorch image; Qwen3-VL needs a newer HF API.
RUN pip install --no-cache-dir transformers==4.57.6 accelerate==1.12.0
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
