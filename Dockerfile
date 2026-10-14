FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY llm_object_classifier ./llm_object_classifier
COPY scripts ./scripts
COPY configs ./configs
COPY examples ./examples
COPY tests ./tests
ENTRYPOINT ["python", "-m", "llm_object_classifier"]
CMD ["--help"]
