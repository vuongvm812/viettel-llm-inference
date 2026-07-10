FROM python:3.11-slim
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" httpx fastembed numpy
# pre-bake the embedding model into the image so there's no runtime download
# (works on an offline judge host). Keep this name in sync with EMBED_MODEL.
ENV EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
COPY app.py /app/app.py
WORKDIR /app
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
