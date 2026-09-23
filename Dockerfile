FROM python:3.12-slim

WORKDIR /app
# Runtime deps only; requirements.txt also pulls torch/transformers for the notebook.
RUN pip install --no-cache-dir tensorflow==2.21.0 fastapi uvicorn

COPY app.py ./
COPY saved_model ./saved_model
COPY static ./static

# 7860 is Hugging Face Spaces' default; other hosts can override PORT.
ENV PORT=7860
EXPOSE 7860
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
