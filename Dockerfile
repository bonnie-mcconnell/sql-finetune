# Containerizes the FastAPI serving layer (src/serve.py).
#
# Expects a merged model at checkpoints/merged/ (created using
# model.merge_and_unload().save_pretrained("checkpoints/merged") 
# to be mounted or copied in at runtime, it's not baked into the
# image, since model weights are large binary artifacts and should'nt
# be in version controlled image layers

FROM python:3.11-slim

WORKDIR /app

COPY requirements-core.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
