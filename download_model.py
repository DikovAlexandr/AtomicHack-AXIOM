import os
from huggingface_hub import snapshot_download

model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
cache_dir = os.path.join(os.getcwd(), "hf_cache") # Сохранить в ./hf_cache

print(f"Downloading model {model_name} to {cache_dir}...")
snapshot_download(repo_id=model_name, cache_dir=cache_dir)
print("Model downloaded successfully!")