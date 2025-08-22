from huggingface_hub import hf_hub_download
import os
import shutil

files = [
    "Task_1/AB-TH/CV_0.pt",
    "Task_1/AB-TH/CV_1.pt",
    "Task_1/AB-TH/CV_2.pt",
    "Task_1/AB-TH/CV_3.pt",
    "Task_1/AB-TH/CV_4.pt",
    "Task_1/HN/CV_0.pt",
    "Task_1/HN/CV_1.pt",
    "Task_1/HN/CV_2.pt",
    "Task_1/HN/CV_3.pt",
    "Task_1/HN/CV_4.pt",
]

for file in files:
    cached_path = hf_hub_download(repo_id="vboussot/Synthrad2025", filename=file, repo_type="model")
    local_path = os.path.join(".", file)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy(cached_path, local_path)
