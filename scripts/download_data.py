"""Fetch the public corpora into data/raw/ (CC BY-NC 4.0 / CC BY 4.0)."""

from huggingface_hub import snapshot_download

from triage.data import HF_DATASETS, RAW

if __name__ == "__main__":
    for local, repo in HF_DATASETS.items():
        path = snapshot_download(repo, repo_type="dataset", local_dir=RAW / local)
        print(f"{repo} -> {path}")
