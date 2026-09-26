"""Upload only the two Q-table model files to the established Hub account."""

from pathlib import Path
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "Notebook/runs/1m/hub"
USERNAME = "tashobi02"  # Matches the existing Unit 1 Hugging Face upload script.


def main():
    api = HfApi()
    username = api.whoami()["name"]
    if username != USERNAME:
        raise RuntimeError(f"Authenticated as {username!r}; expected {USERNAME!r}")
    for name in ("q-FrozenLake-v1-4x4-noSlippery-1m", "q-Taxi-v3-1m"):
        package = PACKAGES / name
        if not package.is_dir():
            raise FileNotFoundError(package)
        model_path = package / "q-learning.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        repo_id = f"{USERNAME}/{name}"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=model_path,
            path_in_repo="q-learning.pkl",
            commit_message=f"Upload 1M-step {name} Q-table",
        )
        print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
