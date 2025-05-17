import os
from huggingface_hub import HfApi, HfFolder

os.makedirs("./saved", exist_ok=True)

api_key = os.getenv("API_KEY")
HfFolder.save_token(api_key or "")


def save_to_space(path):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=f"./saved/{path}",
        path_in_repo=f"./saved/{path}",
        repo_id="Olegert/olgertlm",
        repo_type="space",
    )
