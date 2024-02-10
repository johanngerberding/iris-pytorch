from tqdm import tqdm 
import torch 
import os 
import hashlib
import requests

MODEL_NAME = "vgg16.pth"
MODEL_URL = "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
MODEL_MD5 = "d507d7349b931f0638a25a48a722f98a"

def md5_hash(path: str) -> str: 
    with open(path, "rb") as fp:
        content = fp.read()
    return hashlib.md5(content).hexdigest()

def download(model_path: str):
    os.makedirs(os.path.split(model_path)[0], exist_ok=True) 
    with requests.get(MODEL_URL, stream=True) as r: 
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar: 
            with open(model_path, "wb") as fp: 
                for data in r.iter_content(chunk_size=1024):
                    if data:
                        fp.write(data)
                        pbar.update(1024)


def get_ckpt_path(root: str, check: bool = False) -> str:
    model_path = os.path.join(root, MODEL_NAME)
    if not os.path.exists(model_path) or (check and not md5_hash(model_path) == MODEL_MD5):
        print(f"Downloading VGG16 model to {model_path}")
        download(model_path)
        md5 = md5_hash(model_path)
        assert md5 == MODEL_MD5
    return model_path


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor: 
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor: 
    return x.mean([2, 3], keepdim=keepdim)
