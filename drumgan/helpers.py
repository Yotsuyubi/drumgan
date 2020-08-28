import torch
import os
import requests


def to_tensor(x):
    """Make x to Tensor."""
    try:
        return x.clone().detach().float()
    except:
        return torch.tensor(x).float()

def download_model():
    filepath = os.path.join(os.path.dirname(__file__), 'model/model.pt')
    if os.path.exists(filepath):
        return None
    res = requests.get(
        'https://github.com/Yotsuyubi/drumgan/blob/master/drumgan/model/model.pt?raw=true'
    )
    with open(filepath, "wb") as fp:
        fp.write(res.content)
    return None
