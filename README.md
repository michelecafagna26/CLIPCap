# ClipCap

Implementation of **"ClipCap: CLIP Prefix for Image Captioning"** ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734).

This repo contains the code to run ClipCap finetuned on the [HL and HL-Narratives Dataset](https://github.com/michelecafagna26/HL-dataset) available on 🤗:
- [hl](https://huggingface.co/datasets/michelecafagna26/hl)
- [hl-narratives](https://huggingface.co/datasets/michelecafagna26/hl-narratives)

We provide ClipCap **fine-tuned models** (all available on 🤗) for :
- Scene generation **[clipcap-base-captioning-ft-hl-scenes](https://huggingface.co/michelecafagna26/clipcap-base-captioning-ft-hl-scenes)**
- Action generation **[clipcap-base-captioning-ft-hl-actions](https://huggingface.co/michelecafagna26/clipcap-base-captioning-ft-hl-actions)**
- Rationale  generation **[clipcap-base-captioning-ft-hl-rationales](https://huggingface.co/michelecafagna26/clipcap-base-captioning-ft-hl-rationales)**
- Narrative generation **[clipcap-base-captioning-ft-hl-narratives](https://huggingface.co/michelecafagna26/clipcap-base-captioning-ft-hl-narratives)**

This repo is an adaptation of the [original repo](https://github.com/rmokady/CLIP_prefix_caption/tree/main) that **makes it easy to run pre-trained and finetuned checkpoints, in inference**. If you want to train a new model we suggest using the original repo.

## Models

We finetune both the LM and the Mapping network, for further details see the model cards.

## Installation
```bash
pip install git+https://github.com/michelecafagna26/CLIPCap.git
```

## Example: Narrative Captioning generation 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xcaJOxaAp8TRd8a6x1XnAptVjHQRv3Zj?usp=sharing)

Download the model
```bash
git lfs install # if not installed
git clone https://huggingface.co/michelecafagna26/clipcap-base-captioning-ft-hl-narratives
```
Run:
```python3
from clipcap import ClipCaptionModel
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import torch
import clip
import requests
from PIL import Image

model_path = "clipcap-base-captioning-ft-hl-narratives/pytorch_model.pt" # change accordingly

# load clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prefix_length = 10

# load ClipCap
model = ClipCaptionModel(prefix_length, tokenizer=tokenizer)
model.from_pretrained(model_path)
model = model.eval()
model = model.to(device)

# load the image
img_url = 'https://datasets-server.huggingface.co/assets/michelecafagna26/hl-narratives/--/default/train/3/image/image.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')


# extract the prefix
image = preprocess(raw_image).unsqueeze(0).to(device)
with torch.no_grad():
    prefix = clip_model.encode_image(image).to(
        device, dtype=torch.float32
    )
    prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

# generate the caption   
model.generate_beam(embed=prefix_embed)[0]


# >> "He is riding a skateboard in a skate park, he wants to skate."
```

## Citations

If you use this code, please consider citing:
```BibTeX
@inproceedings{Cafagna2023HLDG,
  title={HL Dataset: Grounding High-Level Linguistic Concepts in Vision},
  author={Michele Cafagna and Kees van Deemter and Albert Gatt},
  year={2023}
}
```
and the original ClipCap work:
```BibTeX
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```
