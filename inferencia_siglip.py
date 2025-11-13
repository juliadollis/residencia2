import os
import random
import numpy as np
import torch
import faiss
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
from PIL import Image, ImageFile
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SEED = 42
TEST_SIZE = 0.1
MAX_LENGTH = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo usado:", device)

#turing552/siglip-wikiart-5ep
model_id = "turing552/siglip-wikiart-5ep"
#"google/siglip-so400m-patch14-384"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()


#laicsiifes/flickr30k-pt-br
#laicsiifes/flickr8k-pt-br
#laicsiifes/coco-captions-pt-br
#lmms-lab/flickr30k
#Artificio/WikiArt
#Marqo/deepfashion-multimodal
#tungdop2/pokemon
#gautamgc17/pokemon_cards_description_vlm
#eltorio/ROCOv2-radiology


DATASET_NAME = "laicsiifes/coco-captions-pt-br"
DATASET_SPLIT = "train"
TEXT_COLUMN_ORIGINAL = "caption"
NUM_WORKERS = max(2, (os.cpu_count() or 2) // 2)

def processar_como_esta(example):
    texto_original = str(example.get(TEXT_COLUMN_ORIGINAL, "")).strip()
    example["text_caption"] = texto_original
    return example

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset = dataset.map(
    processar_como_esta,
    num_proc=NUM_WORKERS,
    remove_columns=[c for c in dataset.column_names if c not in ["image", "text_caption"]]
)

split_full = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
trainval_dataset = split_full["train"]
test_dataset = split_full["test"]

ds = test_dataset

print("Tamanho total do dataset:", len(dataset))
print("Tamanho train+val (não usado aqui):", len(trainval_dataset))
print("Tamanho test usado nesta avaliação:", len(ds))

@torch.no_grad()
def encode_images(pil_images, bs=64):
    vecs = []
    for i in tqdm(range(0, len(pil_images), bs), desc="Imagens"):
        chunk = pil_images[i:i+bs]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        vecs.append(feats.cpu())
    return torch.cat(vecs, dim=0).numpy()

imgs = [ex["image"].convert("RGB") if isinstance(ex["image"], Image.Image) else ex["image"] for ex in ds]
img_vecs = encode_images(imgs, bs=64)

d = img_vecs.shape[1]
index = faiss.IndexFlatIP(d)
index.add(img_vecs)

pairs = []
for i, ex in enumerate(ds):
    cap = ex["text_caption"]
    if cap is None:
        continue
    pairs.append((cap, i))

@torch.no_grad()
def encode_text(texts, bs=128):
    vecs = []
    for i in tqdm(range(0, len(texts), bs), desc="Textos"):
        chunk = texts[i:i+bs]
        inputs = processor(
            text=chunk,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        vecs.append(feats.cpu())
    return torch.cat(vecs, dim=0).numpy()

texts = [t for t, _ in pairs]
text_vecs = encode_text(texts, bs=128)

k = 10
D, I = index.search(text_vecs, k)

def compute_metrics(I, gt_indices, k_list=[1, 5, 10]):
    recalls = {k: [] for k in k_list}
    mrrs = []
    ndcgs = []
    for qi, gt in enumerate(gt_indices):
        ranked = I[qi]
        hits = (ranked == gt).astype(int)
        for k in k_list:
            recalls[k].append(int(gt in ranked[:k]))
        where = np.where(hits == 1)[0]
        if len(where):
            mrrs.append(1.0 / (where[0] + 1))
        else:
            mrrs.append(0.0)
        gains = hits[:10]
        dcg = np.sum(gains / np.log2(np.arange(1, len(gains) + 1) + 1))
        idcg = 1.0
        ndcgs.append(dcg / idcg)
    out = {f"recall@{k}": float(np.mean(recalls[k])) for k in k_list}
    out["mrr"] = float(np.mean(mrrs))
    out["ndcg@10"] = float(np.mean(ndcgs))
    return out

gt = [i for _, i in pairs]
metrics = compute_metrics(I, gt)

print("------------------")
print("Modelo:", model_id)
print("Dataset:", DATASET_NAME)
print("Split: test com TEST_SIZE =", TEST_SIZE, "e SEED =", SEED)
print("------------------")
print(metrics)
