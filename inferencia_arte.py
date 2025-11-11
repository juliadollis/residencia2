import os
import sys
import subprocess

def _ensure(pkgs):
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--quiet"])

_ensure(["protobuf", "sentencepiece", "Pillow", "datasets", "transformers", "numpy", "torch"])

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import SiglipModel, SiglipProcessor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SEED = 42
BATCH_SIZE = 128
K_LIST = [1, 5, 10, 50]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_ID = "google/siglip-base-patch16-224"
FT_MODEL_ID = "juliadollis/siglip-finetuned-artdatasets"

def pick_first_text(x):
    if isinstance(x, list) and len(x) > 0:
        return str(x[0])
    if isinstance(x, str):
        return x
    return ""

def normalize(example, col):
    val = pick_first_text(example.get(col, ""))
    example["text_caption"] = val.strip()
    return example

def try_load(name, split):
    try:
        return load_dataset(name, split=split)
    except:
        return None

def build_test_split_art(seed):
    datasets_list = []
    ds_artbench = try_load("Doub7e/ArtBench-2", "train")
    if ds_artbench:
        ds_artbench = ds_artbench.map(lambda x: normalize(x, "prompt"), num_proc=os.cpu_count() // 2 or 2)
        ds_artbench = ds_artbench.remove_columns([c for c in ds_artbench.column_names if c not in ["image", "text_caption"]])
        datasets_list.append(ds_artbench)
    ds_wikiart = try_load("Artificio/WikiArt", "train")
    if ds_wikiart:
        ds_wikiart = ds_wikiart.map(lambda x: normalize(x, "description"), num_proc=os.cpu_count() // 2 or 2)
        ds_wikiart = ds_wikiart.remove_columns([c for c in ds_wikiart.column_names if c not in ["image", "text_caption"]])
        datasets_list.append(ds_wikiart)
    ds_gpt4o = try_load("Doub7e/ArtBench-2-gpt4o-captions", "train")
    if ds_gpt4o:
        ds_gpt4o = ds_gpt4o.map(lambda x: normalize(x, "prompt"), num_proc=os.cpu_count() // 2 or 2)
        ds_gpt4o = ds_gpt4o.remove_columns([c for c in ds_gpt4o.column_names if c not in ["image", "text_caption"]])
        datasets_list.append(ds_gpt4o)
    if len(datasets_list) == 0:
        raise ValueError("Nenhum dataset de arte carregado.")
    merged = concatenate_datasets(datasets_list).shuffle(seed=seed)
    split = merged.train_test_split(test_size=0.1, seed=seed)
    return split["test"]

def ensure_pad_and_mask(processor, enc):
    tok = processor.tokenizer
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token or tok.unk_token or "[PAD]"
    if "attention_mask" not in enc:
        pad_id = tok.pad_token_id
        if pad_id is not None:
            enc["attention_mask"] = (enc["input_ids"] != pad_id).long()
        else:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    return enc

def compute_embeddings(model, processor, ds):
    imgs, txts = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), BATCH_SIZE):
            batch = ds[i:i+BATCH_SIZE]
            images = []
            for ex in batch["image"]:
                try:
                    img = ex.convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
                images.append(img)
            enc = processor(text=batch["text_caption"], images=images, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            enc = ensure_pad_and_mask(processor, enc)
            img = model.get_image_features(pixel_values=enc["pixel_values"])
            txt = model.get_text_features(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            imgs.append(F.normalize(img, dim=-1).cpu())
            txts.append(F.normalize(txt, dim=-1).cpu())
    return torch.cat(imgs).numpy(), torch.cat(txts).numpy()

def recall_at_k(sim, k):
    n = sim.shape[0]
    idx = np.argsort(-sim, axis=1)
    hits = (idx[:, :k] == np.arange(n)[:, None]).any(axis=1)
    return float(hits.mean())

def mrr(sim):
    n = sim.shape[0]
    idx = np.argsort(-sim, axis=1)
    ranks = np.where(idx == np.arange(n)[:, None])[1] + 1
    return float((1.0 / ranks).mean())

def ndcg_at_k(sim, k):
    idx = np.argsort(-sim, axis=1)[:, :k]
    gains = (idx == np.arange(idx.shape[0])[:, None]).astype(np.float32)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts).sum(axis=1)
    return float(dcg.mean())

def eval_retrieval(img_embs, txt_embs):
    sim_t2i = txt_embs @ img_embs.T
    sim_i2t = img_embs @ txt_embs.T
    metrics = {}
    for k in K_LIST:
        metrics[f"t2i@{k}"] = recall_at_k(sim_t2i, k)
        metrics[f"i2t@{k}"] = recall_at_k(sim_i2t, k)
    metrics["t2i_mrr"] = mrr(sim_t2i)
    metrics["i2t_mrr"] = mrr(sim_i2t)
    metrics["t2i_ndcg10"] = ndcg_at_k(sim_t2i, 10)
    metrics["i2t_ndcg10"] = ndcg_at_k(sim_i2t, 10)
    return metrics

def print_side_by_side(base_metrics, ft_metrics):
    print("\n=== COMPARAÇÃO DE MÉTRICAS (ART DATASETS) ===")
    print(f"{'Métrica':<15} {'Base':>12} {'Fine-tunado':>15}")
    print("-"*45)
    for k in ["t2i@1","t2i@5","t2i@10","t2i@50","t2i_mrr","t2i_ndcg10"]:
        print(f"{k:<15} {base_metrics[k]:>12.4f} {ft_metrics[k]:>15.4f}")
    print("-"*45)
    for k in ["i2t@1","i2t@5","i2t@10","i2t@50","i2t_mrr","i2t_ndcg10"]:
        print(f"{k:<15} {base_metrics[k]:>12.4f} {ft_metrics[k]:>15.4f}")

def compare_models(base_id, ft_id):
    torch.manual_seed(SEED)
    test_ds = build_test_split_art(SEED)
    print(f"Test set size: {len(test_ds)}\n")
    base_model = SiglipModel.from_pretrained(base_id).to(DEVICE)
    base_proc = SiglipProcessor.from_pretrained(base_id)
    base_img, base_txt = compute_embeddings(base_model, base_proc, test_ds)
    base_metrics = eval_retrieval(base_img, base_txt)
    ft_model = SiglipModel.from_pretrained(ft_id).to(DEVICE)
    ft_proc = SiglipProcessor.from_pretrained(ft_id)
    ft_img, ft_txt = compute_embeddings(ft_model, ft_proc, test_ds)
    ft_metrics = eval_retrieval(ft_img, ft_txt)
    print_side_by_side(base_metrics, ft_metrics)

if __name__ == "__main__":
    compare_models(BASE_MODEL_ID, FT_MODEL_ID)
