import os
import sys
import subprocess
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import SiglipModel, SiglipProcessor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def _ensure(pkgs):
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--quiet"])

_ensure(["protobuf", "sentencepiece", "Pillow", "datasets", "transformers", "numpy", "torch"])

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = os.cpu_count() // 2 or 2
K_LIST = [1, 5, 10, 50]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_MODEL_ID = "google/siglip-base-patch16-224"
FT_MODEL_ID = os.environ.get("FT_MODEL_ID", "juliadollis/sTESTEiglip-finetuned-all-pt-br")

torch.manual_seed(SEED)

def pick_first_text(x):
    if isinstance(x, list) and len(x) > 0:
        return str(x[0])
    if isinstance(x, str):
        return x
    return ""

def normalize_text(example, col):
    val = pick_first_text(example.get(col, ""))
    example["text_caption"] = val.strip()
    return example

def normalize_fashioniq(example):
    cap = pick_first_text(example.get("caption", ""))
    example["text_caption"] = cap.strip()
    example["image"] = example.get("target", None)
    return example

def try_load(name, split):
    try:
        return load_dataset(name, split=split)
    except:
        return None

def build_test_split_fashion(seed):
    dsets = []

    dfi = try_load("Marqo/deepfashion-inshop", "data")
    if dfi:
        dfi = dfi.map(lambda x: normalize_text(x, "text"), num_proc=NUM_WORKERS)
        dfi = dfi.remove_columns([c for c in dfi.column_names if c not in ["image", "text_caption"]])
        dsets.append(dfi)

    dfmm = try_load("Marqo/deepfashion-multimodal", "data")
    if dfmm:
        dfmm = dfmm.map(lambda x: normalize_text(x, "text"), num_proc=NUM_WORKERS)
        dfmm = dfmm.remove_columns([c for c in dfmm.column_names if c not in ["image", "text_caption"]])
        dsets.append(dfmm)

    fiq = try_load("royokong/fashioniq_val", "val")
    if fiq:
        fiq = fiq.map(normalize_fashioniq, num_proc=NUM_WORKERS)
        keep = ["image", "text_caption"]
        fiq = fiq.remove_columns([c for c in fiq.column_names if c not in keep])
        dsets.append(fiq)

    if len(dsets) == 0:
        raise ValueError("Nenhum dataset de moda carregado.")

    merged = concatenate_datasets(dsets).shuffle(seed=seed)
    split = merged.train_test_split(test_size=0.1, seed=seed)
    return split["test"]

class FashionDataset(Dataset):
    def __init__(self, ds, processor, max_length=64):
        self.ds = ds
        self.processor = processor
        self.max_length = max_length
        tok = self.processor.tokenizer
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token or tok.unk_token or "[PAD]"
        if getattr(self.processor, "current_processor", None):
            pass
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        try:
            img = ex["image"]
            if isinstance(img, dict):
                img = Image.open(img.get("path", "")).convert("RGB")
            else:
                img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        txt = ex["text_caption"]
        enc = self.processor(
            text=txt,
            images=img,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        if "attention_mask" not in enc:
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                enc["attention_mask"] = (enc["input_ids"] != pad_id).long()
            else:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

def compute_embeddings_parallel(model, processor, ds):
    dataset = FashionDataset(ds, processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    all_img, all_txt = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            img = model.get_image_features(pixel_values=batch["pixel_values"])
            txt = model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            all_img.append(F.normalize(img, dim=-1).cpu())
            all_txt.append(F.normalize(txt, dim=-1).cpu())
    return torch.cat(all_img).numpy(), torch.cat(all_txt).numpy()

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
    print("\n=== COMPARAÇÃO DE MÉTRICAS (FASHION DATASETS) ===")
    print(f"{'Métrica':<15} {'Base':>12} {'Fine-tunado':>15}")
    print("-"*45)
    for k in ["t2i@1","t2i@5","t2i@10","t2i@50","t2i_mrr","t2i_ndcg10"]:
        print(f"{k:<15} {base_metrics[k]:>12.4f} {ft_metrics[k]:>15.4f}")
    print("-"*45)
    for k in ["i2t@1","i2t@5","i2t@10","i2t@50","i2t_mrr","i2t_ndcg10"]:
        print(f"{k:<15} {base_metrics[k]:>12.4f} {ft_metrics[k]:>15.4f}")

def compare_models(base_id, ft_id):
    test_ds = build_test_split_fashion(SEED)
    print(f"Test set size: {len(test_ds)}\n")
    base_model = SiglipModel.from_pretrained(base_id).to(DEVICE)
    base_proc = SiglipProcessor.from_pretrained(base_id)
    ft_model = SiglipModel.from_pretrained(ft_id).to(DEVICE)
    ft_proc = SiglipProcessor.from_pretrained(ft_id)
    base_img, base_txt = compute_embeddings_parallel(base_model, base_proc, test_ds)
    ft_img, ft_txt = compute_embeddings_parallel(ft_model, ft_proc, test_ds)
    base_metrics = eval_retrieval(base_img, base_txt)
    ft_metrics = eval_retrieval(ft_img, ft_txt)
    print_side_by_side(base_metrics, ft_metrics)

if __name__ == "__main__":
    compare_models(BASE_MODEL_ID, FT_MODEL_ID)
