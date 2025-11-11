import os
import math
import torch
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

VL_MODEL = "google/siglip-base-patch16-384"
TEXT_ENCODER = "Salesforce/SFR-Embedding-Mistral"

K_RETR = 20
ALPHA = 0.5
BETA = 0.3
TAU_T2T = 0.07
TAU_I2T = 0.5

TXT_BATCH = 512
IMG_BATCH = 128
MAX_LEN = 64
EVAL_N = None
CORPUS_N = None

DATASETS = [
    {"name": "laicsiifes/flickr8k-pt-br", "split": "test", "col_img": "image", "col_txt": "caption"},
    {"name": "laicsiifes/flickr30k-pt-br", "split": "train", "col_img": "image", "col_txt": "caption"},
    {"name": "laicsiifes/nocaps-pt-br", "split": "dev", "col_img": "image", "col_txt": "annotations_captions"},
    {"name": "laicsiifes/coco-captions-pt-br", "split": "test", "col_img": "image", "col_txt": "caption"},
]

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def canon_list(x):
    return x if isinstance(x, list) else [str(x)]

def build_index_ip(x):
    idx = faiss.IndexFlatIP(x.shape[1])
    try:
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
    except Exception:
        pass
    idx.add(x.astype("float32"))
    return idx

def embed_text_vl_batch(model, processor, texts, batch, max_len):
    outs = []
    for b in chunks(texts, batch):
        inputs = processor(text=b, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device=="cuda"), dtype=torch.bfloat16):
            feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1).cpu().numpy().astype("float32")
        outs.append(feats)
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(outs) if outs else np.zeros((0, 768), dtype="float32")

def embed_image_vl_batch(model, processor, imgs, batch):
    outs = []
    for b in chunks(imgs, batch):
        inputs = processor(images=b, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device=="cuda"), dtype=torch.bfloat16):
            feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1).cpu().numpy().astype("float32")
        outs.append(feats)
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(outs) if outs else np.zeros((0, 768), dtype="float32")

def recall_at_k_ip_queries(img_embs, txt_embs, k):
    index = build_index_ip(txt_embs)
    _, I = index.search(img_embs.astype("float32"), k)
    corr = 0
    for i in range(I.shape[0]):
        if i < txt_embs.shape[0] and i in I[i, :]:
            corr += 1
    return corr / I.shape[0]

def softmax_axis(scores, tau):
    x = scores.astype("float32") / tau
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x, dtype=np.float64)
    s = e.sum(axis=1, keepdims=True)
    return (e / s).astype("float32")

print("Carregando modelos...")
vl_model = AutoModel.from_pretrained(VL_MODEL).to(device)
processor = AutoProcessor.from_pretrained(VL_MODEL)
llm_encoder = SentenceTransformer(TEXT_ENCODER, device="cpu")

for d in DATASETS:
    print(f"\n===== {d['name']} ({d['split']}) =====")
    ds = load_dataset(d["name"], split=d["split"])
    texts_raw = ds[d["col_txt"]]
    imgs_raw = ds[d["col_img"]]

    if isinstance(texts_raw[0], list):
        texts = [t[0] if len(t) > 0 else "" for t in texts_raw]
    else:
        texts = texts_raw
    imgs = imgs_raw

    n_eval = len(texts) if EVAL_N is None else min(EVAL_N, len(texts), len(imgs))
    texts_eval = texts[:n_eval]
    imgs_eval = imgs[:n_eval]

    print(">> Calculando baseline embeddings...")
    txt_emb_eval = embed_text_vl_batch(vl_model, processor, texts_eval, TXT_BATCH, MAX_LEN)
    img_emb_eval = embed_image_vl_batch(vl_model, processor, imgs_eval, IMG_BATCH)

    r1_b = recall_at_k_ip_queries(img_emb_eval, txt_emb_eval, 1)
    r5_b = recall_at_k_ip_queries(img_emb_eval, txt_emb_eval, 5)
    r10_b = recall_at_k_ip_queries(img_emb_eval, txt_emb_eval, 10)

    print(">> Pré-computando corpus para CORE...")
    n_corpus = len(texts) if CORPUS_N is None else min(CORPUS_N, len(texts))
    corpus_texts = texts[:n_corpus]

    llm_corpus = llm_encoder.encode(corpus_texts, batch_size=TXT_BATCH, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    vl_corpus = embed_text_vl_batch(vl_model, processor, corpus_texts, TXT_BATCH, MAX_LEN)

    index_t2t = build_index_ip(llm_corpus)
    index_i2t = build_index_ip(vl_corpus)

    print(">> CORE texto em lote...")
    q_llm_eval = llm_encoder.encode(texts_eval, batch_size=TXT_BATCH, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores_t2t, idx_t2t = index_t2t.search(q_llm_eval, K_RETR)
    probs_t2t = softmax_axis(scores_t2t, TAU_T2T)
    retr_t2t = vl_corpus[idx_t2t]
    wt = (retr_t2t * probs_t2t[:, :, None]).sum(axis=1)
    W_plus = ALPHA * wt + (1 - ALPHA) * txt_emb_eval
    W_plus = W_plus / np.linalg.norm(W_plus, axis=1, keepdims=True)

    print(">> CORE imagem em lote...")
    scores_i2t, idx_i2t = index_i2t.search(img_emb_eval.astype("float32"), K_RETR)
    probs_i2t = softmax_axis(scores_i2t, TAU_I2T)
    retr_i2t = vl_corpus[idx_i2t]
    zt = (retr_i2t * probs_i2t[:, :, None]).sum(axis=1)
    Z_plus = BETA * zt + (1 - BETA) * img_emb_eval
    Z_plus = Z_plus / np.linalg.norm(Z_plus, axis=1, keepdims=True)

    r1_c = recall_at_k_ip_queries(Z_plus, W_plus, 1)
    r5_c = recall_at_k_ip_queries(Z_plus, W_plus, 5)
    r10_c = recall_at_k_ip_queries(Z_plus, W_plus, 10)

    print(f"Baseline Recall@1/5/10: {r1_b:.4f} | {r5_b:.4f} | {r10_b:.4f}")
    print(f"CORE     Recall@1/5/10: {r1_c:.4f} | {r5_c:.4f} | {r10_c:.4f}")
    print("=====================================")
