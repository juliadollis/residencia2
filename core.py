import torch
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
VL_MODEL = "google/siglip-base-patch16-384"
TEXT_ENCODER = "Salesforce/SFR-Embedding-Mistral"
K_RETR = 10
ALPHA = 0.2
BETA = 0.5
TAU_T2T = 1.0
TAU_I2T = 100.0
SUBSET_TEST = 200
SUBSET_TRAIN_CORPUS = 5000

vl_model = AutoModel.from_pretrained(VL_MODEL).to(device)
processor = AutoProcessor.from_pretrained(VL_MODEL)
llm_encoder = SentenceTransformer(TEXT_ENCODER, device=device)

ds = load_dataset("laicsiifes/flickr30k-pt-br")
test_ds = ds["test"]
train_ds = ds["train"]

def canon_list(x):
    return x if isinstance(x, list) else [str(x)]

images = test_ds["image"][:SUBSET_TEST]
test_caps_list = [canon_list(c) for c in test_ds["caption"][:SUBSET_TEST]]
test_caps_flat = [c for caps in test_caps_list for c in caps]
test_caps_offsets = []
acc = 0
for caps in test_caps_list:
    idxs = list(range(acc, acc + len(caps)))
    test_caps_offsets.append(idxs)
    acc += len(caps)

train_caps_corpus = train_ds["caption"][:SUBSET_TRAIN_CORPUS]
train_caps_corpus = [c for caps in train_caps_corpus for c in canon_list(caps)]

def embed_text_vl(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = vl_model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.cpu().numpy().astype("float32")

def embed_image_vl(imgs):
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = vl_model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.cpu().numpy().astype("float32")

def build_index(emb):
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    return index

def recall_at_k_multi(I, gt_lists, k):
    corr = 0
    for i, topk in enumerate(I[:, :k]):
        if any(g in topk for g in gt_lists[i]):
            corr += 1
    return corr / len(gt_lists)

def mrr_multi(I, gt_lists):
    rr = 0.0
    for i, ranks in enumerate(I):
        best = None
        for g in gt_lists[i]:
            pos = np.where(ranks == g)[0]
            if len(pos) > 0:
                p = int(pos[0]) + 1
                best = p if best is None or p < best else best
        rr += 0.0 if best is None else 1.0 / best
    return rr / len(gt_lists)

def ndcg_at_k_multi(I, gt_lists, k):
    dcg = 0.0
    idcg = 0.0
    for i, top in enumerate(I[:, :k]):
        rel = np.zeros(k, dtype=np.float32)
        gset = set(gt_lists[i])
        for j, idx in enumerate(top):
            rel[j] = 1.0 if idx in gset else 0.0
        gains = rel / np.log2(np.arange(2, k + 2))
        dcg += gains.sum()
        m = min(len(gset), k)
        ideal = np.ones(m, dtype=np.float32) / np.log2(np.arange(2, m + 2))
        idcg += ideal.sum()
    return 0.0 if idcg == 0 else dcg / idcg

txt_emb_test = embed_text_vl(test_caps_flat)
img_emb_test = np.vstack([embed_image_vl(img) for img in images])
index_base = build_index(txt_emb_test)
_, I_base = index_base.search(img_emb_test, 10)

r1_base = recall_at_k_multi(I_base, test_caps_offsets, 1)
r5_base = recall_at_k_multi(I_base, test_caps_offsets, 5)
r10_base = recall_at_k_multi(I_base, test_caps_offsets, 10)
mrr_base = mrr_multi(I_base, test_caps_offsets)
ndcg10_base = ndcg_at_k_multi(I_base, test_caps_offsets, 10)

llm_text_emb_train = llm_encoder.encode(train_caps_corpus, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
index_t2t = build_index(llm_text_emb_train)
vl_text_emb_train = embed_text_vl(train_caps_corpus)
index_i2t = build_index(vl_text_emb_train)

def softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def retrieve_t2t_llm(q2d, k, tau):
    scores, idx = index_t2t.search(q2d.astype("float32"), k)
    p = softmax_np(scores[0] / tau)
    items = [train_caps_corpus[i] for i in idx[0]]
    return items, p

def retrieve_i2t_vl(q2d, k, tau):
    scores, idx = index_i2t.search(q2d.astype("float32"), k)
    p = softmax_np(scores[0] / tau)
    items = [train_caps_corpus[i] for i in idx[0]]
    return items, p

def enrich_caption_embed(text):
    base = embed_text_vl([text])[0]
    q = llm_encoder.encode([text], normalize_embeddings=True).astype("float32")
    items, probs = retrieve_t2t_llm(q, K_RETR, TAU_T2T)
    retr = embed_text_vl(items)
    wt = (retr * probs.reshape(-1, 1)).sum(axis=0)
    w_plus = ALPHA * wt + (1 - ALPHA) * base
    w_plus = w_plus / np.linalg.norm(w_plus)
    return w_plus.astype("float32")

def enrich_image_embed(img):
    zq = embed_image_vl(img)[0]
    items, probs = retrieve_i2t_vl(zq.reshape(1, -1), K_RETR, TAU_I2T)
    retr = embed_text_vl(items)
    zt = (retr * probs.reshape(-1, 1)).sum(axis=0)
    z_plus = BETA * zt + (1 - BETA) * zq
    z_plus = z_plus / np.linalg.norm(z_plus)
    return z_plus.astype("float32")

W_plus = np.zeros_like(txt_emb_test)
for i in tqdm(range(len(test_caps_flat)), desc="W+"):
    W_plus[i] = enrich_caption_embed(test_caps_flat[i])

Z_plus = np.zeros_like(img_emb_test)
for i in tqdm(range(len(images)), desc="Z+"):
    Z_plus[i] = enrich_image_embed(images[i])

index_core = build_index(W_plus)
_, I_core = index_core.search(Z_plus, 10)

r1_core = recall_at_k_multi(I_core, test_caps_offsets, 1)
r5_core = recall_at_k_multi(I_core, test_caps_offsets, 5)
r10_core = recall_at_k_multi(I_core, test_caps_offsets, 10)
mrr_core = mrr_multi(I_core, test_caps_offsets)
ndcg10_core = ndcg_at_k_multi(I_core, test_caps_offsets, 10)

print("==== MÉTRICAS BASELINE ====")
print(f"Recall@1:  {r1_base:.4f}")
print(f"Recall@5:  {r5_base:.4f}")
print(f"Recall@10: {r10_base:.4f}")
print(f"MRR:       {mrr_base:.4f}")
print(f"nDCG@10:   {ndcg10_base:.4f}")

print("==== MÉTRICAS CORE ====")
print(f"Recall@1:  {r1_core:.4f}")
print(f"Recall@5:  {r5_core:.4f}")
print(f"Recall@10: {r10_core:.4f}")
print(f"MRR:       {mrr_core:.4f}")
print(f"nDCG@10:   {ndcg10_core:.4f}")
