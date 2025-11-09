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
SUBSET = 500  # ajuste se quiser mais ou menos amostras

print("Carregando modelos...")
vl_model = AutoModel.from_pretrained(VL_MODEL).to(device)
processor = AutoProcessor.from_pretrained(VL_MODEL)
llm_encoder = SentenceTransformer(TEXT_ENCODER, device=device)

print("Carregando dataset ArtBench...")
ds = load_dataset("Doub7e/ArtBench-2-gpt4o-captions", split="train")
captions = ds["prompt"][:SUBSET]
images = ds["image"][:SUBSET]
N = len(captions)
print(f"Total de exemplos: {N}")

def embed_text_vl(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = vl_model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.cpu().numpy().astype("float32")

def embed_image_vl(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = vl_model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats.cpu().numpy().astype("float32")

def recall_at_k(img_embs, txt_embs, k):
    index = faiss.IndexFlatIP(txt_embs.shape[1])
    index.add(txt_embs)
    _, I = index.search(img_embs, k)
    correct = sum([i in I[i, :] for i in range(len(I))])
    return correct / len(I)

print("Calculando baseline...")
img_emb_base = np.vstack([embed_image_vl(img) for img in tqdm(images, desc="Imagens baseline")])
txt_emb_base = np.vstack([embed_text_vl([t]) for t in tqdm(captions, desc="Textos baseline")])

r1_base = recall_at_k(img_emb_base, txt_emb_base, 1)
r5_base = recall_at_k(img_emb_base, txt_emb_base, 5)
r10_base = recall_at_k(img_emb_base, txt_emb_base, 10)

llm_text_emb = llm_encoder.encode(captions, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
index_t2t = faiss.IndexFlatIP(llm_text_emb.shape[1])
index_t2t.add(llm_text_emb)
vl_text_emb = txt_emb_base.copy()
index_i2t = faiss.IndexFlatIP(vl_text_emb.shape[1])
index_i2t.add(vl_text_emb)

def softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def retrieve_t2t_llm(q2d, k, tau):
    scores, idx = index_t2t.search(q2d.astype("float32"), k)
    p = softmax_np(scores[0] / tau)
    items = [captions[i] for i in idx[0]]
    return items, p

def retrieve_i2t_vl(q2d, k, tau):
    scores, idx = index_i2t.search(q2d.astype("float32"), k)
    p = softmax_np(scores[0] / tau)
    items = [captions[i] for i in idx[0]]
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

print("Gerando W+ (text-to-text)...")
W_plus = np.zeros_like(txt_emb_base)
for i in tqdm(range(N)):
    W_plus[i] = enrich_caption_embed(captions[i])

print("Gerando Z+ (image-to-text)...")
Z_plus = np.zeros_like(img_emb_base)
for i in tqdm(range(N)):
    Z_plus[i] = enrich_image_embed(images[i])

r1_core = recall_at_k(Z_plus, W_plus, 1)
r5_core = recall_at_k(Z_plus, W_plus, 5)
r10_core = recall_at_k(Z_plus, W_plus, 10)

print("\n==== RESULTADOS ====")
print(f"Baseline Recall@1:  {r1_base:.4f}")
print(f"Baseline Recall@5:  {r5_base:.4f}")
print(f"Baseline Recall@10: {r10_base:.4f}")
print(f"CORE Recall@1:      {r1_core:.4f}")
print(f"CORE Recall@5:      {r5_core:.4f}")
print(f"CORE Recall@10:     {r10_core:.4f}")
print("=====================")
