import os, yaml, torch, faiss
from pathlib import Path
from tqdm.auto import tqdm
from elip_novo_11.models.backbone_prompted_clip import PromptedCLIP
from elip_novo_11.models.mapper import MLPMapper
from elip_novo_11.utils.datasets import build_pairs
from elip_novo_11.utils.metrics import recall_at_k, mrr, ndcg_at_10

def build_index(x):
    x = torch.nn.functional.normalize(x, dim=-1).cpu().float().numpy()
    d = x.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x)
    return index

def topk_index(index, q, k):
    q = torch.nn.functional.normalize(q, dim=-1).cpu().float().numpy()
    D, I = index.search(q, k)
    return I

def ranks_from_topk(I):
    n = I.shape[0]
    ranks = []
    for i in range(n):
        pos = 10**9
        for j in range(I.shape[1]):
            if I[i, j] == i:
                pos = j + 1
                break
        ranks.append(pos)
    return ranks

def main():
    cfg_path = os.environ.get("ELIP_CFG", "/workspace/elip_novo_11/configs/elip_clip_prompt.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    out_root = Path(cfg["output_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inferência no dispositivo:", device)
    model = PromptedCLIP(cfg["model_name"], cfg["img_size"]).to(device)
    ckpts = sorted(out_root.glob("mapper_epoch*.pt"))
    if len(ckpts) == 0:
        ckpts = [out_root / "mapper_last.pt"]
    print("Carregando checkpoint:", str(ckpts[-1]))
    state = torch.load(str(ckpts[-1]), map_location="cpu", weights_only=True)
    mapper = MLPMapper(in_dim=model.d_proj, hidden=model.d_proj*2, out_dim=model.d_model, num_tokens=cfg["prompt_tokens"]).to(device)
    mapper.load_state_dict(state["mapper"], strict=True)
    mapper.eval()
    images, texts = build_pairs(cfg["val_dataset_name"], cfg["val_dataset_split"], cfg["image_column"], cfg["caption_column"], cfg["caption_is_list"], max_examples=cfg.get("max_val_examples"))
    print("Val. imagens:", len(images), "textos:", len(texts))
    with torch.no_grad():
        z_img_base = []
        bs = 64
        for i in tqdm(range(0, len(images), bs), desc="Embeddings base (imagens)", unit="blk"):
            zi = model.encode_images_base(images[i:i+bs])
            z_img_base.append(zi.cpu())
            torch.cuda.empty_cache()
        z_img_base = torch.cat(z_img_base, dim=0)
        z_txt_blocks = []
        txt_bs = int(cfg.get("txt_bs_hm", 256))
        for i in tqdm(range(0, len(texts), txt_bs), desc="Embeddings base (texto)", unit="blk"):
            zt = model.text_hidden(texts[i:i+txt_bs]).cpu()
            z_txt_blocks.append(zt)
            torch.cuda.empty_cache()
        z_txt = torch.cat(z_txt_blocks, dim=0)
    print("Shapes base. z_img_base:", z_img_base.shape, "z_txt:", z_txt.shape)
    index = build_index(z_img_base)
    print("Índice FAISS criado. Dim:", z_img_base.shape[1], "Base:", z_img_base.shape[0])
    I_base = topk_index(index, z_txt, int(cfg["k_base"]))
    print("Top-k base recuperado. k_base:", int(cfg["k_base"]))
    ranks_base = ranks_from_topk(I_base)
    r1_b = recall_at_k(ranks_base, 1)
    r5_b = recall_at_k(ranks_base, 5)
    r10_b = recall_at_k(ranks_base, 10)
    r50_b = recall_at_k(ranks_base, 50)
    mrr_b = mrr(ranks_base)
    ndcg_b = ndcg_at_10(ranks_base)
    print("Baseline R@1:", f"{r1_b:.4f}")
    print("Baseline R@5:", f"{r5_b:.4f}")
    print("Baseline R@10:", f"{r10_b:.4f}")
    print("Baseline R@50:", f"{r50_b:.4f}")
    print("Baseline MRR:", f"{mrr_b:.4f}")
    print("Baseline nDCG@10:", f"{ndcg_b:.4f}")
    guided_scores = torch.empty(I_base.shape[0], I_base.shape[1], device=device)
    print("Re-ranking por consulta:", I_base.shape[0], "consultas;", "candidatos por consulta:", I_base.shape[1])
    with torch.no_grad():
        for i in tqdm(range(I_base.shape[0]), desc="Re-ranking", unit="qry"):
            pt = mapper(z_txt[i:i+1].to(device))
            ids = I_base[i].tolist()
            imgs = [images[j] for j in ids]
            zi = model.encode_images_guided(imgs, pt.expand(len(imgs), -1, -1))
            sim = (zi @ z_txt[i:i+1].to(device).t()).squeeze(1)
            guided_scores[i] = sim
            torch.cuda.empty_cache()
    new_order = torch.argsort(guided_scores, dim=1, descending=True).cpu()
    I_tensor = torch.from_numpy(I_base)
    reranked = torch.gather(I_tensor, 1, new_order)
    ranks_re = []
    for i in range(reranked.shape[0]):
        pos = 10**9
        for j in range(reranked.shape[1]):
            if reranked[i, j].item() == i:
                pos = j + 1
                break
        ranks_re.append(pos)
    r1_r = recall_at_k(ranks_re, 1)
    r5_r = recall_at_k(ranks_re, 5)
    r10_r = recall_at_k(ranks_re, 10)
    r50_r = recall_at_k(ranks_re, 50)
    mrr_r = mrr(ranks_re)
    ndcg_r = ndcg_at_10(ranks_re)
    print("Rerank R@1:", f"{r1_r:.4f}")
    print("Rerank R@5:", f"{r5_r:.4f}")
    print("Rerank R@10:", f"{r10_r:.4f}")
    print("Rerank R@50:", f"{r50_r:.4f}")
    print("Rerank MRR:", f"{mrr_r:.4f}")
    print("Rerank nDCG@10:", f"{ndcg_r:.4f}")

if __name__ == "__main__":
    main()
