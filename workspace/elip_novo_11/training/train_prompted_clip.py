import os, math, yaml, torch, random, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from elip_novo_11.models.backbone_prompted_clip import PromptedCLIP
from elip_novo_11.models.mapper import MLPMapper
from elip_novo_11.utils.datasets import build_pairs
from elip_novo_11.utils.hardmine import precompute_features, build_hard_batches, HardBatchSampler

class PairDataset(Dataset):
    def __init__(self, images, texts):
        self.images = images
        self.texts = texts
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx]

def collate_pairs(batch):
    imgs, tx = zip(*batch)
    return list(imgs), list(tx)

class InfoNCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))
    def forward(self, zi, zt):
        zi = torch.nn.functional.normalize(zi, dim=-1)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        ls = self.logit_scale.clamp(min=math.log(1/100), max=math.log(100)).exp()
        logits = ls * (zi @ zt.t())
        labels = torch.arange(zi.size(0), device=zi.device)
        li = torch.nn.functional.cross_entropy(logits, labels)
        lt = torch.nn.functional.cross_entropy(logits.t(), labels)
        return 0.5 * (li + lt)

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def build_loader(cfg, model, images, texts):
    ds = PairDataset(images, texts)
    if not cfg.get("hard_mining", True):
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True, collate_fn=collate_pairs)
        return dl
    img_bs = int(cfg.get("img_bs_hm", 64))
    txt_bs = int(cfg.get("txt_bs_hm", 256))
    subset = cfg.get("hm_subset", None)
    subset = int(subset) if subset not in [None, "null"] else None
    z_img, z_txt = precompute_features(model, images, texts, img_bs=img_bs, txt_bs=txt_bs, subset=subset)
    topn = int(cfg.get("hard_topn", cfg["batch_size"] - 1))
    batches = build_hard_batches(z_img, z_txt, int(cfg["batch_size"]), topn)
    sampler = HardBatchSampler(batches)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate_pairs)
    return dl

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg_path = os.environ.get("ELIP_CFG", "/workspace/elip_novo_11/configs/elip_clip_prompt.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
    model = PromptedCLIP(cfg["model_name"], cfg["img_size"]).to(device)
    mapper = MLPMapper(in_dim=model.d_proj, hidden=model.d_proj*2, out_dim=model.d_model, num_tokens=cfg["prompt_tokens"]).to(device)
    crit = InfoNCELoss().to(device)
    print("Mapper inicializado. num_tokens:", cfg["prompt_tokens"])
    params = list(mapper.parameters()) + list(crit.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    print("Otimizador pronto. lr:", cfg["lr"], "weight_decay:", cfg["weight_decay"])
    train_images, train_texts = build_pairs(cfg["train_dataset_name"], cfg["train_dataset_split"], cfg["image_column"], cfg["caption_column"], cfg["caption_is_list"], max_examples=cfg.get("max_train_examples"))
    print("Treino. imagens:", len(train_images), "textos:", len(train_texts))
    dl = build_loader(cfg, model, train_images, train_texts)
    scaler = torch.amp.GradScaler('cuda') if (cfg.get("fp16", True) and torch.cuda.is_available()) else None
    accum = max(int(cfg.get("grad_accum_steps", 1)), 1)
    rebuild_every = max(int(cfg.get("hard_rebuild_every_epochs", 1)), 1)
    clip_norm = float(cfg.get("grad_clip_norm", 0.0))
    print("AMP:", bool(scaler is not None), "accum:", accum, "rebuild_every_epochs:", rebuild_every)
    steps_per_epoch = len(dl)
    print("Steps por época:", steps_per_epoch)
    for epoch in range(1, int(cfg["epochs"]) + 1):
        print("Iniciando época:", epoch)
        mapper.train()
        crit.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        pbar = tqdm(enumerate(dl), total=steps_per_epoch, desc=f"Época {epoch}", unit="step")
        for step, batch in pbar:
            imgs, tx = batch
            with torch.no_grad():
                zt = model.text_hidden(tx)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pt = mapper(zt)
                    zi = model.encode_images_guided(imgs, pt)
                    loss = crit(zi, zt) / accum
                scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    if clip_norm > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(mapper.parameters(), clip_norm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                pt = mapper(zt)
                zi = model.encode_images_guided(imgs, pt)
                loss = crit(zi, zt) / accum
                loss.backward()
                if (step + 1) % accum == 0:
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(mapper.parameters(), clip_norm)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()) * accum
            pbar.set_postfix(loss_avg=running / (step + 1))
        ckpt = {"epoch": epoch, "mapper": mapper.state_dict(), "crit": crit.state_dict(), "cfg": cfg}
        path_ckpt = out_root / f"mapper_epoch{epoch}.pt"
        torch.save(ckpt, path_ckpt)
        print("Checkpoint salvo em:", str(path_ckpt))
        if cfg.get("hard_mining", True) and (epoch % rebuild_every == 0) and epoch < int(cfg["epochs"]):
            print("Reconstruindo lotes difíceis após a época", epoch)
            dl = build_loader(cfg, model, train_images, train_texts)
    last_path = out_root / "mapper_last.pt"
    torch.save({"mapper": mapper.state_dict(), "crit": crit.state_dict(), "cfg": cfg}, last_path)
    print("Checkpoint final salvo em:", str(last_path))

if __name__ == "__main__":
    main()
