import torch, faiss, random
from tqdm.auto import tqdm

class HardBatchSampler:
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        for b in self.batches:
            yield b
    def __len__(self):
        return len(self.batches)

@torch.no_grad()
def precompute_features(model, images, texts, img_bs=64, txt_bs=256, subset=None, seed=3407):
    print("Pré-computando features base para hard mining. img_bs:", img_bs, "txt_bs:", txt_bs, "subset:", subset)
    if subset is not None:
        n = min(subset, len(images), len(texts))
        idx = list(range(len(images)))
        random.Random(seed).shuffle(idx)
        idx = idx[:n]
        images = [images[i] for i in idx]
        texts = [texts[i] for i in idx]
        print("Amostrando subset para hard mining:", n)
    z_img = []
    for i in tqdm(range(0, len(images), img_bs), desc="Embeddings de imagem base", unit="blk"):
        zi = model.encode_images_base(images[i:i+img_bs])
        z_img.append(zi.cpu())
        torch.cuda.empty_cache()
    z_img = torch.cat(z_img, dim=0)
    z_txt_blocks = []
    for i in tqdm(range(0, len(texts), txt_bs), desc="Embeddings de texto base", unit="blk"):
        block = model.text_hidden(texts[i:i+txt_bs]).cpu()
        z_txt_blocks.append(block)
        torch.cuda.empty_cache()
    z_txt = torch.cat(z_txt_blocks, dim=0)
    z_img = torch.nn.functional.normalize(z_img, dim=-1)
    z_txt = torch.nn.functional.normalize(z_txt, dim=-1)
    print("Features pré-computadas. z_img:", z_img.shape, "z_txt:", z_txt.shape)
    return z_img, z_txt

def build_hard_batches(z_img, z_txt, batch_size, topn, seed=3407):
    print("Construindo lotes difíceis. batch_size:", batch_size, "topn:", topn)
    d = z_txt.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(z_txt.numpy().astype('float32'))
    n = z_img.shape[0]
    anchors = list(range(n))
    random.Random(seed).shuffle(anchors)
    batches = []
    used = set()
    for i in tqdm(anchors, desc="Minerando âncoras", unit="anc"):
        if i in used:
            continue
        q = z_img[i:i+1].numpy().astype('float32')
        D, I = index.search(q, topn + 1)
        cand = [int(x) for x in I[0].tolist() if int(x) != i]
        cur = [i]
        for c in cand:
            if len(cur) >= batch_size:
                break
            cur.append(c)
        if len(cur) < batch_size:
            need = batch_size - len(cur)
            j = 0
            while need > 0 and j < n:
                if j not in used and j not in cur:
                    cur.append(j)
                    need -= 1
                j += 1
        for x in cur:
            used.add(x)
        batches.append(cur)
    print("Total de lotes difíceis:", len(batches))
    return batches
