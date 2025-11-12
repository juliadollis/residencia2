import random
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

AUTO_CAPTION_KEYS = ["captions", "caption", "text", "raw", "sentence", "description"]

def pick_first(x):
    if isinstance(x, list) and len(x) > 0:
        return x[0]
    return x

def to_rgb(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "path" in img:
        return Image.open(img["path"]).convert("RGB")
    return Image.open(img).convert("RGB")

def resolve_caption_key(row, caption_col, caption_is_list):
    if caption_col != "auto":
        return caption_col, caption_is_list
    for k in AUTO_CAPTION_KEYS:
        if k in row:
            v = row[k]
            if caption_is_list is None:
                if isinstance(v, list):
                    return k, True
                return k, False
            return k, caption_is_list
    keys = list(row.keys())
    raise KeyError(f"Nenhuma coluna de texto encontrada. Chaves disponíveis: {keys}")

def build_pairs(dataset_name, split, image_col, caption_col, caption_is_list, max_examples=None, seed=3407):
    print("Carregando dataset:", dataset_name, "split:", split)
    ds = load_dataset(dataset_name, split=split)
    n = len(ds)
    print("Total de exemplos no split:", n)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    if max_examples is not None:
        idxs = idxs[:max_examples]
        print("Limitando para max_examples:", len(idxs))
    first_row = ds[0] if n > 0 else None
    if first_row is None:
        raise RuntimeError("Dataset vazio.")
    cap_key, cap_is_list = resolve_caption_key(first_row, caption_col, caption_is_list)
    print("Coluna de imagem:", image_col)
    print("Coluna de texto escolhida:", cap_key, "lista:", cap_is_list)
    images = []
    texts = []
    for i in tqdm(idxs, desc="Construindo pairs", unit="ex"):
        row = ds[i]
        cap = row[cap_key]
        if cap_is_list:
            cap = pick_first(cap)
        images.append(to_rgb(row[image_col]))
        texts.append(str(cap))
    print("Pairs construídos. Imagens:", len(images), "Textos:", len(texts))
    return images, texts
