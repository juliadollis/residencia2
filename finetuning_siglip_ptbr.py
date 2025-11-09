import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import SiglipModel, SiglipProcessor, TrainingArguments, Trainer, DefaultDataCollator
from huggingface_hub import login
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

login(write_permission=True)

MODEL_NAME = "google/siglip-base-patch16-224"
HUB_MODEL_ID = "juliadollis/sTESTEiglip-finetuned-all-pt-br"
OUTPUT_DIR = "./siglip-finetuned-all-pt-br"
BATCH_SIZE = 64
SEED = 42
NUM_WORKERS = os.cpu_count() // 2 or 2

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SiglipProcessor.from_pretrained(MODEL_NAME)
model = SiglipModel.from_pretrained(MODEL_NAME).to(device)
model.train()

tok = processor.tokenizer
if getattr(tok, "pad_token", None) is None:
    tok.pad_token = tok.eos_token or tok.unk_token or "[PAD]"
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tok.pad_token_id

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

datasets_list = []

ds_f8k = try_load("laicsiifes/flickr8k-pt-br", "train")
if ds_f8k:
    ds_f8k = ds_f8k.map(lambda x: normalize(x, "caption"), num_proc=NUM_WORKERS)
    ds_f8k = ds_f8k.remove_columns([c for c in ds_f8k.column_names if c not in ["image", "text_caption"]])
    datasets_list.append(ds_f8k)

ds_f30k = try_load("laicsiifes/flickr30k-pt-br", "train")
if ds_f30k:
    ds_f30k = ds_f30k.map(lambda x: normalize(x, "caption"), num_proc=NUM_WORKERS)
    ds_f30k = ds_f30k.remove_columns([c for c in ds_f30k.column_names if c not in ["image", "text_caption"]])
    datasets_list.append(ds_f30k)

ds_coco = try_load("laicsiifes/coco-captions-pt-br", "train")
if ds_coco:
    ds_coco = ds_coco.map(lambda x: normalize(x, "caption"), num_proc=NUM_WORKERS)
    ds_coco = ds_coco.remove_columns([c for c in ds_coco.column_names if c not in ["image", "text_caption"]])
    datasets_list.append(ds_coco)

ds_nocaps = try_load("laicsiifes/nocaps-pt-br", "test")
if ds_nocaps:
    ds_nocaps = ds_nocaps.map(lambda x: normalize(x, "annotations_captions"), num_proc=NUM_WORKERS)
    ds_nocaps = ds_nocaps.remove_columns([c for c in ds_nocaps.column_names if c not in ["image", "text_caption"]])
    datasets_list.append(ds_nocaps)

if len(datasets_list) == 0:
    raise ValueError("Nenhum dataset carregado com sucesso.")

merged = concatenate_datasets(datasets_list).shuffle(seed=SEED)
split = merged.train_test_split(test_size=0.1, seed=SEED)
train_dataset = split["train"]
eval_dataset = split["test"]

class CaptionDataset(Dataset):
    def __init__(self, hf_ds, processor, max_length=64):
        self.ds = hf_ds
        self.processor = processor
        self.max_length = max_length
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        try:
            ex = self.ds[idx]
            img = ex["image"]
            if isinstance(img, dict):
                img = Image.open(img["path"]).convert("RGB")
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

train_ds = CaptionDataset(train_dataset, processor)
eval_ds = CaptionDataset(eval_dataset, processor)

def contrastive_loss_from_inputs(model, inputs):
    pixel_values = inputs["pixel_values"].to(model.device)
    input_ids = inputs["input_ids"].to(model.device)
    attn = inputs["attention_mask"].to(model.device)
    img = model.get_image_features(pixel_values=pixel_values)
    txt = model.get_text_features(input_ids=input_ids, attention_mask=attn)
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    logits = img @ txt.t() / 0.07
    targets = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    return loss

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = contrastive_loss_from_inputs(model, inputs)
        return (loss, {"loss": loss}) if return_outputs else loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = contrastive_loss_from_inputs(model, inputs)
        return (loss, None, None)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=5e-6,
    warmup_ratio=0.01,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="every_save",
    remove_unused_columns=False,
    dataloader_num_workers=NUM_WORKERS,
    dataloader_pin_memory=True,
    bf16=torch.cuda.is_available(),
    save_safetensors=True,
    seed=SEED
)

trainer = ContrastiveTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DefaultDataCollator()
)

print(f"Treinando com {len(train_ds)} exemplos e {len(eval_ds)} de teste.")
trainer.train()
trainer.save_model()
processor.save_pretrained(args.output_dir)
trainer.push_to_hub()
processor.push_to_hub(HUB_MODEL_ID)
print("--- Fine-tuning completo e push concluído ---")
