import os
import sys
import subprocess
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor, TrainingArguments, Trainer
from huggingface_hub import login
from PIL import Image, ImageFile

def _ensure(pkgs):
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--quiet"])

_ensure(["protobuf"])

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

TOKEN = ""
login(token=TOKEN)

MODEL_NAME = "google/siglip-base-patch16-256-multilingual"
DATASET_NAME = "Artificio/WikiArt"
DATASET_SPLIT = "train"
TEXT_COLUMN_ORIGINAL = "description"
HUB_MODEL_ID = "turing552/siglip-wikiart-5ep"
OUTPUT_DIR = "./siglip-fwikiart-v1"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-6
MAX_LENGTH = 64
SEED = 42
TEST_SIZE = 0.1
VAL_RATIO = 0.1
NUM_WORKERS = max(2, (os.cpu_count() or 2) // 2)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.train()

def processar_como_esta(example):
    texto = example.get(TEXT_COLUMN_ORIGINAL, "")
    if isinstance(texto, list):
        if len(texto) > 0:
            texto_original = texto[0]
        else:
            texto_original = ""
    else:
        texto_original = str(texto)
    example["text_caption"] = str(texto_original).strip()
    return example

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset = dataset.map(
    processar_como_esta,
    num_proc=NUM_WORKERS,
    remove_columns=[c for c in dataset.column_names if c not in ["image", "text_caption"]]
)

split_full = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
trainval_dataset = split_full["train"]
test_dataset = split_full["test"]

split_trainval = trainval_dataset.train_test_split(test_size=VAL_RATIO, seed=SEED)
train_dataset = split_trainval["train"]
eval_dataset = split_trainval["test"]

class CLIPDataset(Dataset):
    def __init__(self, hf_ds, processor, max_length):
        self.ds = hf_ds
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        txt = ex["text_caption"]
        enc = self.processor(
            text=txt,
            images=img,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return {k: v.squeeze(0) for k, v in enc.items()}

train_ds = CLIPDataset(train_dataset, processor, MAX_LENGTH)
eval_ds = CLIPDataset(eval_dataset, processor, MAX_LENGTH)

class CLIPDataCollator:
    def __call__(self, features):
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        return batch

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pixel_values = inputs.pop("pixel_values")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        bsz = input_ids.size(0)
        target_matrix = torch.eye(bsz, device=logits_per_image.device)
        loss_i = F.binary_cross_entropy_with_logits(logits_per_image, target_matrix)
        loss_t = F.binary_cross_entropy_with_logits(logits_per_text, target_matrix)
        loss = (loss_i + loss_t) / 2
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        loss = self.compute_loss(model, inputs, return_outputs=False)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), None, None)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
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
    bf16=False,
    save_safetensors=False,
    seed=SEED
)

trainer = CLIPTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=CLIPDataCollator()
)

print("Tamanho total do dataset:", len(dataset))
print("Tamanho train+val:", len(trainval_dataset))
print("Tamanho test (nunca usado no treino):", len(test_dataset))
print("Tamanho train:", len(train_dataset))
print("Tamanho val:", len(eval_dataset))

trainer.train()
trainer.save_model()
processor.save_pretrained(args.output_dir)
trainer.push_to_hub(token=TOKEN)
processor.push_to_hub(HUB_MODEL_ID, token=TOKEN)
print(f"Concluído: {HUB_MODEL_ID}")
