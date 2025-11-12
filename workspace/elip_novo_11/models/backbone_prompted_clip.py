import torch
from transformers import CLIPModel, CLIPProcessor

class PromptedCLIP(torch.nn.Module):
    def __init__(self, model_name: str, image_size: int):
        super().__init__()
        print("Carregando CLIP", model_name)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32, low_cpu_mem_usage=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "size"):
            if isinstance(self.processor.image_processor.size, dict):
                if "shortest_edge" in self.processor.image_processor.size:
                    self.processor.image_processor.size["shortest_edge"] = int(image_size)
                if "height" in self.processor.image_processor.size and "width" in self.processor.image_processor.size:
                    self.processor.image_processor.size["height"] = int(image_size)
                    self.processor.image_processor.size["width"] = int(image_size)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.image_size = image_size
        self.d_model = self.model.vision_model.config.hidden_size
        self.d_proj = self.model.config.projection_dim
        print("Backbone congelado. d_model:", self.d_model, "d_proj:", self.d_proj)

    @torch.no_grad()
    def text_hidden(self, texts):
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.model.device)
        out = self.model.get_text_features(**inputs)
        z = torch.nn.functional.normalize(out, dim=-1)
        return z

    @torch.no_grad()
    def encode_images_base(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.device)
        out = self.model.get_image_features(pixel_values=pixel_values)
        z = torch.nn.functional.normalize(out, dim=-1)
        return z

    def encode_images_guided(self, images, prompt_tokens):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.device)
        vm = self.model.vision_model
        hs = vm.embeddings(pixel_values)
        cls = hs[:, :1, :]
        patches = hs[:, 1:, :]
        pt = prompt_tokens.to(hs.dtype)
        hs2 = torch.cat([cls, pt, patches], dim=1)
        enc_out = vm.encoder(hs2)
        hs_last = enc_out.last_hidden_state
        cls_last = hs_last[:, 0, :]
        cls_last = vm.post_layernorm(cls_last)
        z = self.model.visual_projection(cls_last)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z
