
import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    BertTokenizer,
    GenerationConfig,
)
from PIL import Image
import numpy as np


class ReportGenerator(nn.Module):
    

    def __init__(self, mode="impression", max_length=128, device="cuda"):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.mode = mode
        model_name = f"IAMJB/chexpert-mimic-cxr-{mode}-baseline"
        print(f"[G3] Loading report generator: {model_name}")

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)

        # Fully freeze
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)

        # Resolve token IDs robustly (handles VED config quirks)
        self._resolve_token_ids()

        # Generation config
        self.gen_config = {
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._pad_token_id,
            "num_return_sequences": 1,
            "max_length": max_length,
            "min_new_tokens": 8,
            "use_cache": True,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
        }
        if self._bos_token_id is not None:
            self.gen_config["bos_token_id"] = self._bos_token_id

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[G3] Mode: {mode} | Params: {total_params/1e6:.1f}M (FROZEN)")

    def _resolve_token_ids(self):
        """Robustly resolve special token IDs from model/tokenizer config."""
        decoder_cfg = self.model.config.decoder
        cls_id = getattr(self.tokenizer, 'cls_token_id', None)
        sep_id = getattr(self.tokenizer, 'sep_token_id', None)

        self._bos_token_id = (
            getattr(self.model.config, 'bos_token_id', None)
            or getattr(decoder_cfg, 'bos_token_id', None)
            or getattr(self.tokenizer, 'bos_token_id', None)
            or cls_id or sep_id
        )
        self._eos_token_id = (
            getattr(self.model.config, 'eos_token_id', None)
            or getattr(decoder_cfg, 'eos_token_id', None)
            or getattr(self.tokenizer, 'eos_token_id', None)
            or sep_id or cls_id
        )
        self._pad_token_id = (
            getattr(self.model.config, 'pad_token_id', None)
            or getattr(decoder_cfg, 'pad_token_id', None)
            or getattr(self.tokenizer, 'pad_token_id', None)
        )
        self._decoder_start_id = (
            getattr(self.model.config, 'decoder_start_token_id', None)
            or getattr(decoder_cfg, 'decoder_start_token_id', None)
            or cls_id or self._bos_token_id
        )

        # Avoid empty generation when decoder_start == eos
        if (self._decoder_start_id is not None
                and self._eos_token_id is not None
                and self._decoder_start_id == self._eos_token_id
                and sep_id is not None
                and sep_id != self._decoder_start_id):
            self._eos_token_id = sep_id

        # Sync model config
        self.model.config.decoder_start_token_id = self._decoder_start_id
        self.model.config.pad_token_id = self._pad_token_id
        self.model.config.eos_token_id = self._eos_token_id
        if self._bos_token_id is not None:
            self.model.config.bos_token_id = self._bos_token_id

    def _tensor_to_pil(self, img_tensor):
        """Convert image tensor to PIL Image for ViT processor.
        Handles both 3ch and 1ch inputs.
        """
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        # [C, H, W] in [-1, 1] -> [H, W, C] in [0, 255]
        arr = ((img_tensor.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        return Image.fromarray(arr)

    @torch.no_grad()
    def generate_reports(self, images, batch_size=8):
        """Generate reports from image tensors or PIL images.

        Args:
            images: [B, 3, H, W] tensor in [-1,1], or list of PIL Images
            batch_size: processing batch size
        Returns:
            list[str]: generated reports
        """
        reports = []
        if isinstance(images, torch.Tensor) and images.dim() == 3:
            images = images.unsqueeze(0)
        n = len(images) if isinstance(images, list) else images.shape[0]

        for i in range(0, n, batch_size):
            if isinstance(images, torch.Tensor):
                batch = images[i:i + batch_size]
                pil_images = [self._tensor_to_pil(img) for img in batch]
            else:
                pil_images = [
                    img.convert('RGB') if img.mode != 'RGB' else img
                    for img in images[i:i + batch_size]
                ]

            pixel_values = self.image_processor(
                images=pil_images, return_tensors="pt"
            ).pixel_values.to(self.device)

            gen_cfg = GenerationConfig(
                **{**self.gen_config,
                   "decoder_start_token_id": self._decoder_start_id}
            )
            output_ids = self.model.generate(pixel_values, generation_config=gen_cfg)

            batch_reports = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            # Fallback retry for empty outputs
            for j, text in enumerate(batch_reports):
                if not text.strip():
                    fallback_cfg = GenerationConfig(
                        **{**self.gen_config,
                           "decoder_start_token_id": self._decoder_start_id,
                           "num_beams": 1, "do_sample": True,
                           "top_p": 0.9, "temperature": 0.9,
                           "min_new_tokens": 12}
                    )
                    ids = self.model.generate(
                        pixel_values[j:j+1], generation_config=fallback_cfg
                    )
                    batch_reports[j] = self.tokenizer.decode(
                        ids[0], skip_special_tokens=True
                    ).strip()

            reports.extend([r.strip() for r in batch_reports])

        return reports

    def generate_single(self, image):
        """Generate report for a single image."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
        reports = self.generate_reports(image, batch_size=1)
        return reports[0] if reports else ""


class DualReportGenerator(nn.Module):
    """
    Loads both 'findings' and 'impression' G3 models.
    Generates both report types for a given CXR image.
    """

    def __init__(self, max_length=128, device="cuda"):
        super().__init__()
        self.device = device
        print("[G3] Loading dual report generator (findings + impression)...")
        self.findings_model = ReportGenerator(
            mode="findings", max_length=max_length, device=device)
        self.impression_model = ReportGenerator(
            mode="impression", max_length=max_length, device=device)

    @torch.no_grad()
    def generate_both(self, image):
        """Generate both findings and impression for a single image.

        Args:
            image: [1, 3, H, W] tensor in [-1,1], or PIL Image
        Returns:
            dict with 'findings' and 'impression' keys
        """
        return {
            "findings": self.findings_model.generate_single(image),
            "impression": self.impression_model.generate_single(image),
        }

    @torch.no_grad()
    def generate_both_batch(self, images, batch_size=8):
        """Generate both findings and impression for a batch of images.

        Args:
            images: [B, 3, H, W] tensor in [-1,1], or list of PIL Images
        Returns:
            list[dict] with 'findings' and 'impression' keys
        """
        findings = self.findings_model.generate_reports(images, batch_size)
        impressions = self.impression_model.generate_reports(images, batch_size)
        return [
            {"findings": f, "impression": i}
            for f, i in zip(findings, impressions)
        ]
