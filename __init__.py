"""
ComfyUI-Mamad8 - Router nodes for optimized batch processing
"""
import os
import math
import hashlib
import torch
import pickle
from pathlib import Path

import comfy.utils
import node_helpers

# Get the directory where this custom node is located
CURRENT_DIR = Path(__file__).parent
CACHE_DIR = CURRENT_DIR / "conditioning_cache"

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)

class QwenImagePreprocessMamad8:
    """Resize images the same way as TextEncodeQwenImageEditPlus."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("vl_image", "vae_image", "vae_latents")
    FUNCTION = "resize"
    CATEGORY = "Mamad8/Image"

    def resize(self, image, vae=None):
        """Apply Qwen image preprocessing resize with optional VAE encoding."""
        samples = image.movedim(-1, 1)

        # Vision-language resize: scale to match a 384x384 area while keeping aspect ratio.
        vl_target_area = 384 * 384
        scale_by = math.sqrt(vl_target_area / (samples.shape[3] * samples.shape[2]))
        vl_width = round(samples.shape[3] * scale_by)
        vl_height = round(samples.shape[2] * scale_by)
        vl_samples = comfy.utils.common_upscale(samples, vl_width, vl_height, "area", "disabled")
        vl_image = vl_samples.movedim(1, -1)

        # Reference latent resize: scale to match a 1024x1024 area, snapped to multiples of 8.
        latent_target_area = 1024 * 1024
        latent_scale = math.sqrt(latent_target_area / (samples.shape[3] * samples.shape[2]))
        latent_width = max(8, round(samples.shape[3] * latent_scale / 8.0) * 8)
        latent_height = max(8, round(samples.shape[2] * latent_scale / 8.0) * 8)
        latent_samples = comfy.utils.common_upscale(samples, latent_width, latent_height, "area", "disabled")
        vae_image = latent_samples.movedim(1, -1)

        vae_latents = None
        if vae is not None:
            vae_latents = vae.encode(vae_image[:, :, :, :3])

        return (vl_image, vae_image, vae_latents)


class TextEncodeQwenImageEditPlusMamad8:
    """Qwen image edit encoder with configurable resize targets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": True,
                }),
                "vl_target_pixels": ("INT", {
                    "default": 384 * 384,
                    "min": 64 * 64,
                    "max": 8192 * 8192,
                    "step": 64,
                    "display": "number",
                }),
                "latent_target_pixels": ("INT", {
                    "default": 1024 * 1024,
                    "min": 64 * 64,
                    "max": 8192 * 8192,
                    "step": 64,
                    "display": "number",
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Mamad8/Text"

    def encode(
        self,
        clip,
        prompt,
        vl_target_pixels,
        latent_target_pixels,
        vae=None,
        image1=None,
        image2=None,
        image3=None,
    ):
        """Resize inputs for Qwen image edit using custom pixel targets."""

        images = [image1, image2, image3]
        images_vl = []
        ref_latents = []

        vl_target_pixels = max(1, int(vl_target_pixels))
        latent_target_pixels = max(1, int(latent_target_pixels))

        llama_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        image_prompt = ""

        for idx, image in enumerate(images):
            if image is None:
                continue

            samples = image.movedim(-1, 1)
            source_area = samples.shape[3] * samples.shape[2]
            if source_area == 0:
                continue

            # Resize for the vision-language encoder using the custom area target.
            vl_scale = math.sqrt(vl_target_pixels / source_area)
            vl_width = max(1, round(samples.shape[3] * vl_scale))
            vl_height = max(1, round(samples.shape[2] * vl_scale))
            vl_samples = comfy.utils.common_upscale(
                samples,
                vl_width,
                vl_height,
                "area",
                "disabled",
            )
            images_vl.append(vl_samples.movedim(1, -1))

            if vae is not None:
                # Resize for reference latents with the custom area, snapped to multiples of 8.
                latent_scale = math.sqrt(latent_target_pixels / source_area)
                latent_width = max(8, round(samples.shape[3] * latent_scale / 8.0) * 8)
                latent_height = max(8, round(samples.shape[2] * latent_scale / 8.0) * 8)
                latent_samples = comfy.utils.common_upscale(
                    samples,
                    latent_width,
                    latent_height,
                    "area",
                    "disabled",
                )
                ref_latents.append(
                    vae.encode(latent_samples.movedim(1, -1)[:, :, :, :3])
                )

            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(idx + 1)

        tokens = clip.tokenize(
            image_prompt + prompt,
            images=images_vl,
            llama_template=llama_template,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True,
            )

        return (conditioning,)


class CLIPTextEncodeMamad8:
    """
    A CLIP Text Encode node with caching capability.
    When caching is enabled, it saves/loads conditioning results to/from disk based on text hash.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": True
                }),
                "clip": ("CLIP",),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Cache ON",
                    "label_off": "Cache OFF"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Mamad8/Text"

    def encode(self, text, clip, use_cache):
        """
        Encode text using CLIP with optional caching.
        """
        if not use_cache:
            # Regular CLIP Text Encode behavior
            tokens = clip.tokenize(text)
            output = clip.encode_from_tokens(tokens, return_pooled=True)
            cond = output[0]
            pooled = output[1]
            return ([[cond, {"pooled_output": pooled}]],)
        
        # Caching is enabled
        # Create hash of the text for filename
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = CACHE_DIR / f"{text_hash}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_conditioning = pickle.load(f)
                print(f"[Mamad8] Loaded cached conditioning for text hash: {text_hash}")
                return (cached_conditioning,)
            except Exception as e:
                print(f"[Mamad8] Failed to load cache file {cache_file}: {e}")
                # Fall through to generate new conditioning
        
        # Generate new conditioning
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True)
        cond = output[0]
        pooled = output[1]
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(conditioning, f)
            print(f"[Mamad8] Saved conditioning to cache: {cache_file}")
        except Exception as e:
            print(f"[Mamad8] Failed to save to cache: {e}")
        
        return (conditioning,)




# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImagePreprocessMamad8": "Qwen Image Preprocess (Mamad8)",
    "TextEncodeQwenImageEditPlusMamad8": "Text Encode Qwen Image Edit Plus (Mamad8)",
    "CLIPTextEncodeMamad8": "CLIP Text Encode (Mamad8)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 

