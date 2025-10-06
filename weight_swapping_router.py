"""
Weight Swapping Router - Ultra-fast model switching via weight replacement
Compiles once, swaps weights instantly
"""
import torch
from typing import Dict, Any, Optional
import copy
from pathlib import Path


class WeightSwappingRouter:
    """
    A router that maintains one compiled model and swaps weights for different variants.
    Perfect for Wan base model + different LoRAs where architecture is identical.
    """
    
    def __init__(self):
        self._compiled_model = None
        self._weight_cache: Dict[int, Dict] = {}  # Cached state dicts
        self._current_weights = None
        self._compile_settings = None
        self._is_compiled = False
        self._base_model_template = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_selector": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number"
                }),
                # Compilation settings
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": True}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True}),
                "dynamo_cache_size_limit": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 1}),
                "cache_weights_in_ram": ("BOOLEAN", {"default": True, "tooltip": "Keep all model weights cached in RAM for instant switching"}),
            },
            "optional": {
                f"model_{i+1}": ("MODEL",) for i in range(8)
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "route_with_weights"
    CATEGORY = "Mamad8/Routing"
    DESCRIPTION = "Ultra-fast model switching via weight replacement (same architecture only)"
    
    def extract_weights(self, model):
        """Extract state dict from a ComfyUI ModelPatcher for caching"""
        print(f"[Mamad8] Extracting weights from model type: {type(model)}")
        
        try:
            # For ComfyUI ModelPatcher objects, access the underlying model
            if hasattr(model, 'model'):
                print(f"[Mamad8] Using model.model (type: {type(model.model)})")
                pytorch_model = model.model
            elif hasattr(model, 'get_model_object'):
                print(f"[Mamad8] Using model.get_model_object('diffusion_model')")
                pytorch_model = model.get_model_object("diffusion_model")
            else:
                # Fallback for raw PyTorch models
                print(f"[Mamad8] Using model directly")
                pytorch_model = model
            
            print(f"[Mamad8] Extracting parameters from PyTorch model type: {type(pytorch_model)}")
            weights = {name: param.clone().cpu() for name, param in pytorch_model.named_parameters()}
            print(f"[Mamad8] Successfully extracted {len(weights)} weight tensors")
            return weights
            
        except Exception as e:
            print(f"[Mamad8] Error extracting weights: {e}")
            print(f"[Mamad8] Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            
            # Try alternative method - get state dict directly
            if hasattr(model, 'state_dict'):
                print(f"[Mamad8] Trying model.state_dict() fallback")
                state_dict = model.state_dict()
                return {name: param.clone().cpu() for name, param in state_dict.items()}
            else:
                raise e
    
    def apply_weights(self, target_model, state_dict, device='cuda'):
        """Apply cached weights to the compiled ComfyUI model"""
        print(f"[Mamad8] Applying {len(state_dict)} cached weights to model type: {type(target_model)}")
        
        try:
            # For ComfyUI ModelPatcher objects, access the underlying model
            if hasattr(target_model, 'model'):
                print(f"[Mamad8] Applying weights to target_model.model (type: {type(target_model.model)})")
                pytorch_model = target_model.model
            elif hasattr(target_model, 'get_model_object'):
                print(f"[Mamad8] Applying weights to target_model.get_model_object('diffusion_model')")
                pytorch_model = target_model.get_model_object("diffusion_model")
            else:
                # Fallback for raw PyTorch models
                print(f"[Mamad8] Applying weights to target_model directly")
                pytorch_model = target_model
                
            applied_count = 0
            with torch.no_grad():
                for name, param in pytorch_model.named_parameters():
                    if name in state_dict:
                        # Move weight to target device and copy in-place
                        new_weight = state_dict[name].to(device=device, dtype=param.dtype)
                        param.data.copy_(new_weight)
                        applied_count += 1
                    else:
                        print(f"[Mamad8] Warning: Weight {name} not found in cached state dict")
            
            print(f"[Mamad8] Successfully applied {applied_count} weight tensors")
                        
        except Exception as e:
            print(f"[Mamad8] Error applying weights: {e}")
            # Try alternative method - load state dict directly
            try:
                if hasattr(target_model, 'load_state_dict'):
                    # Convert state dict to proper device/dtype
                    device_state_dict = {}
                    for name, param in state_dict.items():
                        device_state_dict[name] = param.to(device=device)
                    target_model.load_state_dict(device_state_dict, strict=False)
                else:
                    raise e
            except Exception as e2:
                print(f"[Mamad8] Failed to apply weights with fallback method: {e2}")
                raise e2
    
    def setup_compiled_model(self, template_model, model_idx, backend, fullgraph, mode, 
                           dynamic, compile_transformer_blocks_only, dynamo_cache_size_limit):
        """Set up the compiled model using the first model as template"""
        print(f"[Mamad8] Setting up compiled model using model {model_idx} as template...")
        
        # Clone the template to avoid modifying original
        self._compiled_model = template_model.clone()
        self._base_model_template = template_model
        
        # Set dynamo cache size
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        try:
            # Import the compile wrapper from KJNodes
            from comfy_api.torch_helpers import set_torch_compile_wrapper
            
            diffusion_model = self._compiled_model.get_model_object("diffusion_model")
            
            if compile_transformer_blocks_only:
                compile_key_list = []
                for i, block in enumerate(diffusion_model.blocks):
                    compile_key_list.append(f"diffusion_model.blocks.{i}")
            else:
                compile_key_list = ["diffusion_model"]
            
            set_torch_compile_wrapper(
                model=self._compiled_model, 
                keys=compile_key_list, 
                backend=backend, 
                mode=mode, 
                dynamic=dynamic, 
                fullgraph=fullgraph
            )
            
            print(f"[Mamad8] Model compilation successful")
            
            # Warmup the compiled model
            self._warmup_compiled_model()
            
            self._is_compiled = True
            print(f"[Mamad8] Weight-swapping router ready! All future switches will be instant.")
            
        except Exception as e:
            print(f"[Mamad8] Failed to compile base model: {e}")
            raise e
    
    def _warmup_compiled_model(self):
        """Warmup the compiled model to avoid first-run delays"""
        print(f"[Mamad8] Warming up compiled model...")
        
        try:
            # Create dummy inputs for WanVideo
            dummy_latent = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16)
            dummy_timestep = torch.tensor([500], device='cuda')
            dummy_context = torch.randn(1, 77, 768, device='cuda', dtype=torch.float16)
            
            dummy_kwargs = {
                'context': dummy_context,
                'freqs': None,
            }
            
            # Run dummy inference
            with torch.no_grad():
                # Handle ComfyUI ModelPatcher structure
                if hasattr(self._compiled_model, 'get_model_object'):
                    diffusion_model = self._compiled_model.get_model_object("diffusion_model")
                elif hasattr(self._compiled_model, 'model'):
                    diffusion_model = self._compiled_model.model
                else:
                    diffusion_model = self._compiled_model
                
                _ = diffusion_model(dummy_latent, dummy_timestep, **dummy_kwargs)
            
            print(f"[Mamad8] Compiled model warmup completed")
            
        except Exception as e:
            print(f"[Mamad8] Warmup failed: {e}")
            print(f"[Mamad8] First inference may be slow, but subsequent ones will be fast")
    
    def cache_model_weights(self, model, model_idx, cache_weights_in_ram):
        """Cache model weights for instant switching"""
        if cache_weights_in_ram and model_idx not in self._weight_cache:
            print(f"[Mamad8] Preparing weight cache slot for model {model_idx}...")
            # Store the model reference instead of extracting weights immediately
            # We'll extract weights only when actually switching
            self._weight_cache[model_idx] = model
            print(f"[Mamad8] Model {model_idx} reference cached (weights will be extracted on-demand)")
    
    def switch_to_weights(self, model_idx, target_model):
        """Switch the compiled model to use different weights"""
        if model_idx == self._current_weights:
            print(f"[Mamad8] Already using weights from model {model_idx}")
            return
        
        print(f"[Mamad8] Switching to weights for model {model_idx}...")
        
        # Use cached model reference or provided target_model
        source_model = self._weight_cache.get(model_idx, target_model)
        
        # Extract and apply weights directly (no intermediate caching to save memory)
        weights = self.extract_weights(source_model)
        self.apply_weights(self._compiled_model, weights)
        
        # Clear the weights from memory immediately after applying
        del weights
        
        # Force garbage collection and clear GPU cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._current_weights = model_idx
        print(f"[Mamad8] Weight switch completed!")
    
    def route_with_weights(self, model_selector, backend, fullgraph, mode, dynamic,
                          compile_transformer_blocks_only, dynamo_cache_size_limit, 
                          cache_weights_in_ram, **kwargs):
        """
        Route to the selected model using weight swapping
        """
        selected_input = f"model_{model_selector}"
        
        if selected_input not in kwargs or kwargs[selected_input] is None:
            raise ValueError(f"Model input {model_selector} (model_{model_selector}) is not connected")
        
        selected_model = kwargs[selected_input]
        
        # Check if compilation settings changed
        current_settings = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "compile_transformer_blocks_only": compile_transformer_blocks_only,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
        }
        
        if self._compile_settings != current_settings:
            print(f"[Mamad8] Compilation settings changed, resetting compiled model")
            self._compiled_model = None
            self._is_compiled = False
            self._weight_cache.clear()
            self._current_weights = None
            self._compile_settings = current_settings
        
        # Set up compiled model if needed (first time or after settings change)
        if not self._is_compiled:
            self.setup_compiled_model(
                selected_model, model_selector, backend, fullgraph, mode,
                dynamic, compile_transformer_blocks_only, dynamo_cache_size_limit
            )
            # Cache the initial model reference (not weights)
            if cache_weights_in_ram:
                self.cache_model_weights(selected_model, model_selector, cache_weights_in_ram)
            self._current_weights = model_selector
            return (self._compiled_model,)
        
        # Cache weights for future use if enabled
        if cache_weights_in_ram:
            self.cache_model_weights(selected_model, model_selector, cache_weights_in_ram)
        
        # Switch to the requested model's weights
        self.switch_to_weights(model_selector, selected_model)
        
        return (self._compiled_model,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when model_selector changes
        return kwargs.get("model_selector", 1) 