"""
Compiled Model Router - Maintains multiple compiled models in memory simultaneously
"""
import torch
from typing import Dict, Any, Tuple, Optional
import copy


class CompiledModelRouter:
    """
    A router that maintains multiple compiled models in memory simultaneously.
    Unlike regular ModelRouter, this keeps compiled versions separate to avoid recompilation.
    """
    
    def __init__(self):
        self._compiled_models: Dict[int, Any] = {}
        self._compile_settings = None
        self._original_models: Dict[int, Any] = {}
        self._warmed_up: set = set()  # Track which models have been warmed up
    
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
                "prevent_unloading": ("BOOLEAN", {"default": True, "tooltip": "Attempt to prevent ComfyUI from unloading compiled models"}),
            },
            "optional": {
                f"model_{i+1}": ("MODEL",) for i in range(8)
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "route_compiled"
    CATEGORY = "Mamad8/Routing"
    DESCRIPTION = "Routes between multiple models while maintaining their compiled states"
    
    def warmup_model(self, compiled_model, model_idx):
        """
        Run dummy inference to trigger kernel compilation and autotuning.
        This forces PyTorch to compile all GPU kernels upfront.
        """
        print(f"[Mamad8] Warming up model {model_idx}...")
        
        try:
            # Create dummy inputs matching typical WanVideo inference
            dummy_latent = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16)
            dummy_timestep = torch.tensor([500], device='cuda')  # Mid-range timestep
            dummy_context = torch.randn(1, 77, 768, device='cuda', dtype=torch.float16)
            
            # For WanVideo models, may need additional inputs
            dummy_kwargs = {
                'context': dummy_context,
                'freqs': None,  # Some models use frequency embeddings
            }
            
            # Run dummy inference to trigger kernel compilation
            with torch.no_grad():
                diffusion_model = compiled_model.get_model_object("diffusion_model")
                _ = diffusion_model(dummy_latent, dummy_timestep, **dummy_kwargs)
            
            print(f"[Mamad8] Model {model_idx} warmup completed successfully")
            self._warmed_up.add(model_idx)
            
        except Exception as e:
            print(f"[Mamad8] Warmup failed for model {model_idx}: {e}")
            print(f"[Mamad8] Model will still work but first inference may be slow")
            # Don't add to warmed_up set so we can retry later
    
    def compile_model_if_needed(self, model, model_idx, backend, fullgraph, mode, dynamic, 
                               compile_transformer_blocks_only, dynamo_cache_size_limit, prevent_unloading):
        """
        Compile a model if it hasn't been compiled yet with current settings.
        """
        # Check if compilation settings have changed
        current_settings = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "compile_transformer_blocks_only": compile_transformer_blocks_only,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "prevent_unloading": prevent_unloading
        }
        
        if self._compile_settings != current_settings:
            # Settings changed, clear all compiled models
            self._compiled_models.clear()
            self._warmed_up.clear()  # Also clear warmup state
            self._compile_settings = current_settings
            print(f"[Mamad8] Compilation settings changed, clearing compiled model cache")
        
        # Check if this model is already compiled
        if model_idx in self._compiled_models:
            cached_model = self._compiled_models[model_idx]
            print(f"[Mamad8] Using cached compiled model {model_idx}")
            
            # Check if warmup is needed
            if prevent_unloading and model_idx not in self._warmed_up:
                self.warmup_model(cached_model, model_idx)
            
            # Check if model is still on GPU (debugging)
            if prevent_unloading and hasattr(cached_model, 'model'):
                try:
                    if hasattr(cached_model.model, 'device'):
                        print(f"[Mamad8] Model {model_idx} is on device: {cached_model.model.device}")
                    elif hasattr(cached_model.model, 'parameters'):
                        first_param = next(cached_model.model.parameters(), None)
                        if first_param is not None:
                            print(f"[Mamad8] Model {model_idx} parameters on device: {first_param.device}")
                except Exception as debug_e:
                    print(f"[Mamad8] Could not check device for model {model_idx}: {debug_e}")
            
            return cached_model
        
        print(f"[Mamad8] Compiling model {model_idx} for the first time...")
        
        # Store original model reference
        self._original_models[model_idx] = model
        
        # Clone the model to avoid modifying the original
        m = model.clone()
        
        # CRITICAL: Prevent ComfyUI from unloading this model by setting memory management flags
        # Force the model to stay loaded in memory
        if prevent_unloading:
            if hasattr(m, 'model_options'):
                if 'model_management' not in m.model_options:
                    m.model_options['model_management'] = {}
                m.model_options['model_management']['force_load'] = True
                m.model_options['model_management']['keep_loaded'] = True
                print(f"[Mamad8] Set model {model_idx} to stay loaded in memory")
            
            # Also try to set model to not unload
            if hasattr(m, 'model'):
                if hasattr(m.model, 'model_options'):
                    if 'model_management' not in m.model.model_options:
                        m.model.model_options['model_management'] = {}
                    m.model.model_options['model_management']['force_load'] = True
                    m.model.model_options['model_management']['keep_loaded'] = True
        
        # Set dynamo cache size
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        try:
            # Import the compile wrapper from KJNodes
            from comfy_api.torch_helpers import set_torch_compile_wrapper
            
            diffusion_model = m.get_model_object("diffusion_model")
            
            if compile_transformer_blocks_only:
                compile_key_list = []
                for i, block in enumerate(diffusion_model.blocks):
                    compile_key_list.append(f"diffusion_model.blocks.{i}")
            else:
                compile_key_list = ["diffusion_model"]
            
            set_torch_compile_wrapper(
                model=m, 
                keys=compile_key_list, 
                backend=backend, 
                mode=mode, 
                dynamic=dynamic, 
                fullgraph=fullgraph
            )
            
            # Store the compiled model
            self._compiled_models[model_idx] = m
            
            # EXPERIMENTAL: Try to keep model on GPU
            if prevent_unloading:
                try:
                    # Force model to GPU and keep it there
                    if hasattr(m, 'model') and hasattr(m.model, 'to'):
                        m.model.to('cuda')
                        print(f"[Mamad8] Forced model {model_idx} to GPU")
                    
                    # Disable automatic memory management for this model
                    if hasattr(m, 'model_options'):
                        m.model_options['disable_smart_memory'] = True
                        
                except Exception as gpu_e:
                    print(f"[Mamad8] Could not force model {model_idx} to GPU: {gpu_e}")
            
            print(f"[Mamad8] Successfully compiled and cached model {model_idx}")
            
            # Warmup the newly compiled model if requested
            if prevent_unloading:
                self.warmup_model(m, model_idx)
            
        except Exception as e:
            print(f"[Mamad8] Failed to compile model {model_idx}: {e}")
            # Return original model if compilation fails
            return model
        
        return m
    
    def route_compiled(self, model_selector, backend, fullgraph, mode, dynamic,
                      compile_transformer_blocks_only, dynamo_cache_size_limit, prevent_unloading, **kwargs):
        """
        Route to the selected model, compiling it if necessary.
        """
        selected_input = f"model_{model_selector}"
        
        if selected_input not in kwargs or kwargs[selected_input] is None:
            raise ValueError(f"Model input {model_selector} (model_{model_selector}) is not connected")
        
        model = kwargs[selected_input]
        
        # Get or create compiled version
        compiled_model = self.compile_model_if_needed(
            model, model_selector, backend, fullgraph, mode, dynamic,
            compile_transformer_blocks_only, dynamo_cache_size_limit, prevent_unloading
        )
        
        return (compiled_model,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when model_selector changes
        return kwargs.get("model_selector", 1) 