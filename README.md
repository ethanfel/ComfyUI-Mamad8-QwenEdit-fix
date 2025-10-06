# ComfyUI-Mamad8

A ComfyUI custom node package that provides router nodes for optimized batch processing, preventing unnecessary model loading and unloading.

## Overview

When generating multiple images with different prompts in ComfyUI, the text encoder and VAE are typically loaded and unloaded for each generation. This creates inefficiency as models are repeatedly moved in and out of memory.

ComfyUI-Mamad8 provides router nodes and caching functionality that optimize workflows by:
1. **Batching**: Process all text conditioning at once before unloading the text encoder
2. **Batching**: Process all latent operations together before unloading the VAE  
3. **Caching**: Save conditioning results to disk and reload instantly for repeated prompts
4. **Routing**: Dynamically switch between multiple loaded models

This significantly improves performance when generating multiple variations or reusing prompts.

## Nodes

### ConditioningRouter
- **Purpose**: Routes multiple conditioning inputs to outputs
- **Inputs**: Up to 8 optional conditioning inputs (`conditioning_1` through `conditioning_8`)
- **Outputs**: Up to 8 conditioning outputs (same order as inputs)
- **Use Case**: Batch multiple text prompts to avoid repeated text encoder loading/unloading

### LatentRouter
- **Purpose**: Routes multiple latent inputs to outputs  
- **Inputs**: Up to 8 optional latent inputs (`latent_1` through `latent_8`)
- **Outputs**: Up to 8 latent outputs (same order as inputs)
- **Use Case**: Batch multiple latent operations to avoid repeated VAE loading/unloading

### ModelRouter
- **Purpose**: Selects one model from multiple model inputs based on an integer selector
- **Inputs**: 
  - `model_selector` (required): Integer from 1 to 8 to select which model to output
  - Up to 8 optional model inputs (`model_1` through `model_8`)
- **Outputs**: 1 model output (the selected model)
- **Use Case**: Dynamic model switching while keeping multiple models loaded in memory

### CLIP Text Encode (Mamad8)
- **Purpose**: CLIP text encoding with disk caching for repeated prompts
- **Inputs**:
  - `text` (required): Text prompt to encode
  - `clip` (required): CLIP model
  - `use_cache` (required): Boolean to enable/disable caching
- **Outputs**: 1 conditioning output
- **Use Case**: Cache conditioning results for frequently used prompts to avoid re-encoding

### Compiled Model Router (Mamad8)
- **Purpose**: Routes between multiple models while maintaining separate compiled states + warmup
- **Inputs**:
  - `model_selector` (required): Integer from 1 to 8 to select which model to output
  - Compilation settings (all required):
    - `backend`: "inductor" or "cudagraphs"
    - `fullgraph`: Enable full graph compilation
    - `mode`: Compilation optimization mode
    - `dynamic`: Handle dynamic shapes
    - `compile_transformer_blocks_only`: Compile only transformer blocks
    - `dynamo_cache_size_limit`: PyTorch Dynamo cache size (default: 1024)
    - `prevent_unloading`: Attempt to prevent ComfyUI from unloading + trigger warmup (default: True)
  - Up to 8 optional model inputs (`model_1` through `model_8`)
- **Outputs**: 1 compiled model output
- **Use Case**: Switch between multiple torch.compiled models with automatic warmup to reduce first-run delays

### Weight Swapping Router (Mamad8) 🚀
- **Purpose**: Ultra-fast model switching via weight replacement (same architecture models only)
- **Inputs**:
  - `model_selector` (required): Integer from 1 to 8 to select which model to output
  - Compilation settings (all required):
    - `backend`: "inductor" or "cudagraphs"
    - `fullgraph`: Enable full graph compilation
    - `mode`: Compilation optimization mode
    - `dynamic`: Handle dynamic shapes
    - `compile_transformer_blocks_only`: Compile only transformer blocks
    - `dynamo_cache_size_limit`: PyTorch Dynamo cache size (default: 1024)
    - `cache_weights_in_ram`: Keep all model weights cached in RAM (default: True)
  - Up to 8 optional model inputs (`model_1` through `model_8`)
- **Outputs**: 1 compiled model output
- **Use Case**: **BEST for Wan base + LoRAs** - Compile once, switch weights instantly (~1-2s)

## Usage Example

1. **Multiple Text Prompts**: Connect multiple CLIP Text Encode nodes to the ConditioningRouter inputs
2. **Batch Processing**: The router ensures all conditioning is processed before the text encoder is unloaded
3. **Generation**: Connect the router outputs to your K-Samplers or other nodes
4. **Latent Processing**: Similarly, use LatentRouter for VAE operations
5. **Model Selection**: Use ModelRouter to dynamically switch between different models
   - Connect multiple models to `model_1`, `model_2`, etc.
   - Set `model_selector` to 1, 2, 3... to choose which model to use
   - Example: Set selector to 3 to use the model connected to `model_3` input
6. **Cached Text Encoding**: Use CLIP Text Encode (Mamad8) for frequently used prompts
   - Enable `use_cache` to save conditioning results to disk
   - First time: processes text and saves to cache
   - Subsequent times: loads instantly from cache
   - Disable `use_cache` for normal CLIP Text Encode behavior
7. **Compiled Model Routing**: Use Compiled Model Router for torch.compiled models
   - Replaces: ModelRouter → TorchCompileModelWanVideoV2 workflow
   - Connect multiple models to `model_1`, `model_2`, etc.
   - First time each model is selected: compiles, caches, and warms up
   - Subsequent selections: fast switching (warmup moves delay to controlled point)
   - All compilation settings integrated into the router
8. **Weight Swapping Routing** (RECOMMENDED for Wan base + LoRAs):
   - **Perfect for your use case**: All models share same architecture
   - Connect multiple model variants to `model_1`, `model_2`, etc.
   - First model selection: compiles once and warms up (~160s one-time cost)
   - Subsequent selections: **Instant switching** (~1-2s weight copy)
   - All model weights cached in your 192GB RAM
   - **True zero-penalty switching** after initial setup

## Installation

1. Clone or download this repository to your ComfyUI `custom_nodes` directory:
   ```
   cd ComfyUI/custom_nodes
   git clone <repository-url> ComfyUI-Mamad8
   ```
2. Restart ComfyUI
3. The nodes will appear under the "Mamad8/Routing" category

## Benefits

- **Performance**: Reduces model loading/unloading overhead + instant cached text encoding
- **Memory Efficiency**: Better VRAM management for multi-generation workflows
- **Time Savings**: Skip repeated CLIP encoding for identical prompts
- **Simplicity**: Drop-in solution that works with existing workflows
- **Scalability**: Supports up to 8 parallel operations
- **Persistent Cache**: Conditioning cache persists across ComfyUI restarts

## Technical Details

- **ConditioningRouter & LatentRouter**: 
  - Use optional inputs, so you only need to connect what you're using
  - Input/output order is preserved (input_1 → output_1, input_2 → output_2, etc.)
  - Unused outputs return `None` and can be ignored
  - Compatible with all ComfyUI conditioning and latent types

- **ModelRouter**:
  - Requires a `model_selector` integer input (1-8)
  - Only outputs the model connected to the selected input slot
  - If selected model input is not connected, throws an error (expected behavior)
  - Compatible with all ComfyUI model types (UNET, VAE, CLIP, etc.)

- **CLIP Text Encode (Mamad8)**:
  - Uses MD5 hashing for fast text fingerprinting
  - Cache files stored in `ComfyUI-Mamad8/conditioning_cache/` directory
  - Cache files named with text hash (e.g., `a1b2c3d4e5f6.pkl`)
  - Automatic cache directory creation on first use
  - Graceful fallback to normal encoding if cache fails
  - Console logging for cache hits/misses and operations

- **Compiled Model Router (Mamad8)**:
  - **Compilation caching + automatic warmup**: Prevents recompilation and first-run delays
  - Maintains separate compiled instances for each model in RAM
  - Automatic model warmup with dummy inference when `prevent_unloading=True`
  - Warmup moves 60-160s delay to a controlled point (after cache retrieval, before real inference)
  - Stores compilation state per model index (1-8)  
  - Detects compilation setting changes and recompiles if needed
  - Console logging shows compilation, warmup, and device status
  - Designed for different model architectures or when weight swapping isn't applicable

- **Weight Swapping Router (Mamad8)** 🚀:
  - **Ultimate solution for same-architecture models**: Compile once, swap weights instantly
  - Perfect for Wan base model + different LoRAs (your exact use case!)
  - Single compiled model instance with hot-swappable weights in RAM
  - Weight switching takes ~1-2 seconds (just memory copy operations)
  - One-time compilation and warmup cost (~160s), then truly instant switching
  - Automatic weight caching in RAM with `cache_weights_in_ram=True`
  - Console logging shows weight caching and switching operations
  - **Solves the PyTorch lazy kernel compilation issue completely**

## License

This project is released under the same license as ComfyUI. 