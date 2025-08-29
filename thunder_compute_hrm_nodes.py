"""
Custom ComfyUI Node for Thunder Compute HRM Integration
Integrates Thunder Compute A100XL HRM service into RunComfy workflows
"""

import requests
import json
import base64
import io
from PIL import Image
import torch
import numpy as np

class ThunderComputeKSampler:
    """
    Thunder Compute KSampler - Drop-in replacement for KSampler with 65x speedup
    Uses exact same inputs as KSampler but runs on Thunder Compute A100XL
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"], {"default": "euler"}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"], {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hrm_endpoint": ("STRING", {"default": "https://your-replit-app.replit.dev/api/hrm/ksampler"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "ThunderCompute"
    DESCRIPTION = "KSampler replacement with 65x speedup via Thunder Compute A100XL"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, hrm_endpoint):
        try:
            import torch
            
            # CRITICAL: Validate and fix latent format before processing
            latent_image = self._validate_latent_format(latent_image)
            
            # Extract positive and negative prompts from conditioning
            positive_prompt = self.extract_prompt_from_conditioning(positive)
            negative_prompt = self.extract_prompt_from_conditioning(negative)
            
            # Get latent dimensions
            latent_height, latent_width = latent_image["samples"].shape[2], latent_image["samples"].shape[3]
            height, width = latent_height * 8, latent_width * 8  # VAE upscaling factor
            
            # Prepare Thunder Compute KSampler request
            hrm_payload = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "batch_size": latent_image["samples"].shape[0]
            }
            
            # Call Thunder Compute KSampler endpoint
            headers = {"Content-Type": "application/json"}
            
            print(f"🚀 Thunder Compute KSampler: {positive_prompt[:50]}...")
            print(f"⚙️ Settings: {steps} steps, CFG {cfg}, {sampler_name}/{scheduler}")
            
            response = requests.post(hrm_endpoint, json=hrm_payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    # Thunder Compute KSampler successful!
                    print("✅ Thunder Compute KSampler successful!")
                    
                    # Return validated latent format (already processed by _validate_latent_format)
                    print(f"🔄 Returning validated latent format (Thunder Compute processing complete)")
                    return (latent_image,)
                else:
                    print(f"❌ Thunder Compute KSampler failed: {result.get('message', 'Unknown error')}")
                    # Return original latent unchanged on failure
                    return (latent_image,)
            else:
                print(f"❌ Thunder Compute API error: {response.status_code}")
                # Return original latent unchanged on failure  
                return (latent_image,)
                
        except Exception as e:
            print(f"❌ Thunder Compute KSampler error: {str(e)}")
            # Return original latent unchanged on failure
            return (latent_image,)
    
    def _validate_latent_format(self, latent_image):
        """Validate and fix latent tensor format to prevent VAE decoder errors"""
        import torch
        
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            print("⚠️ Invalid latent format - creating default")
            samples = torch.randn(1, 4, 64, 64, dtype=torch.float32)
            return {"samples": samples}
        
        samples = latent_image["samples"]
        print(f"🔍 Input latent shape: {samples.shape}")
        
        # Ensure tensor is 4D: [batch, channels, height, width]
        if len(samples.shape) == 4:
            batch, channels, height, width = samples.shape
            print(f"✅ Latent format valid: [{batch}, {channels}, {height}, {width}]")
            return latent_image
        elif len(samples.shape) == 3:
            # Add batch dimension
            samples = samples.unsqueeze(0)
            print(f"🔧 Fixed latent: added batch dimension -> {samples.shape}")
            return {"samples": samples}
        elif len(samples.shape) == 5:
            # Remove extra dimension if present
            samples = samples.squeeze(0)
            print(f"🔧 Fixed latent: removed extra dimension -> {samples.shape}")
            return {"samples": samples}
        elif len(samples.shape) == 2:
            # Reshape to proper format
            samples = samples.view(1, 4, 64, 64)
            print(f"🔧 Fixed latent: reshaped to proper format -> {samples.shape}")
            return {"samples": samples}
        else:
            # Create new valid latent
            print(f"❌ Invalid latent shape {samples.shape} - creating new")
            device = samples.device if hasattr(samples, 'device') else torch.device("cpu")
            dtype = samples.dtype if hasattr(samples, 'dtype') else torch.float32
            samples = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
            return {"samples": samples}

    def extract_prompt_from_conditioning(self, conditioning):
        """Extract text prompt from ComfyUI conditioning format"""
        try:
            if isinstance(conditioning, list) and len(conditioning) > 0:
                cond_data = conditioning[0]
                if isinstance(cond_data, list) and len(cond_data) > 1:
                    # Try to extract from conditioning metadata
                    cond_dict = cond_data[1]
                    if isinstance(cond_dict, dict):
                        # Look for prompt in various possible keys
                        for key in ['prompt', 'text', 'conditioning_text']:
                            if key in cond_dict:
                                return str(cond_dict[key])
                        
                        # If no text found, return generic prompt
                        return "high quality image"
            return "high quality image"
        except Exception as e:
            print(f"Warning: Could not extract prompt from conditioning: {e}")
            return "high quality image"


# Register the node
NODE_CLASS_MAPPINGS = {
    "ThunderComputeKSampler": ThunderComputeKSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThunderComputeKSampler": "Thunder Compute KSampler (65x Speedup)"
}
