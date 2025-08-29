"""
Thunder Compute HRM Integration for ComfyUI
Custom nodes for ultra-fast image generation with 65x speedup
"""

import sys
import traceback

try:
    from .thunder_compute_hrm_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("✅ Thunder Compute HRM nodes loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Thunder Compute HRM nodes: {e}")
    print(traceback.format_exc())
    # Create empty mappings to prevent ComfyUI from failing
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']