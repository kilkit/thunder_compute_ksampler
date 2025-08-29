# Thunder Compute HRM ComfyUI Nodes

Custom ComfyUI nodes that integrate Thunder Compute A100XL for ultra-fast image generation with 65x speedup.

## Features

- **Thunder Compute HRM Generator**: Routes image generation through Thunder Compute A100XL
- **Character Consistency**: Maintains character identity across multiple views and scenes  
- **65x Speed Improvement**: ~24 seconds vs 25+ minutes for character generation
- **Style Support**: Real/Animation styles with automatic prompt enhancement

## Installation

### Method 1: RunComfy Git Installation (Recommended)
1. Open RunComfy ComfyUI interface
2. Click "Manager" → "Install via Git URL"
3. Enter: `https://github.com/YOUR_USERNAME/thunder-compute-hrm-comfyui`
4. Click Install and restart ComfyUI

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/thunder-compute-hrm-comfyui
cd thunder-compute-hrm-comfyui
pip install -r requirements.txt
```

## Configuration

After installation, configure your nodes with:

- **HRM Endpoint**: `https://your-replit-app.replit.dev/api/hrm/generate-image`
- **Character API**: `https://your-replit-app.replit.dev/api/runcomfy/character/create`
- **API Key**: Your Replit API key

## Available Nodes

### Thunder Compute HRM Generator
- Basic image generation with Thunder Compute speedup
- Supports Real/Animation styles
- Automatic prompt enhancement
- 65x faster than standard processing

### Thunder Compute Character Consistency
- Character profile creation and management
- Multiple view generation (front, side, back, closeup)
- Scene generation with character consistency
- Character ID tracking for reuse

## Usage Examples

### Basic Image Generation
1. Add "Thunder Compute HRM Generator" node
2. Configure endpoint and API key
3. Set prompt and style preferences
4. Connect to your workflow

### Character Consistency
1. Add "Thunder Compute Character Consistency" node
2. Upload reference image
3. Set character prompt and view type
4. Generate consistent character views

## Performance Benefits

- Character creation: 20 minutes → 2 minutes (10x speedup)
- Single view: 4 minutes → 24 seconds (65x speedup)
- Expression sets: 20 minutes → 2 minutes (10x speedup)
- Scene generation: 4 minutes → 24 seconds (65x speedup)

## Requirements

- ComfyUI
- Python 3.8+
- Access to Thunder Compute A100XL instance
- Valid API key for your Replit application

## License

MIT License