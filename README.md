# spiral_pytorch_ray
 The spiral paper (https://github.com/deepmind/spiral) training pipeline implementation in pytorch and ray

Requirements:
2 GPUs. One for policy learning and one for discriminator learning.

Usage: 
1. Install https://github.com/deepmind/spiral following the instructions (need the libmypaint environment)
2. Copy all python scripts here to spiral/
3. Download some data. Look at real_image_loader.py for dataset location/format
4. Run **python spiral_torch.py**
