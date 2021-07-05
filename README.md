# spiral_pytorch_ray
 The original spiral paper (https://arxiv.org/abs/1804.01118) training pipeline implementation in pytorch and ray

Requirements:
2 GPUs. One for policy learning and one for discriminator learning.

Usage: 
1. Install https://github.com/deepmind/spiral following the instructions (need the libmypaint environment)
2. Copy all python scripts here to spiral/
3. Download some data. Look at real_image_loader.py for dataset location/format
4. Run **python spiral_torch.py**

<br/>

15000 policy training steps on digit 4 in mnist (each training step is n_batches * n_timesteps, or 64*10):

<br/>

![Alt Text](examples/4_drawing_1.gif)
![Alt Text](examples/4_drawing_3.gif)
![Alt Text](examples/4_drawing_5.gif)

<br/>
<br/>

![Alt Text](examples/4_training.gif)
