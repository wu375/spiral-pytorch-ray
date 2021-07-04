import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policies import A2C
from networks import Discriminator
import spiral.environments.libmypaint as libmypaint

from PIL import Image
from pathlib import Path
import cv2

image_paths = list(Path('../datasets/mnist_png/').resolve().glob('*/4/*.png'))
print(image_paths)
exit()


conv0 = nn.Conv2d(3, 16, 5, 2, 2)
conv0.to('cuda:1')
exit()



BRUSHES_PATH = 'third_party/mypaint-brushes-1.3.0/'
env = libmypaint.LibMyPaint(
    episode_length=20,
    canvas_width=64,
    grid_width=32,
    brush_type="classic/dry_brush",
    brush_sizes=[1, 2, 4, 8], # [1, 2, 4, 8, 12, 24],
    use_color=False,
    use_pressure=True,
    use_alpha=False,
    background="white",
    brushes_basedir=BRUSHES_PATH)


# observations keys are 'canvas', 'episode_step', 'episode_length', 'action_mask'
time_step = env.reset()

action_spec = env.action_spec()

checkpoint = torch.load('train_log/run101/model.checkpoint')

a2c = A2C(env.action_spec(), input_shape=(64, 64), grid_shape=(32, 32), action_order="libmypaint")
i = 0
for p in a2c.parameters():
    i += 1
print('policy training with ' + str(i) + ' parameters')

# a2c.set_weights(checkpoint['a2c_weights'])
# discriminator = Discriminator()
# discriminator.set_weights(checkpoint['d_weights'])

# empty_batch = np.stack([time_step.observation["canvas"] for i in range(8)])
# print(np.sum(empty_batch, axis=(1,2,3))/24576)
# # print(len(np.where(empty_batch == 1)[0])/4096)
# exit()
# empty_batch = torch.FloatTensor(empty_batch)
# empty_batch = empty_batch.permute(0, 3, 1, 2)
# score = discriminator(empty_batch).mean()
# print(score)
# exit('success')

a2c.eval()
# Everything is ready for sampling.
state = a2c.initial_state(1)
# noise_sample = np.random.normal(size=(1, 10)).astype(np.float32)
noise_sample = torch.randn(1, 10)

with torch.no_grad():
    for t in range(19):
        # print(time_step.observation['action_mask'])
        time_step.observation["noise_sample"] = noise_sample
        agent_out, state = a2c.PI(time_step.step_type, time_step.observation, state)
        action = agent_out.action
        # print(action)
        # print(agent_out.policy_logits)
        # print()
        # print()
        time_step = env.step(action)

img = time_step.observation["canvas"]
if img.shape[2] != 3:
    img = np.repeat(img, 3, axis=2)

img = Image.fromarray(np.asarray(np.uint8(np.clip(img, 0, 1)*255)))
img.save("example.jpg")
print('success')