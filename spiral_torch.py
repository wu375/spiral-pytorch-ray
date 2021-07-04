import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import ray

import collections
from storage import SharedStorage
from painter import Painter
from learners import DiscLearner, PolicyLearner

from networks import Discriminator
from config import SpiralConfig
from policies import A2C

def init_weights(checkpoint, cfg):
    a2c = A2C(cfg.action_spec, input_shape=cfg.input_shape, grid_shape=cfg.grid_shape, action_order="libmypaint")
    a2c_weights = a2c.get_weights()
    checkpoint['a2c_weights'] = a2c_weights
    del a2c
    d = Discriminator()
    d_weights = d.get_weights()
    checkpoint['d_weights'] = d_weights
    del d
    return checkpoint

def train_loop(storage, config):

    writer = SummaryWriter(config.results_path)

    for i in range(20):
        print()

    counter = 0
    keys = [
        "terminate",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "a2c_training_steps",
        "d_loss",
        "d_fake",
        "d_real",
        "d_training_steps",
        "num_played_games",
        "reward",
        "value",
        "render_sample",
        "real_sample",
        "fake_sample",
        "neg_log",
        "adv",
        "grad_norm",
        "n_batches_skipped",
    ]
    info = ray.get(storage.get_info.remote(keys))
    try:
        while info["a2c_training_steps"] < config.training_steps and not info["terminate"]:
            info = ray.get(storage.get_info.remote(keys))
            if info['a2c_training_steps'] > 0 and info['a2c_training_steps'] % config.log_interval == 0:
                writer.add_scalar("loss", info["loss"], counter)
                writer.add_scalar("policy_loss", info["policy_loss"], counter)
                writer.add_scalar("value_loss", info["value_loss"], counter)
                writer.add_scalar("entropy_loss", info["entropy_loss"], counter)
                writer.add_scalar("d_loss", info["d_loss"], counter)
                writer.add_scalar("d_fake", info["d_fake"], counter)
                writer.add_scalar("d_real", info["d_real"], counter)
                writer.add_scalar("a2c_training_steps", info["a2c_training_steps"], counter)
                writer.add_scalar("d_training_steps", info["d_training_steps"], counter)
                writer.add_scalar("num_played_games", info["num_played_games"], counter)
                writer.add_scalar("reward", info["reward"], counter)
                writer.add_scalar("value", info["value"], counter)
                writer.add_scalar("neg_log", info["neg_log"], counter)
                writer.add_scalar("adv", info["adv"], counter)
                writer.add_scalar("grad_norm", info["grad_norm"], counter)
                writer.add_scalar("n_batches_skipped", info["n_batches_skipped"], counter)

            if info['a2c_training_steps'] > 0 and info['a2c_training_steps'] % config.log_draw_interval == 0:
                if info['render_sample'] is not None:
                    writer.add_image("sample", info["render_sample"], counter, dataformats='HWC')
                if info['real_sample'] is not None:
                    writer.add_image("real sample", info["real_sample"], counter, dataformats='HWC')
                if info['fake_sample'] is not None:
                    writer.add_image("fake sample", info["fake_sample"], counter, dataformats='HWC')
                
            print(
                f'Training step: {info["a2c_training_steps"]}/{config.training_steps}. Discriminator step: {info["d_training_steps"]}. Num of paints: {info["num_played_games"]}.',
                end="\r",
            )
            counter += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    ray.get(storage.save_checkpoint.remote())
    storage.set_info.remote("terminate", True)

def debug():
    import spiral.environments.libmypaint as libmypaint
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
    action_spec = env.action_spec()
    a2c = A2C(action_spec, input_shape=(64, 64), grid_shape=(32, 32), action_order="libmypaint")
    a2c.to('cuda')
    batch_size = 16
    n_time = 19

    image_batch = torch.rand(batch_size, n_time, 64, 64, 3).to('cuda')
    prev_action_batch = collections.OrderedDict([(spec, np.ones((batch_size, n_time))) for spec in a2c._action_spec])
    action_batch = collections.OrderedDict([(spec, np.ones((batch_size, n_time))) for spec in a2c._action_spec])
    reward_batch = torch.rand(batch_size, n_time).to('cuda')
    adv_batch = torch.rand(batch_size, n_time).to('cuda')
    action_masks = collections.OrderedDict([(spec, np.ones((batch_size, n_time+1))) for spec in a2c._action_spec])
    noise_samples = torch.rand(batch_size, n_time, 10).to('cuda')

    image_batch = torch.cuda.FloatTensor(image_batch)
    reward_batch = torch.cuda.FloatTensor(reward_batch)
    adv_batch = torch.cuda.FloatTensor(adv_batch)
    noise_samples = torch.cuda.FloatTensor(noise_samples)

    total_loss, policy_loss, value_loss, entropy_loss, neg_log = a2c.optimize(image_batch, prev_action_batch, action_batch, reward_batch, adv_batch, action_masks, noise_samples)

    print('success')

def train():
    ray.init(num_gpus=2)

    cfg = SpiralConfig()


    if cfg.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_path)
        checkpoint['terminate'] = False
    else:
        checkpoint = {
            "a2c_weights": None,
            "d_weights": None,
            "optimizer_state": None,
            "a2c_training_steps": 0,
            "d_training_steps": 0,
            "lr": 0,
            "num_played_games": 0,
            "terminate": False,
            "loss":None,
            "policy_loss":None,
            "value_loss":None,
            "entropy_loss":None,
            "d_loss":None,
            "d_fake":None,
            "d_real":None,
            "reward":None,
            "value":None,
            "render_sample":None,
            "real_sample":None,
            "fake_sample":None,
            "neg_log":None,
            "adv":None,
            "grad_norm":None,
            "d_std":None,
            "d_mean":None,
            "n_batches_skipped":0,
        }
        checkpoint = init_weights(checkpoint, cfg)

    if cfg.reward_mode != 'wgan':
        checkpoint['d_loss'] = 0
        checkpoint['d_fake'] = 0
        checkpoint['d_real'] = 0

    storage = SharedStorage.remote(checkpoint, cfg)

    painters = [Painter.remote(cfg) for i in range(cfg.n_painters)]
    [painter.play.remote(storage=storage) for painter in painters]


    if cfg.reward_mode == 'wgan':
        d_learner = DiscLearner.remote(cfg, checkpoint)
        d_learner.learn.remote(storage=storage)

    policy_learner = PolicyLearner.remote(cfg, checkpoint)
    policy_learner.learn.remote(storage=storage)

    train_loop(storage, cfg)

    ray.shutdown()


if __name__ == '__main__':
    train()
    # debug()
