import os
import numpy
import torch

class SpiralConfig:
    def __init__(self):
        self.seed = 0
        self.dataset = 'mnist' # 'celeba' or 'mnist'
        self.training_steps = 1000000
        self.n_paint_steps = 10 # 19
        self.input_shape = (64, 64)
        self.grid_shape = (32, 32)
        self.buffer_size = 20
        self.batch_size = 64
        self.n_painters = self.batch_size
        self.optimizer = "adam"
        self.checkpoint_interval = 2000
        self.weight_copy_interval = 1
        self.log_interval = 20
        self.log_draw_interval = 50
        self.d_lr = 0.0001
        self.a2c_lr = 0.00005
        self.entropy_weight = 0.04
        self.value_weight = 1.0
        self.training_steps_ratio = 5 # control d_steps/policy_steps. Only support value < ~10. Set to None if not constrained
        self.reward_mode = 'wgan' # 'l2' or 'wgan'
        i = 0
        while os.path.isdir('train_log/run'+str(i)):
            i += 1
        log_dir = 'train_log/run'+str(i)+'/'
        self.results_path = log_dir

        self.checkpoint_path = None # "train_log/run180/model_5000.checkpoint"

        self.libmypaint_params = {
            "episode_length": 20,
            "canvas_width": 64,
            "grid_width": 32,
            "brush_type": "classic/dry_brush",
            "brush_sizes": [1, 2, 4, 8], # [1, 2, 4, 8, 12, 24],
            "use_color": False,
            "use_pressure": True,
            "use_alpha": False,
            "background": "white",
            "brushes_basedir": 'third_party/mypaint-brushes-1.3.0/',
        }
        self.action_spec = self._get_action_spec()


    def _get_action_spec(self):
        import spiral.environments.libmypaint as libmypaint
        BRUSHES_PATH = 'third_party/mypaint-brushes-1.3.0/'
        env = libmypaint.LibMyPaint(**self.libmypaint_params)
        action_spec = env.action_spec()
        del env
        return action_spec