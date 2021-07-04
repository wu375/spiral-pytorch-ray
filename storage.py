import os
import ray
import collections
import copy
from random import sample
import numpy as np
import torch

@ray.remote
class SharedStorage:
    def __init__(self, checkpoint, config):
        self.config = config
        self._action_spec = config.action_spec
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.total_size = self.buffer_size * self.batch_size
        self.trajectories = [] # queue for policy learner
        self.images_rb = [None for _ in range(self.total_size)]
        self.pointer = 0
        self.image_pointer = 0
        self.images_rb_len = 0

        self.current_checkpoint = copy.deepcopy(checkpoint)

    def get_tragectories(self):
        return self.trajectories

    def get_images(self):
        return np.array(self.images_rb)

    def save_trajectory(self, trajectory, final_render):
        self.trajectories.append(trajectory)
        # self.noise_samples[self.pointer] = noise_sample

        self.images_rb[self.pointer] = final_render
        # if not self.buffer_ready and self.pointer == self.batch_size-1:
        #     self.buffer_ready = True
        self.pointer = (self.pointer + 1) % self.total_size

    def is_buffer_ready(self):
        return self.pointer >= self.batch_size or self.images_rb[-1] is not None

    def is_queue_ready(self):
        return len(self.trajectories) >= self.batch_size

    def increment_n_games(self):
        self.set_info('num_played_games', self.current_checkpoint['num_played_games']+1)

    def get_trajectory_batch(self):
        image_batch = []
        value_batch = []
        action_batch = collections.OrderedDict([(spec, []) for spec in self._action_spec])
        prev_action_batch = collections.OrderedDict([(spec, []) for spec in self._action_spec])
        action_mask_batch = collections.OrderedDict([(spec, []) for spec in self._action_spec])

        traj_batch = self.trajectories[:self.batch_size]
        self.trajectories = self.trajectories[self.batch_size:]

        noise_batch = [t['noise_sample'] for t in traj_batch]
        render_batch = [t['final_render'] for t in traj_batch]

        for t in traj_batch:
            image_batch.append(np.array(t['images']))
            for key in self._action_spec:
                action_batch[key].append(t['actions'][key])
                prev_action_batch[key].append(t['prev_actions'][key])
                action_mask_batch[key].append(t['action_masks'][key])
            # action_batch.append(t['actions'])
            value_batch.append(t['values'])

        # lastly stack actions
        for key in self._action_spec:
            action_batch[key] = np.stack(action_batch[key])
            prev_action_batch[key] = np.stack(prev_action_batch[key])
            action_mask_batch[key] = np.stack(action_mask_batch[key])

        # image_batch: (batch, time, 64, 64, 3)
        # each action should be (batch, time)
        # value: (batch, time)
        return np.array(image_batch), action_batch,\
            prev_action_batch, action_mask_batch,\
            np.array(value_batch), np.array(noise_batch), np.array(render_batch)

    def get_final_render_batch(self):
        if self.images_rb[-1] is None: # buffer not full
            sampled = range(self.pointer)
        else:
            sampled = range(self.total_size)
        sampled = sample(sampled, self.batch_size)
        sampled = [self.images_rb[i] for i in sampled]
        return np.array(sampled)


    def save_checkpoint(self, savename=None):
        if not savename:
            savename = "model"
        savename = os.path.join(self.config.results_path, savename+'.checkpoint')

        torch.save(self.current_checkpoint, savename)

    def get_checkpoint(self):
        return copy.deepcopy(self.curren_ctheckpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint.get(keys, 0)
        elif isinstance(keys, list):
            return {key: self.current_checkpoint.get(key, 0) for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        # if keys in ('d_std', 'd_mean') and self.current_checkpoint.get(keys, None) is not None:
        #     self.current_checkpoint[keys] = self.current_checkpoint[keys] * 0.999 + self.current_checkpoint[keys] * 0.001
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

if __name__ == '__main__':
    import config
    from spiral_torch import A2C
    import spiral.environments.libmypaint as libmypaint
    config = config.SpiralConfig()
    checkpoint = {
        "agent_weights": None,
        "optimizer_state": None,
        "training_step": 0,
        "lr": 0,
        "num_played_games": 0,
        "num_played_steps": 0,
        "terminate": False,
    }
    storage = SharedStorage(checkpoint, config)
    print(storage.get_info('training_step'))
    print(storage.get_info('terminate'))
    print(storage.get_info('agent_weights'))


    BRUSHES_PATH = 'third_party/mypaint-brushes-1.3.0/'
    env = libmypaint.LibMyPaint(
        episode_length=20,
        canvas_width=64,
        grid_width=32,
        brush_type="classic/dry_brush",
        brush_sizes=[1, 2, 4, 8, 12, 24],
        use_color=True,
        use_pressure=True,
        use_alpha=False,
        background="white",
        brushes_basedir=BRUSHES_PATH)
    # observations keys are 'canvas', 'episode_step', 'episode_length', 'action_mask'

    for _ in range(7):
        time_step = env.reset()
        action_spec = env.action_spec()
        a2c = A2C(action_spec, input_shape=(64, 64), grid_shape=(32, 32), action_order="libmypaint")
        a2c.eval()
        state = a2c.initial_state(1)
        trajectory = {
            'images': [],
            'prev_actions': collections.OrderedDict([(spec, [state.prev_action[spec]]) for spec in action_spec]),
            'actions': collections.OrderedDict([(spec, []) for spec in action_spec]),
            'values': [],
        }
        noise_sample = torch.randn(1, 10)
        with torch.no_grad():
            for t in range(19):
                time_step.observation["noise_sample"] = noise_sample
                agent_out, state = a2c.PI(time_step.step_type, time_step.observation, state)
                action = agent_out.action
                trajectory['images'].append(time_step.observation["canvas"])
                for key in trajectory['actions']:
                    trajectory['actions'][key].append(action[key])
                    trajectory['prev_actions'][key].append(state.prev_action[key])
                trajectory['values'].append(agent_out.baseline.numpy())
                time_step = env.step(action)

        for key in trajectory['prev_actions']:
            trajectory['prev_actions'][key] = trajectory['prev_actions'][key][:-1] # each action is (time,)
        storage.save_trajectory(trajectory, time_step.observation["canvas"])
        print('is batch ready: ', storage.buffer_ready)
        print(storage.pointer)


    image_batch, action_batch, prev_action_batch, value_batch = storage.get_trajectory_batch()
    print('images: ', image_batch.shape)
    print('actions: ')
    for key in action_batch:
        print(key)
        print('action batch shape: ', action_batch[key].shape)
        print('prev action batch shape: ', prev_action_batch[key].shape)
    print('values: ', value_batch.shape)

    image_batch = torch.tensor(image_batch)
    total_loss, policy_loss, value_loss, entropy_loss = a2c.optimize(image_batch, action_batch, reward_batch, adv_batch)

    print('success')