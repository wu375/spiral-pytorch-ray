import time
import copy
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from networks import Discriminator
from real_image_loader import get_real_image_loader
from policies import A2C
from torchvision.utils import save_image

@ray.remote(num_gpus=1)
class DiscLearner:
    def __init__(self, config, checkpoint=None):
        print('discriminator using gpu: ', ray.get_gpu_ids())
        self.config = config
        self.batch_size = config.batch_size
        self.discriminator = Discriminator()
        if checkpoint is not None:
            self.discriminator.set_weights(checkpoint['d_weights'])
        self.discriminator.cuda()
        
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.d_lr, betas=(0.5, 0.999))
        if checkpoint is not None:
            self.discriminator_step = checkpoint['d_training_steps']
        else:
            self.discriminator_step = 0
        
        self.training_steps_ratio = config.training_steps_ratio

        self.real_image_loader = get_real_image_loader(dataset=config.dataset, batch_size=config.batch_size)

    def _compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates).squeeze().unsqueeze(1)
        fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _wgan_update(self, fake_images):
        real_images = next(self.real_image_loader)
        real_sample = real_images[0].numpy()
        real_images = real_images.cuda().float()
        real_images = real_images.permute(0, 3, 1, 2)

        self.optimizer.zero_grad()
        gradient_penalty = self._compute_gradient_penalty(self.discriminator, real_images.data, fake_images.data)
        fake_scores = self.discriminator(fake_images)
        real_scores = self.discriminator(real_images)
        d_fake = fake_scores.mean()
        d_real = real_scores.mean()
        loss = d_fake - d_real + 10*gradient_penalty
        
        loss.backward()
        self.optimizer.step()
        # for p in self.discriminator.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        scores = torch.cat([fake_scores, real_scores], dim=0).detach()
        d_std = scores.std()
        d_mean = scores.mean()

        return loss, d_fake, d_real, d_std, d_mean, real_sample

    def learn(self, storage):
        while not ray.get(storage.is_buffer_ready.remote()):
            time.sleep(0.1)

        batch_future = storage.get_final_render_batch.remote()

        while not ray.get(storage.get_info.remote("terminate")):
            # image_batch: (batch, 64, 64, 3)
            image_batch = ray.get(batch_future)
            batch_future = storage.get_final_render_batch.remote()

            image_batch = np.array(image_batch)
            fake_sample = image_batch[0]
            image_batch = torch.cuda.FloatTensor(image_batch)
            image_batch = image_batch.permute(0, 3, 1, 2)

            loss, d_fake, d_real, d_std, d_mean, real_sample = self._wgan_update(image_batch)


            self.discriminator_step += 1

            if self.discriminator_step % self.config.weight_copy_interval == 0:
                info = {
                    "d_weights": copy.deepcopy(self.discriminator.get_weights()),
                    "d_training_steps": self.discriminator_step,
                }
                if self.discriminator_step < 10000: # freeze std and mean when stable
                    info["d_std"] = d_std.cpu().numpy()
                    info["d_mean"] = d_mean.cpu().numpy()
                storage.set_info.remote(info)
            if self.discriminator_step % self.config.log_interval == 0:
                storage.set_info.remote(
                    {
                        "d_loss": loss.detach().cpu().numpy(),
                        "d_fake": d_fake.detach().cpu().numpy(),
                        "d_real": d_real.detach().cpu().numpy(),
                    }
                )
            if self.discriminator_step % self.config.log_draw_interval == 0:
                storage.set_info.remote(
                    {
                        "real_sample": real_sample,
                        "fake_sample": fake_sample,
                    }
                )

            if self.training_steps_ratio is not None:
                while self.discriminator_step // ray.get(storage.get_info.remote("a2c_training_steps")) > self.training_steps_ratio:
                    time.sleep(0.2)

@ray.remote(num_gpus=1)
class PolicyLearner:
    def __init__(self, config, checkpoint=None):
        print('policy using gpu: ', ray.get_gpu_ids())
        self.batch_size = config.batch_size
        self.config = config

        if checkpoint is not None:
            self.training_step = checkpoint['a2c_training_steps']
        else:
            self.training_step = 0

        self.a2c = A2C(config.action_spec, input_shape=(64, 64), grid_shape=(32, 32), action_order="libmypaint", cuda=True)
        if checkpoint is not None:
            self.a2c.set_weights(checkpoint['a2c_weights'])
        self.a2c.cuda()

        self.optimizer = torch.optim.Adam(
            self.a2c.parameters(),
            lr=self.config.a2c_lr,
        )

        # i = 0
        # for p in self.a2c.parameters():
        #     i += 1
        # print('policy training with ' + str(i) + ' parameters')

        if config.reward_mode == 'l2':
            self.real_image_loader = get_real_image_loader(dataset=config.dataset, batch_size=config.batch_size)
        else:
            self.discriminator = Discriminator()
            self.discriminator.eval()

        self.reward_mode = config.reward_mode

        self.n_batches_skipped = 0

    def _get_rewards(self, final_renders, storage, gamma=0.99):
        # final_renders: (batch, 64, 64, 3)
        if self.reward_mode == 'l2':
            real_images = next(self.real_image_loader).numpy()
            reward = -np.sqrt(np.sum(np.square(final_renders - real_images), axis=(1,2,3)))
            reward = reward[:, np.newaxis] # (batch, 1)
            reward = (reward + 50)/50
        else:

            info = ray.get(storage.get_info.remote(['d_std', 'd_mean', 'd_weights']))
            self.discriminator.set_weights(info['d_weights'])

            with torch.no_grad():
                final_renders_tensor = torch.FloatTensor(final_renders)
                final_renders_tensor = final_renders_tensor.permute(0, 3, 1, 2)
                reward = self.discriminator(final_renders_tensor).squeeze() # (batch,)
                reward = reward.numpy()[:, np.newaxis] # (batch, 1)
                reward = (reward - info['d_mean']) / info['d_std']
                # reward = reward / 150

        if self.config.dataset == 'mnist':
            sparse_reward = np.sum(final_renders, axis=(1,2,3))/12288
            sparse_reward = np.where(sparse_reward > 0.95, -3*np.ones_like(sparse_reward), np.zeros_like(sparse_reward))
            sparse_reward = sparse_reward[:, np.newaxis]
            reward = reward + sparse_reward
            # reward -= 1*sparse_reward*np.abs(reward)
        return reward

    def _get_adv(self, R, V):
        return R - V

    def learn(self, storage):
        while ray.get(storage.get_info.remote("d_training_steps")) < 1:
            time.sleep(0.1)

        while not ray.get(storage.is_queue_ready.remote()):
            time.sleep(0.1)
        batch_future = storage.get_trajectory_batch.remote()
        while self.training_step < self.config.training_steps and not ray.get(storage.get_info.remote("terminate")):
            # action_batch should be a dict, each value is (batch, time, size[i]), where i comes from LOCATION_KEYS
            image_batch, action_batch, prev_action_batch, action_masks, value_batch, noise_samples, final_renders = ray.get(batch_future)
            
            image_batch = np.array(image_batch)
            # final_renders = image_batch[:, -1, :, :, :] # (batch, 64, 64, 3)
            final_renders = np.array(final_renders)
            reward_batch = self._get_rewards(final_renders, storage) # (batch,)
            reward_mean = np.mean(reward_batch)
            
            reward_batch = np.repeat(reward_batch, image_batch.shape[1], axis=1) # (batch, time)

            value_mean = np.mean(value_batch)
            adv_batch = self._get_adv(reward_batch, value_batch) # (batch, time)

            noise_samples = noise_samples[:, np.newaxis, :] # (batch, 1, 10)
            noise_samples = np.repeat(noise_samples, image_batch.shape[1], axis=1) # (batch, time, 10)

            image_batch = torch.cuda.FloatTensor(image_batch)
            reward_batch = torch.cuda.FloatTensor(reward_batch)
            adv_batch = torch.cuda.FloatTensor(adv_batch)
            noise_samples = torch.cuda.FloatTensor(noise_samples)

            results = self.a2c.optimize(
                image_batch, 
                prev_action_batch, 
                action_batch, 
                reward_batch, 
                adv_batch, 
                action_masks, 
                noise_samples,
                entropy_weight=self.config.entropy_weight,
                value_weight=self.config.value_weight,
            )

            if results is None: # numerical issue
                self.n_batches_skipped += 1
            else:
                total_loss, policy_loss, value_loss, entropy_loss, neg_log = results

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.a2c.parameters(), 0.5)
                self.optimizer.step()

                self.training_step += 1

                if self.training_step % self.config.weight_copy_interval == 0:
                    storage.set_info.remote(
                        {
                            "a2c_weights": copy.deepcopy(self.a2c.get_weights()),
                            "a2c_training_steps": self.training_step,
                        }
                    )

                if self.training_step % self.config.log_interval == 0:
                    entropy_sample = entropy_loss.detach().cpu().numpy()
                    if entropy_sample < 3.0: # stop training if entropy collapsed
                        storage.set_info.remote("terminate", True)
                    else:
                        grad_norm = 0
                        for p in self.a2c.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                grad_norm += param_norm.item() ** 2
                        grad_norm = grad_norm ** 0.5

                        storage.set_info.remote(
                            {
                                "loss": total_loss.detach().cpu().numpy(),
                                "policy_loss": policy_loss.detach().cpu().numpy(),
                                "value_loss": value_loss.detach().cpu().numpy(),
                                "entropy_loss": entropy_sample,
                                "reward": reward_mean,
                                "value": value_mean,
                                "neg_log": neg_log.detach().cpu().numpy(),
                                "adv": adv_batch.mean().cpu().numpy(),
                                "grad_norm": grad_norm,
                                "n_batches_skipped": self.n_batches_skipped,
                            }
                        )

                if self.training_step % self.config.log_draw_interval == 0:
                    storage.set_info.remote(
                        {
                            "render_sample": final_renders[0],
                        }
                    )
                    save_image(torch.tensor(final_renders[:25]).permute(0,3,1,2), (self.config.results_path+"%d.png") % self.training_step, nrow=5, normalize=True)

                if self.training_step % self.config.checkpoint_interval == 0:
                    storage.save_checkpoint.remote(savename=str(self.training_step))

                # if self.training_step < 2000:
                #     time.sleep(1)
                # if self.training_step < 5000:
                #     time.sleep(0.5)

            while not ray.get(storage.is_queue_ready.remote()):
                time.sleep(0.1)
            batch_future = storage.get_trajectory_batch.remote()


if __name__ == '__main__':
    import config
    config = config.SpiralConfig()
    disc_learner = DiscLearner(config)