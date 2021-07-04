import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dm_env as environment
from networks import *
import utils

LOCATION_KEYS = ("end", "control")
ORDERS = {
  "libmypaint": ["flag", "end", "control", "size", "pressure",
                 "red", "green", "blue"],
  "fluid": ["flag", "end", "control", "size", "speed",
            "red", "green", "blue", "alpha"],
}

def xy_grid(batch_size, width, height, cuda=False):
    x_grid = torch.linspace(-1, 1, steps=width)
    x_grid = x_grid.reshape((1, 1, width, 1))
    x_grid = x_grid.tile((batch_size, height, 1, 1))
    y_grid = torch.linspace(-1, 1, steps=height)
    y_grid = y_grid.reshape((1, height, 1, 1))
    y_grid = y_grid.tile((batch_size, 1, width, 1))
    if cuda:
        return x_grid.cuda(), y_grid.cuda()
    return x_grid, y_grid
    

def lstm_initial_state(batch_size, hidden_size):
    hiddens = torch.zeros((1, batch_size, hidden_size))
    cells = torch.zeros((1, batch_size, hidden_size))
    return (hiddens, cells)


class A2C(nn.Module):
    def __init__(self, action_spec, input_shape, grid_shape, action_order, cuda=False):
        super(A2C, self).__init__()
        self.conv0 = nn.Conv2d(5, 32, kernel_size=5, stride=1, padding=2)

        self._action_order = action_order
        self._action_spec = collections.OrderedDict(action_spec)
        

        order = [k for k in ORDERS[action_order] if k in action_spec]
        action_fcs = {
            k: nn.Linear(self._action_spec[k].maximum-self._action_spec[k].minimum+1, 16)
            for k in order
            if k not in LOCATION_KEYS
        }
        for k in LOCATION_KEYS:
            action_fcs[k] = nn.Linear(2, 16)
        self.action_fcs = nn.ModuleDict(action_fcs)

        self.action_mlp = build_mlp(16*len(self._action_spec), [64, 32, 32])
        self.noise_cond_mlp = build_mlp(10, [64, 32, 32])

        self._grid_height, self._grid_width = grid_shape
        input_height, input_width = input_shape
        enc_factor_h = input_height // 8
        enc_factor_w = input_width // 8
        dec_factor_h = self._grid_height // 4  # Height of feature after core is 4
        dec_factor_w = self._grid_width // 4  # Width of feature after core is 4
        self.conv_encoder = ConvEncoder(
            factor_h=enc_factor_h,
            factor_w=enc_factor_w,
            num_hiddens=32,
            num_residual_layers=8,
            num_residual_hiddens=32,
        )
        self._decoder_params = {
            "factor_h": dec_factor_h,
            "factor_w": dec_factor_w,
            "num_hiddens": 32,
            "num_residual_layers": 8,
            "num_residual_hiddens": 32,
            "num_output_channels": 1,
        }
        self.encoder_fc = nn.Linear(512, 256)

        self.lstm_hidden_size = 256
        self.lstm_core = nn.LSTM(input_size=256, batch_first=True, hidden_size=self.lstm_hidden_size)
        
        self.decoder_head = AutoregressiveHeads(
            z_dim=self.lstm_hidden_size,
            embed_dim=16,
            action_spec=self._action_spec,
            grid_height=self._grid_height,
            grid_width=self._grid_width,
            decoder_params=self._decoder_params,
            order=self._action_order,
            cuda=cuda,
        )

        self.value_fc = nn.Linear(self.lstm_hidden_size, 1)

    def forward(self, obs):
        pass

    def initial_state(self, batch_size=1):
        return utils.AgentState(
            lstm_state=lstm_initial_state(batch_size, self.lstm_hidden_size),
            prev_action=collections.OrderedDict([(spec, np.asarray([0 for _ in range(batch_size)], dtype=np.int32).squeeze()) for spec in self._action_spec])
            # prev_action=collections.OrderedDict([(spec, torch.zeros((batch_size, ))) for spec in self._action_spec])
            # prev_action={
            #     spec: torch.zeros((batch_size, 1))
            #     for spec in self._action_spec
            # }
        )

    def _encode_action_condition(self, action, mask, is_train=False):
        if is_train:
            mask = mask.values()
        else:
            mask = tuple(mask[k] for k in self._action_spec.keys())
        conds = []

        action = action.values()
        for i, (k, a, m) in enumerate(zip(self._action_spec.keys(), action, mask)):
            # some ugly shape adaptations to train in batch
            if is_train:
                a = np.array(a)
                a = np.expand_dims(a, axis=1) # (batch*time, 1)
                a = torch.FloatTensor(a)
                m = np.array(m)
                m = np.expand_dims(m, axis=1)
                m = torch.FloatTensor(m)
                a = a.cuda()
                m = m.cuda()
            else:
                a = np.expand_dims(a, axis=(0, 1)) # (1, 1)
                a = torch.FloatTensor(a)
            depth = self._action_spec[k].maximum - self._action_spec[k].minimum + 1
            embed = self.action_fcs[k]
            if k in LOCATION_KEYS:
                w = self._grid_width
                h = self._grid_height
                y = a // w
                x = a % w
                if not is_train:
                    y = torch.FloatTensor(y)
                    x = torch.FloatTensor(x)
                y = -1.0 + 2.0 * y / (h - 1)
                x = -1.0 + 2.0 * x / (w - 1)
                a_vec = torch.cat([y, x], dim=1)
            else:
                a_vec = F.one_hot(a.long(), depth)[:, 0, :].float()

            cond = embed(a_vec) * m # (batch*time, 16) * (batch*time, 1)
            conds.append(cond)
        cond = torch.cat(conds, dim=1)
        cond = self.action_mlp(cond)
        return cond

    def _torso(self, obs, prev_action, should_reset):
        # batch_size, x_h, x_w, _ = obs["canvas"].get_shape().as_list()
        x_h, x_w, c = list(obs["canvas"].shape)
        if c != 3:
            obs["canvas"] = np.repeat(obs["canvas"], 3, axis=2)
        batch_size = 1
        x_grid, y_grid = xy_grid(batch_size, x_h, x_w)

        # spatial_inputs = [obs['canvas']]
        spatial_inputs = [torch.tensor(obs['canvas'][None])]
        spatial_inputs += [x_grid, y_grid]
        data = torch.cat(spatial_inputs, dim=-1)
        data = data.permute(0, 3, 1, 2)
        h = self.conv0(data) # (batch, 32, 64, 64)

        cond = self._encode_action_condition(prev_action, obs['action_mask'])

        cond += self.noise_cond_mlp(obs["noise_sample"])
        cond = cond.view(batch_size, -1, 1, 1)

        h += cond
        h = F.relu(h)

        h = self.conv_encoder(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.encoder_fc(h))
        return h

    def _torso_for_train(self, observations, prev_actions, action_masks, noise_samples):
        # observations: (batch, time, 64, 64, 3)
        # prev_actions: dict, each (batch, time)
        # action_masks: dict, each (batch, time+1)
        # noise_samples: (batch, time, 10)

        batch_size, n_ts, x_h, x_w, n_c = observations.shape
        x_grid, y_grid = xy_grid(batch_size*n_ts, x_h, x_w, cuda=True)

        observations = observations.view(batch_size*n_ts, x_h, x_w, n_c)

        spatial_inputs = [observations, x_grid, y_grid]
        data = torch.cat(spatial_inputs, dim=-1)
        data = data.permute(0, 3, 1, 2)
        h = self.conv0(data) # (batch*time, 32, 64, 64)

        action_masks = copy.deepcopy(action_masks)
        for k in self._action_spec:
            prev_actions[k] = prev_actions[k].reshape((batch_size*n_ts))
            action_masks[k] = action_masks[k][:, :-1]
            action_masks[k] = action_masks[k].reshape((batch_size*n_ts))

        cond = self._encode_action_condition(prev_actions, action_masks, is_train=True)
        cond += self.noise_cond_mlp(noise_samples).view(batch_size*n_ts, -1)
        cond = cond.view(batch_size*n_ts, -1, 1, 1) # (batch*time, 32, 1, 1)

        h += cond
        h = F.relu(h)

        self.conv_encoder.cuda()
        h = self.conv_encoder(h)
        h = h.reshape(batch_size*n_ts, -1)
        h = F.relu(self.encoder_fc(h)) # (batch*time, 256)
        h = h.view(batch_size, n_ts, -1)
        return h

    def _head(self, core_output):
        logits, actions = self.decoder_head(core_output)

        baseline = self.value_fc(core_output).squeeze()

        return utils.AgentOutput(actions, logits, baseline)

    def PI(self, step_type, observation, prev_state):
        should_reset = step_type == environment.StepType.FIRST
        torso_output = self._torso(observation, prev_state.prev_action, should_reset) # (batch, 256)
        torso_output = torso_output[None]

        lstm_state = prev_state.lstm_state
        core_output, new_core_state = self.lstm_core(torso_output, lstm_state)

        core_output = core_output[0]

        agent_output = self._head(core_output)

        new_state = utils.AgentState(
            prev_action=agent_output.action,
            lstm_state=new_core_state)

        return agent_output, new_state

    def trajectory_forward(self, observations, prev_actions, action_masks, noise_samples):
        # torso_output: (batch, time, 256)
        torso_output = self._torso_for_train(observations, prev_actions, action_masks, noise_samples)

        core_output, _ = self.lstm_core(torso_output) # batch_first is True, (batch, time, 256)

        core_output = core_output.reshape(core_output.size(0)*core_output.size(1), -1)
        agent_output = self._head(core_output) # access by .actions, .logits, .baseline
        return agent_output

    def get_entropy(self, logits):  # (None, a_n)
        a0 = logits - torch.max(logits, 1, keepdims=True)[0]
        ea0 = torch.exp(a0)  # between 0 and 1
        z0 = torch.sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), 1)

    def choose_action(self, obs):
        pass

    def neg_log_loss(self, logits, actions, action_masks):
        # logits and labels are dicts with same shape: each (batch, time)
        all_neg_log_probs = []
        for key in logits:
            # logit = logits[key].view(-1, logits[key].size(-1)) # should be (batch*time, action_dim)
            logit = logits[key]
            action = torch.LongTensor(np.array(actions[key])).view(-1) # should be (batch*time)
            action = action.cuda()
            action_mask = torch.tensor(action_masks[key]).cuda()
            action_mask = action_mask[:, 1:].reshape(-1)
            neg_log_prob = nn.CrossEntropyLoss(reduction='none')(logit, action) * action_mask
            all_neg_log_probs.append(neg_log_prob) # should be (batch*time)
        all_neg_log_probs = torch.stack(all_neg_log_probs, dim=1).sum(dim=-1) # (batch*time)
        return all_neg_log_probs

    def entropy_loss(self, logits, action_masks):
        all_entropies = []
        for key in logits:
            # logit = logits[key].view(-1, logits[key].size(-1))
            logit = logits[key]
            action_mask = torch.tensor(action_masks[key]).cuda()
            action_mask = action_mask[:, 1:].reshape(-1)
            entropies = self.get_entropy(logit) * action_mask # (batch*time)
            all_entropies.append(entropies)
        all_entropies = torch.stack(all_entropies, dim=1).sum(dim=-1) # (batch*time)
        all_entropies = all_entropies.mean()
        return all_entropies

    def optimize(self, 
        image_batch, 
        prev_action_batch, 
        action_batch, 
        reward_batch, 
        adv_batch, 
        action_masks, 
        noise_samples,
        entropy_weight=0.04,
        value_weight=1.0,
    ):
        # image_batch: (batch, time, 64, 64, 3)
        # prev_action_batch: dict, each (batch, time)
        # action_batch: dict, each (batch, time)
        # reward_batch: (batch, time)
        # adv_batch: (batch, time)
        # action_masks: dict, each (batch, time+1)
        # noise_samples: (batch, time, 10)

        agent_output = self.trajectory_forward(image_batch, prev_action_batch, action_masks, noise_samples)
        logits = agent_output.policy_logits # a dict
        neg_log_pi = self.neg_log_loss(logits, action_batch, action_masks) # (batch*time) tensor
        neg_log_pi_mean = neg_log_pi.mean()
        if neg_log_pi_mean > 20:
            # skip this batch due to numerical issue caused by off-policy data
            return None

        adv_batch = adv_batch.view(-1)
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
        policy_loss = (neg_log_pi * adv_batch).mean(dim=0) # scaler

        entropy = self.entropy_loss(logits, action_masks) # scaler

        values = agent_output.baseline # (batch * time)
        reward_batch = reward_batch.view(-1)
        value_loss = torch.square(reward_batch - values).mean() # scaler

        total_loss = policy_loss + value_weight * value_loss - entropy_weight * entropy

        return total_loss, policy_loss, value_loss, entropy, neg_log_pi_mean

    def get_weights(self):
        return utils.dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)
