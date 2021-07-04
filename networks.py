import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

LOCATION_KEYS = ("end", "control")
ORDERS = {
        "libmypaint": ["flag", "end", "control", "size", "pressure",
        "red", "green", "blue"],
        "fluid": ["flag", "end", "control", "size", "speed",
        "red", "green", "blue", "alpha"],
    }

def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, in_out_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_out_channels, hidden_channels, kernel_size=3, stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, in_out_channels, kernel_size=1, stride=(1, 1))

    def forward(self, x):
        h = x
        h = F.relu(h)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h += x
        h = F.relu(h)
        return h

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_layers, num_residual_hiddens):
        super().__init__()
        self.res_stack = nn.ModuleList([ResidualBlock(num_hiddens, num_residual_hiddens) for _ in range(num_layers)])

    def forward(self, x):
        for block in self.res_stack:
            x = block(x)
        return x

class ConvEncoder(nn.Module):
    def __init__(self,
            factor_h,
            factor_w,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            initializers=None,
            data_format="NHWC",
            name="conv_encoder",
        ):

        super(ConvEncoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._initializers = initializers
        self._data_format = data_format

        # Note that implicitly the network uses conv strides of 2.
        # input height / output height == factor_h.
        self._num_steps_h = factor_h.bit_length() - 1
        # input width / output width == factor_w.
        self._num_steps_w = factor_w.bit_length() - 1
        num_steps = max(self._num_steps_h, self._num_steps_w)
        if factor_h & (factor_h - 1) != 0:
            raise ValueError("`factor_h` must be a power of 2. It is %d" % factor_h)
        if factor_w & (factor_w - 1) != 0:
            raise ValueError("`factor_w` must be a power of 2. It is %d" % factor_w)
        self._num_steps = num_steps

        ds_stack = []
        for i in range(self._num_steps):
            stride = (2 if i < self._num_steps_h else 1,
                2 if i < self._num_steps_w else 1)
            h = nn.Conv2d(
                in_channels=self._num_hiddens,
                out_channels=self._num_hiddens,
                kernel_size=4,
                stride=stride,
            )
            ds_stack.append(h)
            ds_stack.append(nn.ReLU())
        self.ds_stack = nn.Sequential(*ds_stack)

        self.conv3x3 = nn.Conv2d(
            in_channels=self._num_hiddens,
            out_channels=self._num_hiddens,
            kernel_size=3,
            stride=(1, 1),
        )

        self.res_stack = ResidualStack(  # pylint: disable=not-callable
            num_hiddens=self._num_hiddens,
            num_layers=self._num_residual_layers,
            num_residual_hiddens=self._num_residual_hiddens,
        )

    def forward(self, x):
        x = self.ds_stack(x)
        x = self.conv3x3(x)
        x = self.res_stack(x)
        return x

class AutoregressiveHeads(nn.Module):

    ORDERS = {
        "libmypaint": ["flag", "end", "control", "size", "pressure",
                     "red", "green", "blue"],
        "fluid": ["flag", "end", "control", "size", "speed",
                "red", "green", "blue", "alpha"],
    }

    def __init__(self,
            z_dim,
            embed_dim,
            action_spec,
            decoder_params,
            order,
            grid_height,
            grid_width,
            cuda=False,
        ):
        super(AutoregressiveHeads, self).__init__()

        self._z_dim = z_dim
        self._action_spec = action_spec
        self._grid_height = grid_height
        self._grid_width = grid_width

        self.cuda = cuda
        # Filter the order of actions according to the actual action specification.
        order = self.ORDERS[order]
        self._order = [k for k in order if k in action_spec]

        _action_embeds = collections.OrderedDict([
            (k, nn.Linear(action_spec[k].maximum-action_spec[k].minimum+1, embed_dim))
            if k not in LOCATION_KEYS else
            (k, nn.Linear(2, embed_dim))
            for k in action_spec
        ])
        self._action_embeds = nn.ModuleDict(_action_embeds)

        _action_heads = []
        for k in action_spec:
            v = action_spec[k]
            if k in LOCATION_KEYS:
                action_head = ConvVectorDecoder(**decoder_params)
            else:
                output_size = v.maximum - v.minimum + 1
                action_head = nn.Linear(self._z_dim, output_size)
            _action_heads.append((k, action_head))
        _action_heads = collections.OrderedDict(_action_heads)
        self._action_heads = nn.ModuleDict(_action_heads)

        _residual_mlps = {}
        for k in action_spec:
            v = action_spec[k]
            _residual_mlps[k] = build_mlp(embed_dim+self._z_dim, [16, 32, self._z_dim])
        self._residual_mlps = nn.ModuleDict(_residual_mlps)


    def forward(self, z):
        logits = {}
        action = {}
        for k in self._order:
            logits[k] = self._action_heads[k](z) # (batch, action_dim)
            a = torch.multinomial(nn.Softmax(dim=-1)(logits[k]), 1).float() # (batch, 1)
            # a = torch.argmax(logits[k], dim=1, keepdim=True).float()
            action[k] = a.squeeze().cpu().numpy().astype(np.int32) # (batch,)
            
            depth = self._action_spec[k].maximum - self._action_spec[k].minimum + 1
            if self.cuda:
                a = a.cuda()
            if k in LOCATION_KEYS:
                w = self._grid_width
                h = self._grid_height
                a = a[:, 0] # only sample one action
                y = a // w
                x = a % w
                y = -1.0 + 2.0 * y / (h - 1)
                x = -1.0 + 2.0 * x / (w - 1)
                a_vec = torch.stack([y, x], dim=1)
            else:
                a_vec = F.one_hot(a.long(), depth)[:, 0, :].float() # (batch, 1, action_dim) => (batch, action_dim)
            a_embed = self._action_embeds[k](a_vec)
            residual = self._residual_mlps[k](torch.cat([z, a_embed], dim=1))
            z = F.relu(z + residual)

        action = collections.OrderedDict([(k, action[k]) for k in self._action_spec]) # each (batch,)
        logits = collections.OrderedDict([(k, logits[k]) for k in self._action_spec]) # each (batch, action_dim)

        return logits, action

class ConvVectorDecoder(nn.Module):
    def __init__(self,
            factor_h,
            factor_w,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            num_output_channels=3,):
        super(ConvVectorDecoder, self).__init__()

        self.conv_decoder = ConvDecoder(
            factor_h,
            factor_w,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            num_output_channels,)

    def forward(self, z):
        z = z.view(z.size(0),-1, 4, 4)
        z = self.conv_decoder(z)
        z = z.view(z.size(0), -1)
        return z

class ConvDecoder(nn.Module):

    def __init__(self,
            factor_h,
            factor_w,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            num_output_channels=3,
        ):

        super(ConvDecoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._num_output_channels = num_output_channels

        self._num_steps_h = factor_h.bit_length() - 1
        self._num_steps_w = factor_w.bit_length() - 1

        num_steps = max(self._num_steps_h, self._num_steps_w)
        if factor_h & (factor_h - 1) != 0:
            raise ValueError("`factor_h` must be a power of 2. It is %d" % factor_h)
        if factor_w & (factor_w - 1) != 0:
            raise ValueError("`factor_w` must be a power of 2. It is %d" % factor_w)
        self._num_steps = num_steps

        self.in_conv = nn.Conv2d(
            in_channels=16, # 256/(4*4)
            out_channels=self._num_hiddens,
            kernel_size=3,
            stride=(1, 1),
        )

        self.res_stack = ResidualStack(
            num_hiddens=self._num_hiddens,
            num_layers=self._num_residual_layers,
            num_residual_hiddens=self._num_residual_hiddens,
        )

        us_stack = []
        for i in range(self._num_steps):
            # Does reverse striding -- puts stride-2s after stride-1s.
            stride = (2 if (self._num_steps - 1 - i) < self._num_steps_h else 1,
                2 if (self._num_steps - 1 - i) < self._num_steps_w else 1)
            h = nn.ConvTranspose2d(
                in_channels=self._num_hiddens,
                out_channels=self._num_hiddens,
                kernel_size=(4, 4),
                stride=stride,
            )
            us_stack.append(h)
            us_stack.append(nn.ReLU())
        self.us_stack = nn.Sequential(*us_stack)

        self.out_conv = nn.Conv2d(
            in_channels=self._num_hiddens,
            out_channels=self._num_output_channels,
            kernel_size=3,
            stride=(1, 1),
            padding=2,
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_stack(x)
        x = self.us_stack(x)
        x = self.out_conv(x)
        return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Filters [256, 512, 1024]
#         # Input_dim = channels (Cx64x64)
#         # Output_dim = 1
#         channels = 3
#         self.main_module = nn.Sequential(
#             # Image (Cx32x32)
#             nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=256),
#             nn.LeakyReLU(0.2, inplace=True),

#             # State (256x16x16)
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=512),
#             nn.LeakyReLU(0.2, inplace=True),

#             # State (512x8x8)
#             nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(num_features=1024),
#             nn.LeakyReLU(0.2, inplace=True)

#             # # State (512x8x8)
#             # nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, padding=1),
#             # nn.BatchNorm2d(num_features=1024),
#             # nn.LeakyReLU(0.2, inplace=True)
#         )

#         self.output = nn.Sequential(
#             # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
#             nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=0))

#         self.apply(weights_init_normal)

#     def forward(self, x):
#         x = self.main_module(x)
#         x = self.output(x)
#         return x

#     def feature_extraction(self, x):
#         # Use discriminator for feature extraction then flatten to vector of 16384
#         x = self.main_module(x)
#         return x.view(-1, 1024*4*4)

#     def save(self, path):
#         torch.save(self.state_dict(), path)

#     def load(self, path):
#         self.load_state_dict(torch.load(path))

#     def get_weights(self):
#         return utils.dict_to_cpu(self.state_dict())

#     def set_weights(self, weights):
#         self.load_state_dict(weights)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv0 = nn.utils.weight_norm(nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2))
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2))
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2))
        self.conv3 = nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2))
        self.conv4 = nn.utils.weight_norm(nn.Conv2d(128, 1, kernel_size=5, stride=2, padding=2))

        # self.conv0 = nn.Conv2d(3, 16, 5, 2, 2)
        # self.conv1 = nn.Conv2d(16, 32, 5, 2, 2)
        # self.conv2 = nn.Conv2d(32, 64, 5, 2, 2)
        # self.conv3 = nn.Conv2d(64, 128, 5, 2, 2)
        # self.conv4 = nn.Conv2d(128, 1, 5, 2, 2)

        self.relu0 = nn.LeakyReLU(negative_slope=0.2)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(-1, 1)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_weights(self):
        return utils.dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)