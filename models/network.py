import math
import warnings
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models.utils import conv, deconv, update_registered_buffers
from models.basic import CompressionModel

class myEntropyBottleneck(EntropyBottleneck):
    def forward(self, x, _noise=None, training=None):
        if _noise is not None:
            x = x + _noise

        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize
        if _noise is None:
            outputs = self.quantize(
                values, "noise" if training else "dequantize", self._get_medians()
            )
        else:
            outputs = values

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood


class myGaussianConditional(GaussianConditional):
    def forward(self, inputs, scales, means=None, training=None, _noise=None):
        if training is None:
            training = self.training
        if _noise is None:
            outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        else:
            outputs = inputs + _noise
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

class PruneMask(nn.Module):
    def __init__(self, mask_n, _modules) -> None:
        super().__init__()

        for _module_name, _module in _modules.items():
            for n, p in _module.named_parameters():
                self.register_buffer(f"{_module_name}__{n.replace('.', '__')}_prune_masks", torch.ones(size=[mask_n]+list(p.shape)))
                self.register_buffer(f"{_module_name}__{n.replace('.', '__')}_untrainable_mask", torch.zeros_like(p))

    def post_train_pruning(self, model, model_old, encoder_id, prune_ratio):
        _modules = model.prune_modules_dict
        if model_old is not None: self.weight_final(_modules, model_old.modules_state_dict)
        for _module_name, _module in _modules.items():
            for n, p in _module.named_parameters():
                _untrainable_mask = getattr(self, f"{_module_name}__{n.replace('.', '__')}_untrainable_mask")
                _masks = getattr(self, f"{_module_name}__{n.replace('.', '__')}_prune_masks")
                _mask = getattr(self, f"{_module_name}__{n.replace('.', '__')}_prune_masks")[encoder_id]
                _weights = p.data * _mask
                if _module_name == "context_prediction" and n == "weight":
                    cutoff_value = torch.abs(_weights).view(-1).kthvalue(np.math.floor(_weights.numel() * (13 + 12 * prune_ratio) / 25))
                else:
                    cutoff_value = torch.abs(_weights).view(-1).kthvalue(np.math.floor(_weights.numel() * (prune_ratio)))
                _mask_pruned = torch.where(torch.less_equal(torch.abs(_weights), cutoff_value[0]), torch.zeros_like(_weights), torch.ones_like(_weights))
                _mask = torch.minimum(_mask_pruned + _untrainable_mask * _mask, torch.ones_like(_mask))
                _masks[encoder_id] = _mask
                self.register_buffer(f"{_module_name}__{n.replace('.', '__')}_prune_masks", _masks)

        self.update_untrainable_mask(_modules, encoder_id)

    def update_untrainable_mask(self, _modules, encoder_id):
        for _module_name, _module in _modules.items():
            for n, p in _module.named_parameters():
                _mask = getattr(self, f"{_module_name}__{n.replace('.', '__')}_prune_masks")[encoder_id]
                _untrainable_mask = getattr(self, f"{_module_name}__{n.replace('.', '__')}_untrainable_mask")
                _untrainable_mask = torch.where(torch.eq(_untrainable_mask, _mask), _untrainable_mask, torch.ones_like(_untrainable_mask))
                self.register_buffer(f"{_module_name}__{n.replace('.', '__')}_untrainable_mask", _untrainable_mask)

    def get_masked_weight(self, _modules, encoder_id):
        for _module_name, _module in _modules.items():
            parameters_masked = _module.state_dict()
            for n, p in _module.named_parameters():
                _mask = getattr(self, f"{_module_name}__{n.replace('.', '__')}_prune_masks")[encoder_id]
                p.data = p.data * _mask
                parameters_masked.update({n: p})

            _module.load_state_dict(parameters_masked)

    def proxy_updating(self):
        # TODO
        pass


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class JointAutoregressiveHierarchicalPriors(CompressionModel):

    def __init__(self, N, M, **kwargs):
        super(CompressionModel, self).__init__()
        self.entropy_bottleneck = None
        self.entropy_bottlenecks = nn.ModuleList([
            myEntropyBottleneck(N) for _ in range(5)
        ])

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True)
        )

        self.recon_layers = nn.ModuleList([
            deconv(N, 3, kernel_size=5, stride=2) for _ in range(5)
        ])

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1, bias=False
        )

        self.prune_masks = PruneMask(5, self.prune_modules_dict)

        self.gaussian_conditional = myGaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def prune_modules_dict(self):
        return {"context_prediction": self.context_prediction, 
                "entropy_parameters": self.entropy_parameters,
                "h_s": self.h_s}
    
    @property
    def modules_state_dict(self):
        _state_dict = dict()
        for _module_name, module in self.prune_modules_dict.items():
            _state_dict.update({_module_name: copy.deepcopy(module.state_dict())})
        return _state_dict

    def forward(self, x, token_id, _noise=[None, None], train_entropy=True):
        y = self.g_a(x)

        entropy_output = self.entropy_prior(y, token_id, _noise, train_entropy)
        y_hat = entropy_output["y_hat"]
        likelihods = entropy_output["likelihoods"]

        x_hat = self.recon_layers[token_id](self.g_s(y_hat))

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "y": y,
            "likelihoods": likelihods,
        }

    def entropy_prior(self, y, token_id, _noise, train_entropy=True):
        z = self.h_a(y)
        if _noise[0] is None:
            z_hat, z_likelihoods = self.entropy_bottlenecks[token_id](z)
        else:
            z_hat, z_likelihoods = self.entropy_bottlenecks[token_id](z, _noise[0])
        params = self.h_s(z_hat)

        if _noise[1] is None:
            y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        else:
            y_hat = y + _noise[1]

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        if not train_entropy:
            gaussian_params = gaussian_params.detach()
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        if _noise[1] is None:
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        else:
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat, _noise=_noise[1])

        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def aux_loss(self, token_id):
        return self.entropy_bottlenecks[token_id].loss()

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    @property
    def _shape_y_param(self):
        return [self.M, 2 ** 4]
    
    @property
    def _shape_z_param(self):
        return [self.N, 2 ** (4 + 2)]

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        # updated |= super().update(force=force)
        for i in range(len(self.entropy_bottlenecks)):
            updated |= self.entropy_bottlenecks[i].update(force=force)
        return updated
