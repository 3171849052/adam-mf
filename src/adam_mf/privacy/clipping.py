"""Exact scaled norms in Opacus' two-backward Fast/Ghost clipping structure."""
import torch
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)


class ScaledGhostModule(GradSampleModuleFastGradientClipping):
    """Replace norm samplers by exact layerwise scaled Fast Clipping samplers.

    An arbitrary elementwise Adam scale does not factor into the usual
    Linear Ghost identity. Materialize only one layer's per-example gradients,
    multiply by its frozen FP32 scale, reduce immediately, and discard them.
    Opacus still handles activation/backprop hooks and the two-pass protocol.
    No full-model per-example gradients are retained.
    """
    def __init__(self, model, max_grad_norm):
        self.NORM_SAMPLERS = {}
        for layer_type, grad_sampler in self.GRAD_SAMPLERS.items():
            def scaled_norm(layer, activations, backprops, sampler=grad_sampler):
                samples = sampler(layer, activations, backprops.float())
                return {p: (gs.float() * p._logical_scale).flatten(1).norm(2, dim=1)
                        for p, gs in samples.items()}
            self.NORM_SAMPLERS[layer_type] = scaled_norm
        self.NORM_SAMPLERS[torch.nn.Embedding] = lambda layer, a, b: embedding_norm(layer, a, b, scaled=True)
        super().__init__(model, loss_reduction='sum', max_grad_norm=max_grad_norm,
                         use_ghost_clipping=True)
        for p in self.parameters():
            p._logical_scale = torch.ones_like(p, dtype=torch.float32)


@torch.no_grad()
def freeze_adam_scale(optimizer, eps_scale):
    """Use vhat of the previous completed Adam step; vhat_0=0."""
    scales = []
    for group in optimizer.param_groups:
        beta2 = group['betas'][1]
        for p in group['params']:
            state = optimizer.state[p]
            if len(state) == 0:
                vhat = torch.zeros_like(p, dtype=torch.float32)
            else:
                step = int(state['step'].item())
                vhat = state['exp_avg_sq'].float() / (1 - beta2 ** step)
            p._logical_sqrt_vhat = vhat.sqrt()
            scale = 1.0 / (p._logical_sqrt_vhat + eps_scale)
            p._logical_scale = scale
            scales.append(scale)
    return scales


def clipped_microbatch(model, inputs, targets, max_grad_norm):
    """First backward measures norms; second backward sums clipped gradients."""
    losses = torch.nn.functional.cross_entropy(model(inputs), targets, reduction='none')
    losses.sum().backward(retain_graph=True)
    norms = model.get_norm_sample().detach()
    coefficients = (max_grad_norm / norms.clamp_min(1e-30)).clamp(max=1.0)
    model.zero_grad(set_to_none=True)
    model.disable_hooks()
    (losses * coefficients).sum().backward()
    model.enable_hooks()
    if not torch.isfinite(norms).all() or not torch.isfinite(losses).all():
        raise FloatingPointError('non-finite loss or clipping norms')
    model.last_norms = norms
    model.last_coefficients = coefficients
    return float(losses.detach().sum()), int((norms > max_grad_norm).sum())


@torch.no_grad()
def embedding_norm(layer, activations, backprops, scaled=False):
    """Exact sparse norm: sum repeated-token derivatives BEFORE squaring.

    g_i[v] = sum_{j: token_ij=v} b_ij. Thus ||s*g_i||^2 is the sum
    over unique (sample, token) pairs. Storage O(B*L*H), never O(B*V*H).
    Padding has zero derivative, matching nn.Embedding(padding_idx=...).
    """
    ids = activations[0].reshape(backprops.shape[0], -1)
    B,L = ids.shape
    keys = (torch.arange(B,device=ids.device)[:,None]*layer.num_embeddings+ids).flatten()
    unique, inverse = torch.unique(keys,sorted=True,return_inverse=True)
    values = backprops.float().reshape(-1,layer.embedding_dim)
    if layer.padding_idx is not None:
        values = values * (ids.flatten()!=layer.padding_idx)[:,None]
    summed = torch.zeros(len(unique),layer.embedding_dim,device=values.device)
    summed.index_add_(0,inverse,values)
    if scaled:
        summed.mul_(layer.weight._logical_scale[unique % layer.num_embeddings])
    squared = summed.square().sum(1)
    norms = torch.zeros(B,device=values.device)
    norms.index_add_(0,unique // layer.num_embeddings,squared)
    return {layer.weight:norms.sqrt()}


class StandardGhostModule(GradSampleModuleFastGradientClipping):
    """Opacus Ghost clipping, with exact padding-aware Embedding norm."""
    def __init__(self, model, max_grad_norm):
        self.NORM_SAMPLERS = dict(type(self).NORM_SAMPLERS)
        self.NORM_SAMPLERS[torch.nn.Embedding] = embedding_norm
        super().__init__(model,loss_reduction='sum',max_grad_norm=max_grad_norm,use_ghost_clipping=True)
