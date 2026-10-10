"""DUSTQ: the only behavioural subclass of upstream DUST (Plan v3.1 section 4.1).

Upstream hard-codes every noise scale as a literal inside DUST.step(). DUSTQ.step() is an
exact copy of upstream step() except that the sigmas come from ``self.sigma``. With the
default sigmas (equal to the upstream literals) a DUSTQ step must be *bitwise identical*
to a DUST step from the same seed — enforced by tests/test_phase0.py. The copy is
deliberate: upstream is never edited, and the diff against upstream step() is exactly the
sigma substitutions (review with: git diff --no-index on the two step functions).
"""

from __future__ import annotations

import torch

from .upstream import ensure_upstream_on_path

ensure_upstream_on_path()

from dust import DUST, average_ranks  # noqa: E402

# Upstream's literals (dust.py, DUST.step): keep in sync or the bitwise test fails.
DEFAULT_SIGMAS = {
    "writer": 0.2,       # attn.c_proj + mlp.c_proj output jitter
    "hidden": 0.2,       # mlp.c_fc output jitter
    "emb": 0.2,          # token embedding jitter
    "hub_shallow": 0.2,  # attention output o@l, first half of blocks
    "hub_deep": 0.4,     # attention output o@l, deeper half
    "head": 0.05,        # lm_head logits slabs
    "local": 0.05,       # attention internals (q, k, v, gate, ve)
    "scalar": 0.03,      # residual mixing scalars (weight-space ES)
}


class DUSTQ(DUST):
    """DUST with configurable noise scales and a resumable noise generator state."""

    def __init__(self, model, settings, seed=42, rank=0, world=1, sigmas=None):
        super().__init__(model, settings, seed=seed, rank=rank, world=world)
        self.sigma = dict(DEFAULT_SIGMAS)
        if sigmas:
            unknown = set(sigmas) - set(DEFAULT_SIGMAS)
            if unknown:
                raise ValueError(f"unknown sigma keys: {sorted(unknown)}")
            self.sigma.update(sigmas)

    def state_dict(self):
        """Noise-generator state, for resume (model/optimizer state is handled by runner)."""
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """One clean pass, direct-loss estimates, local attention estimates, then an SGD update."""
        self.capture(x, y)
        r, world = self.settings, self.world
        gradients = {}
        for layer in range(self.model.config.n_layer):
            prefix = f'transformer.h.{layer}.'
            groups = [
                ([prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], r.writers[layer], self.sigma['writer']),
                ([prefix + 'mlp.c_fc'], r.hidden[layer], self.sigma['hidden']),
            ]
            for sites, count, sigma in groups:
                for site, error in self.direct_errors(sites, count // world, sigma).items():
                    gradients[site + '.weight'] = self.reassemble(site, error)

        site = 'transformer.wte'
        error = self.direct_errors([site], r.embedding // world, self.sigma['emb'])[site]
        gradients[site + '.weight'] = self.reassemble(site, error)

        # Attention outputs, each block alone. The deeper half moves the loss too little at
        # the shallow sigma for a readable reward, so it uses a deeper sigma. The targets are
        # averaged over GPUs so that every GPU scores the attention internals against the
        # estimate from the whole population.
        targets = {}
        for layer in range(self.model.config.n_layer):
            site = f'o@{layer}'
            sigma = self.sigma['hub_shallow'] if layer < self.model.config.n_layer // 2 else self.sigma['hub_deep']
            targets[layer] = average_ranks(self.direct_errors([site], r.hubs[layer] // world, sigma)[site])

        gradients['lm_head.weight'] = self.reassemble('lm_head', self.head_error(r.head // world, self.sigma['head']))
        for kind, count in r.local.items():
            for site, error in self.local_errors(kind, count // world, targets, self.sigma['local']).items():
                gradients[site + '.weight'] = self.reassemble(site, error)
        gradients.update(self.scalar_gradients(8 // world, self.sigma['scalar']))

        # Dividing the summed estimates by the token count gives the gradient of the mean loss.
        valid_tokens = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            gradient = average_ranks((gradients[name] / valid_tokens).to(parameter.dtype))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = gradient
        optimizer.step()
        return self.clean_loss[y != -1].mean().item()
