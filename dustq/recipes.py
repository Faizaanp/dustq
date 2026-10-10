"""Draw-allocation recipes, generalising upstream dust.recipe() to L2 and L4 models.

Plan v3.1 section 4.3: the L8 path is byte-equal to upstream (we call upstream's
recipe() directly). Other depths merge adjacent block pairs of the L8 base lists until
the length matches, so the per-step draw totals — and therefore the FLOPs — stay
identical to upstream at the same K. Supported depths: 2, 4, 8.
"""

from __future__ import annotations

from .upstream import ensure_upstream_on_path

ensure_upstream_on_path()

from dust import (  # noqa: E402  (needs third_party/dust on sys.path first)
    DEFAULT_CONFIGS,
    DRAW_CHUNK_SIZES,
    Recipe,
    recipe as upstream_recipe,
)

SUPPORTED_LAYERS = (2, 4, 8)

# Upstream's base per-block lists (scale = population // 256 is applied after merging).
_BASE_WRITERS = [38, 32, 16, 10, 6, 4, 2, 2]
_BASE_HUBS = [6, 6, 4, 2, 2, 2, 2, 2]
_BASE_LOCAL = [2, 20, 6, 12, 8]  # q, v, gate, k, ve (per kind, layer-agnostic)


def _merge_pairs(values: list[int], target_len: int) -> list[int]:
    """Merge adjacent block pairs until len(values) == target_len."""
    values = list(values)
    if len(values) < target_len:
        raise ValueError(f"cannot expand {len(values)} entries to {target_len}")
    while len(values) > target_len:
        merged = [values[i] + values[i + 1] for i in range(0, len(values) - 1, 2)]
        if len(values) % 2:
            merged.append(values[-1])
        values = merged
    return values


def recipe(tokens: str, population: int, n_layer: int) -> Recipe:
    """Upstream recipe() for L8; merged-pair recipes for L2 and L4."""
    if n_layer == 8:
        return upstream_recipe(tokens, population)
    if n_layer not in SUPPORTED_LAYERS:
        raise NotImplementedError(f"recipes are defined for n_layer in {SUPPORTED_LAYERS}, got {n_layer}")
    if tokens not in DEFAULT_CONFIGS:
        raise ValueError(f"tokens must be one of {list(DEFAULT_CONFIGS)}, got {tokens}")
    if population not in DRAW_CHUNK_SIZES:
        raise ValueError(f"population must be one of {list(DRAW_CHUNK_SIZES)}, got {population}")

    optimizer = DEFAULT_CONFIGS[tokens]["optimizer"][population]
    nrep = DRAW_CHUNK_SIZES[population]
    scale = population // 256
    override = DEFAULT_CONFIGS[tokens].get("allocation", {}).get(population, {})

    writers = [v * scale for v in _merge_pairs(override.get("writers", _BASE_WRITERS), n_layer)]
    hubs = [v * scale for v in _merge_pairs(override.get("hubs", _BASE_HUBS), n_layer)]
    hidden = writers.copy()
    embedding = 10 * scale
    local = [v * scale for v in _BASE_LOCAL]

    return Recipe(
        writers, hidden, hubs, embedding, 4 * population,
        dict(zip(("q", "v", "gate", "k", "ve"), local)),
        nrep, **optimizer, gamma=override.get("gamma", 0.98),
    )
