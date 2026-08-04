#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run official fixture layers with source-bound canonical expert routes.

The unmodified official router is audited separately against the complete BF16
cutoff equivalence class. Numerical activation comparison needs a deterministic
expert path, so this evidence runner substitutes the committed route pairs at
the gate output before expert evaluation. The same pairs are supplied to the C
canonical runner.

The official gate returns only routed logits, even though its matrix also emits
shared logits. The adapter therefore re-runs the gate's own BF16 linear operation
from the same hidden state, verifies its routed slice against the returned
logits, and normalizes canonical routed IDs together with the true shared slice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import inkling_fixture_reference as implementation
from inkling_canonical_route_layer_parity import (
    CanonicalRouteError,
    RouteRow,
    load_canonical_routes,
)
from inkling_fixture_reference import FixtureReferenceError
from inkling_release_config import (
    build_transformers_text_config,
    canonicalize_router_pairs,
)


class CanonicalOfficialRouteError(RuntimeError):
    """The official canonical route cannot be applied without guessing."""


class CanonicalOfficialGate(nn.Module):
    """Preserve logits while replacing routed pairs before expert evaluation."""

    def __init__(self, inner: nn.Module, rows: list[RouteRow]) -> None:
        super().__init__()
        if not rows:
            raise CanonicalOfficialRouteError("canonical route rows are empty")
        self.inner = inner
        self.rows = rows
        self.applied = False

    @staticmethod
    def _hidden_states(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        value = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(value, torch.Tensor):
            raise CanonicalOfficialRouteError(
                "official gate call does not expose hidden states"
            )
        return value

    def forward(self, *args: Any, **kwargs: Any):
        hidden_states = self._hidden_states(args, kwargs)
        output = self.inner(*args, **kwargs)
        if not isinstance(output, (tuple, list)) or len(output) < 4:
            raise CanonicalOfficialRouteError(
                "official gate output does not contain logits and route pairs"
            )
        router_logits, _actual_weights, _actual_indices, _actual_shared = output[:4]
        if not isinstance(router_logits, torch.Tensor) or router_logits.ndim != 2:
            raise CanonicalOfficialRouteError(
                f"official router logits have shape {getattr(router_logits, 'shape', None)}"
            )
        tokens = int(router_logits.shape[0])
        if tokens != len(self.rows):
            raise CanonicalOfficialRouteError(
                f"official gate produced {tokens} rows; canonical route has {len(self.rows)}"
            )
        width = len(self.rows[0].indices)
        if width < 1 or any(
            len(row.indices) != width or len(row.weights) != width
            for row in self.rows
        ):
            raise CanonicalOfficialRouteError("canonical route changes top-k width")

        gate = self.inner
        n_routed = int(getattr(gate, "num_experts"))
        n_shared = int(getattr(gate, "n_shared_experts"))
        hidden_dim = int(getattr(gate, "hidden_dim"))
        weight = getattr(gate, "weight")
        if not isinstance(weight, torch.Tensor):
            raise CanonicalOfficialRouteError("official gate has no matrix weight")
        flat = hidden_states.reshape(-1, hidden_dim)
        full_logits = F.linear(flat, weight)
        if full_logits.ndim != 2 or full_logits.shape != (
            tokens,
            n_routed + n_shared,
        ):
            raise CanonicalOfficialRouteError(
                f"reconstructed full router logits have shape {tuple(full_logits.shape)}"
            )
        reconstructed_routed = full_logits[:, :n_routed]
        if reconstructed_routed.shape != router_logits.shape or not torch.equal(
            reconstructed_routed, router_logits
        ):
            delta = float(
                (reconstructed_routed.float() - router_logits.float()).abs().max()
            )
            raise CanonicalOfficialRouteError(
                "reconstructed routed logits differ from official gate output; "
                f"max_abs={delta}"
            )
        shared_logits = full_logits[:, n_routed:]

        ids = torch.tensor(
            [row.indices for row in self.rows],
            device=router_logits.device,
            dtype=torch.int64,
        )
        if torch.any(ids < 0) or torch.any(ids >= n_routed):
            raise CanonicalOfficialRouteError("canonical route expert is out of range")
        if any(len(set(row.indices)) != width for row in self.rows):
            raise CanonicalOfficialRouteError("canonical route repeats an expert")

        selected_logits = torch.cat(
            (router_logits.gather(1, ids), shared_logits), dim=1
        )
        log_probs = F.logsigmoid(selected_logits)
        normalized = torch.exp(
            log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        )
        normalized = (
            normalized
            * float(getattr(gate, "route_scale"))
            * getattr(gate, "global_scale").to(router_logits.dtype)
        )
        computed_routed = normalized[:, :width]
        computed_shared = normalized[:, width:]
        committed_weights = torch.tensor(
            [row.weights for row in self.rows],
            device=router_logits.device,
            dtype=computed_routed.dtype,
        )
        if not torch.equal(computed_routed, committed_weights):
            delta = float(
                (computed_routed.float() - committed_weights.float()).abs().max()
            )
            raise CanonicalOfficialRouteError(
                "canonical routed weights do not reproduce from official logits; "
                f"max_abs={delta}"
            )

        replaced = (
            router_logits,
            committed_weights,
            ids,
            computed_shared,
            *output[4:],
        )
        self.applied = True
        return list(replaced) if isinstance(output, list) else tuple(replaced)


def run_layer_reference_canonical(
    fixture: Any,
    config: Any,
    layer: int,
    inputs: list[list[float]],
    rows: list[RouteRow] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Run one layer, installing a canonical gate only for sparse execution."""
    original_build = implementation.build_layer_from_fixture
    installed: list[CanonicalOfficialGate] = []

    def build(*args: Any, **kwargs: Any):
        module = original_build(*args, **kwargs)
        if rows is not None:
            wrapper = CanonicalOfficialGate(module.mlp.gate, rows)
            module.mlp.gate = wrapper
            installed.append(wrapper)
        return module

    implementation.build_layer_from_fixture = build
    try:
        values = implementation.run_layer_reference(
            fixture,
            config,
            layer,
            inputs,
            device=device,
            dtype=dtype,
        )
    finally:
        implementation.build_layer_from_fixture = original_build
    if rows is not None:
        if len(installed) != 1 or not installed[0].applied:
            raise CanonicalOfficialRouteError(
                f"canonical official route was not applied for layer {layer}"
            )
        canonicalize_router_pairs(values)
    return values


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--route-override", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=tuple(implementation.DTYPES), default="bfloat16")
    args = ap.parse_args(argv)
    try:
        _, load_fixture, write_activation_archive, *_ = implementation._repo_modules()
        fixture = load_fixture(args.fixture)
        config_json = implementation.verify_config_binding(
            fixture, args.model_config
        )
        config = build_transformers_text_config(config_json)
        inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
        if not isinstance(inputs, list) or not inputs:
            raise CanonicalOfficialRouteError(
                "--inputs must contain a nonempty JSON list"
            )
        layers = [int(value) for value in args.layers.split(",") if value]
        if not layers:
            raise CanonicalOfficialRouteError("--layers must be nonempty")
        sparse_layers = [
            layer
            for layer in layers
            if implementation._is_sparse(config, layer)
        ]
        routes = load_canonical_routes(
            args.route_override, fixture, inputs, sparse_layers
        )
        device = torch.device(args.device)
        dtype = implementation.DTYPES[args.dtype]
        _, _, _, transformers_version = implementation._import_transformers()

        merged: dict[str, torch.Tensor] = {}
        for layer in layers:
            traced = run_layer_reference_canonical(
                fixture,
                config,
                layer,
                inputs,
                routes.get(layer),
                device=device,
                dtype=dtype,
            )
            overlap = sorted(set(merged).intersection(traced))
            if overlap:
                raise CanonicalOfficialRouteError(
                    "duplicate activation names while merging layers: "
                    + ", ".join(overlap)
                )
            merged.update(traced)
        write_activation_archive(
            args.out,
            merged,
            metadata={
                "runtime": "transformers-fixture-layer-canonical-route",
                "transformers": transformers_version,
                "layers": layers,
                "positions": len(inputs),
                "dtype": args.dtype,
                "fixture": str(args.fixture),
                "route_override": str(args.route_override),
                "config_sha256": fixture.source.get("config_sha256"),
            },
        )
    except (
        CanonicalOfficialRouteError,
        CanonicalRouteError,
        FixtureReferenceError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        ap.error(str(exc))
        return 2
    print(
        f"wrote canonical-route official layer trace for {layers} to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
