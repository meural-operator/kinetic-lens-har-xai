# Add and test an explanation technique

An explanation class declares `requires` and implements `explain(model, frames, target=None, **options)`, returning an `Explanation` with numeric data and limitations. Start with `examples/custom_xai.py`:

```sh
kinetic-lens --config configs/movinet_kard.json explain --video /path/to/clip.mp4 --method examples.custom_xai:ReverseSensitivity
```

Custom methods work through the CLI without registry edits. To expose a method in the dashboard, add its import path to `METHODS` in `registry.py`, its display description/renderer in `web/app.js`, and a server option whitelist if it needs new inputs. This keeps remotely supplied import paths or file paths out of the API. Local config and CLI plugin arguments are trusted code execution.

The model's **capabilities**, not its class name, determine compatibility:

| Capability | Hook | Enables |
|---|---|---|
| `predict` | `predict(frames)` | Input perturbations and temporal controls |
| `linear_head` | `linear_evidence(frames)` | Exact affine class contrasts and feature-space concepts |
| `spatial_gradients` | `spatial_map(frames,target,rival)` | Model-specific gradient attribution |
| `channels` | `channel_activation`, `channel_predict` | Removal/retention at an internal model cut |

The default MoViNet hooks replay the original downstream layers with fresh states. An unchanged replay must match whole-model logits before intervention. Never assume a similarly named layer in another model is equivalent.

## Concepts and channel selections

Bundled concept directions come from **training-only, action-label-derived family probes**, in the saved classifier's standardized feature coordinates. They are tied to the portable head hash. Four proxy families are arm waving, kicking, object interaction and posture transition. The bundled concepts are not independently annotated arm/leg/object attributes. The concept method reports projection, directional derivative and deletion effects, not per-clip TCAV sign significance.

For a new affine-head model, fit probes from training features only, using `fit_probe(features, binary_targets, alpha)` in `xai/concept.py`. Choose preprocessing/regularization/thresholds without test leakage, test held-out-action and held-out-subject transfer, and compare matched random directions. Add independent concept annotations before making semantic claims. The supplied ridge utility fits a direction; it does not automate those validation studies.

Channel profiles contain training mean values and complete channel rankings for each supported target. The bundled rankings compare activation separation and functional sensitivity. New rankings require your own training-only selection, a matching model head identity and unchanged-replay tests. Mean and zero replacement are artificial references. Cross-class collateral effects, natural-donor restoration and multiple cuts are needed before claiming a complete action-specific circuit.

## What to test

1. Repeatability, finite outputs and identity/no-op reconstruction.
2. Agreement with the actual inference adapter and unchanged weights/state.
3. Prespecified references, seeds, target classes and controls.
4. Matched input/feature perturbations, rather than heatmap appearance alone.
5. Correct predictions and failures, all competing classes, and effects on unrelated actions.
6. New-subject generalization; do not treat repeated interventions as independent people.

See `tests/test_framework.py` for an independent numerical oracle and API/session checks, `scripts/verify_workspace_model.py` for actual-model replay and the detailed [research interpretations](research/INTERPRETATION.md) for positive and negative findings. Method execution is not semantic validation.
