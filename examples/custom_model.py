"""Copy this adapter and replace only loading/preprocessing/predict.

Config adapter: examples.custom_model:ExampleModel
Run from repository root, or install your plugin as a separate Python package.
"""
from kinetic_lens.models.demo import DemoModel


class ExampleModel(DemoModel):
    name="Example custom adapter — replace with your trained model"
    # predict receives sampled RGB uint8 [T,H,W,3]. Return raw logits [classes].
    # Preserve the training preprocessing, class order and fresh-state semantics.
