"""Pre-sigmoid logits for the project's binary classifiers (RUNBOOK_v2 Section 3).

Every model in models.py ends in either Dense(1, activation="sigmoid") or a
separate Activation("sigmoid") layer.  Probabilities saturate at exactly 1.0
in float32, so many commits tie at the top score.  The logit (the value fed
into the sigmoid) keeps the ordering.  It is computed from the layer feeding
the sigmoid, in float64, so it is not affected by float32 saturation.
"""

from __future__ import annotations

import keras
import numpy as np


def _pre_sigmoid_parts(model):
    """Return (feature_model, dense_layer_or_None) for computing logits."""
    last = model.layers[-1]
    activation = getattr(last, "activation", None)
    if isinstance(last, keras.layers.Activation):
        # logits = output of the layer before the Activation
        return keras.Model(model.inputs, model.layers[-2].output), None
    if isinstance(last, keras.layers.Dense) and activation is keras.activations.sigmoid:
        # logits = h @ W + b, where h is the Dense layer's input
        return keras.Model(model.inputs, last.input), last
    raise ValueError(f"Unsupported output layer {type(last).__name__} "
                     f"(activation={activation}); cannot compute logits")


# One logit sub-model per model, built on first use. Building a new keras.Model on every call made full-corpus
# scoring (one call per repo per fold model) compile thousands of models on the GPU until it ran out of memory
# (runs_v2/NOTES.md s9; tests in runs_v2/12_oom_fix/tests/). The sub-model shares the model's layers and weights,
# so the output is identical. Keyed by id(); the model is stored with its entry so the id cannot be reused.
_PARTS_CACHE: dict = {}


def _cached_pre_sigmoid_parts(model):
    entry = _PARTS_CACHE.get(id(model))
    if entry is None or entry[0] is not model:
        entry = (model, _pre_sigmoid_parts(model))
        _PARTS_CACHE[id(model)] = entry
    return entry[1]


def predict_logits(model, X, batch_size: int = 256) -> np.ndarray:
    """Pre-sigmoid scores, shape (n,), float64."""
    feature_model, dense = _cached_pre_sigmoid_parts(model)
    h = np.asarray(feature_model.predict(X, batch_size=batch_size, verbose=0), dtype=np.float64)
    if dense is None:
        return h.reshape(-1)
    kernel, bias = (np.asarray(w, dtype=np.float64) for w in dense.get_weights())
    return (h @ kernel + bias).reshape(-1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
