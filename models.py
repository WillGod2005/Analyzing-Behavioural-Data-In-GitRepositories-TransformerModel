"""
models.py — Neural network architecture definitions for security patch detection.

This module defines all model architectures used in the project:

PRIMARY MODELS (used in experiments):
  - transformer():  Pre-norm Transformer Encoder (~180K params) — our main
                    contribution. Uses self-attention over event windows to
                    detect behavioral patterns around security patches.
  - conv1d():       Farhi's original Conv1D baseline architecture. Kept
                    unchanged for fair comparison.

ALTERNATIVE MODELS (available but not primary):
  - lstm, gru, bilstm, bigru:  Recurrent architectures
  - lstm_cnn, gru_cnn:         Hybrid CNN+RNN architectures
  - conv1d2, conv1d_more_layers, super_complicated, lstm_autoencoder

CUSTOM KERAS LAYERS (9 layers, all serializable for model saving):
  - CenterTokenPooling:       Extracts the center timestep (target commit)
  - LastTokenPooling:          Extracts the last timestep (causal mode)
  - PaddingMask:               Detects left-padded (all-zero) timesteps
  - ExpandMaskDim:             Reshapes mask for attention compatibility
  - MaskedGlobalAveragePooling: Averages only non-padded timesteps
  - AddPositionalEmbedding:    Learned position encodings
  - StochasticDepth:           Layer-drop regularisation (tested, not used)
  - GatherFeatures:            Selects feature columns by index
  - ExtractStatic:             Extracts static features from center timestep

All model factory functions follow the same interface:
    model = model_fn(xshape1, xshape2, optimizer)
    where xshape1 = sequence length, xshape2 = number of features (435)

Used by: train.py (model_selector), evaluate_cvefixes.py (model loading),
         analyse.py (model training for statistical tests)
"""

import tensorflow as tf
from tensorflow import keras

from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Convolution1D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
    LSTM,
    Layer,
    LayerNormalization,
    MaxPool1D,
    MaxPooling1D,
    MultiHeadAttention,
    RepeatVector,
    TimeDistributed,
)
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, AdamW


# =============================================================================
# Feature classification — which features change per event vs stay constant
# =============================================================================

# Features that VARY per event (temporal features).  These change at each
# timestep in the event window — they represent what happened at/around each
# event (commit counts, code changes, event type flags, timing features).
#
# Everything NOT in this set is STATIC per repo (metadata features like
# language one-hot, owner flags, creation year) — identical at every timestep.
#
# This distinction is used by the two-stream and temporal-focus architectures
# to separate temporal signals from static context.  The single-stream
# architecture (our best performer) processes all 435 features together.
TEMPORAL_FEATURE_NAMES = {
    # Raw numeric features from the repo CSV
    "Add", "Del", "Files",
    # Event type one-hot flags (21 GitHub event types)
    "PullRequestEvent", "PushEvent", "ReleaseEvent", "DeleteEvent",
    "issues", "CreateEvent", "releases", "IssuesEvent", "ForkEvent",
    "WatchEvent", "PullRequestReviewCommentEvent", "stargazers",
    "pullRequests", "commits", "CommitCommentEvent", "MemberEvent",
    "GollumEvent", "IssueCommentEvent", "forks", "PullRequestReviewEvent",
    "PublicEvent",
    # GraphQL-derived recency features (days since last activity in each category)
    "days_since_forks", "days_since_issues", "days_since_pullRequests",
    "days_since_releases", "days_since_stargazers",
    # Derived temporal features engineered in dataset_utils.py:fix_repo_shape()
    "time_delta_hours",   # Log-scaled hours since previous event
    "code_churn",         # Log-scaled total lines changed (|Add| + |Del|)
    "add_del_ratio",      # Proportion of additions vs total changes
    "is_commit",          # Binary: 1 if event has a commit hash
    "event_entropy",      # Shannon entropy of event types in sliding window
    "rel_position",       # Normalised position in sequence (-1 to +1)
    "burst_rate",         # Log-scaled event count within ±1 hour
    "hour_sin", "hour_cos",  # Cyclical encoding of hour-of-day
    "dow_sin", "dow_cos",    # Cyclical encoding of day-of-week
    "type_changed",       # Binary: 1 if event type differs from previous
    "cum_commits",        # Running cumulative count of commits
    "repo_age_days",      # Log-scaled days since repo creation
    "gql_additions", "gql_deletions",  # GraphQL repo-wide code churn (log-scaled)
    # Feature interactions (products of two temporal features)
    "commit_urgency",     # time_delta_hours × is_commit
    "churn_diversity",    # code_churn × event_entropy
}


def get_feature_indices(columns):
    """Split column names into temporal (per-event) and static (per-repo) index lists.

    Used by the two-stream and temporal-focus transformer architectures to
    route features to the appropriate processing branch.

    Args:
        columns: List/Index of feature column names (length 435).

    Returns:
        temporal_idx: List of integer indices for temporal features (~47 features).
        static_idx:   List of integer indices for static features (~388 features).
    """
    temporal_idx = []
    static_idx = []
    for i, col in enumerate(columns):
        if col in TEMPORAL_FEATURE_NAMES:
            temporal_idx.append(i)
        else:
            static_idx.append(i)
    return temporal_idx, static_idx


# =============================================================================
# Alternative model architectures (from Farhi's original codebase)
# =============================================================================
# These are kept for comparison but are NOT the primary models.
# All follow the same interface: model_fn(xshape1, xshape2, optimizer) → Model

def lstm(xshape1, xshape2, optimizer):
    """4-layer stacked LSTM with decreasing hidden sizes (100→50→25→12).

    A standard recurrent architecture that processes the event sequence
    step-by-step.  The final LSTM layer does not return sequences, producing
    a single summary vector that feeds into the sigmoid classifier.
    Dropout of 0.2 is applied within LSTM layers 2-4 for regularisation.
    """
    model = Sequential()
    # Layer 1: 100 units, processes full sequence, outputs sequence for next layer
    model.add(LSTM(units=100, activation='tanh',
             input_shape=(xshape1, xshape2), return_sequences=True))
    # Layers 2-3: Progressively smaller, with 20% recurrent dropout
    model.add(LSTM(units=50, activation='tanh',
             return_sequences=True, dropout=0.2))
    model.add(LSTM(units=25, activation='tanh',
             return_sequences=True, dropout=0.2))
    # Layer 4: Final LSTM, outputs single vector (return_sequences=False)
    model.add(LSTM(units=12, activation='tanh', dropout=0.2))
    # Binary classifier: single sigmoid output for vuln probability
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return model


def gru(xshape1, xshape2, optimizer):
    """4-layer stacked GRU with decreasing hidden sizes (100→50→25→12).

    Same structure as the LSTM model but uses GRU cells, which have fewer
    parameters (2 gates vs LSTM's 3) and can train faster on small datasets.
    """
    model = Sequential()
    model.add(GRU(units=100, activation='tanh', name='first_lstm',
             input_shape=(xshape1, xshape2), return_sequences=True))
    model.add(GRU(units=50, activation='tanh',
             return_sequences=True, dropout=0.2))
    model.add(GRU(units=25, activation='tanh',
             return_sequences=True, dropout=0.2))
    model.add(GRU(units=12, activation='tanh', dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return model


def conv1d2(xshape1, xshape2, optimizer):
    """Conv1D + GRU hybrid: 3 conv blocks → GRU → dense classifier.

    Uses 3 Conv1D layers with MaxPooling to extract local patterns from
    the event sequence, then a GRU to capture sequential dependencies
    in the downsampled representation.
    """
    x = Sequential()
    # 3 convolutional blocks: extract local patterns (kernel_size=2 = pairs of events)
    x.add(Conv1D(filters=64, kernel_size=2, padding='same',
          activation='relu', input_shape=(xshape1, xshape2)))
    x.add(MaxPooling1D(pool_size=2))      # Halve sequence length
    x.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
    x.add(MaxPooling1D(pool_size=2))      # Halve again
    x.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
    x.add(MaxPooling1D(pool_size=2))      # Halve again → seq_len/8
    # GRU processes the downsampled sequence
    x.add(GRU(64))
    x.add(Dropout(0.5))
    x.add(Dense(10, activation="relu"))
    x.add(Dense(1, activation="sigmoid"))
    x.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
    return x


def super_complicated(xshape1, xshape2, optimizer):
    """Conv1D + LSTM with per-timestep output (Functional API).

    A wide Conv1D (kernel_size=8) captures longer-range local patterns,
    followed by an LSTM that processes the entire sequence.  Note: this
    produces a per-timestep output via Dense(1) on return_sequences=True,
    which is atypical for binary classification.
    """
    input_layer = Input(shape=(xshape1, xshape2))
    # Wide convolution captures patterns spanning 8 consecutive events
    conv1 = Conv1D(filters=32,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(input_layer)
    # LSTM processes the convolved features, returning full sequence
    lstm1 = LSTM(32, return_sequences=True)(conv1)
    # Per-timestep binary prediction
    output_layer = Dense(1, activation='sigmoid')(lstm1)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def conv1d_more_layers(xshape1, xshape2, optimizer):
    """Deep Conv1D with 7 progressively narrowing dense layers.

    Heavy architecture with 500 conv filters and a deep MLP classifier
    (100→100→100→70→50→30→1).  Dropout 0.1 between every layer.
    """
    conv1d = Sequential()
    conv1d.add(Conv1D(filters=500, kernel_size=2, activation='relu',
               input_shape=(xshape1, xshape2)))
    conv1d.add(Dropout(0.1))
    conv1d.add(MaxPooling1D(pool_size=2))
    conv1d.add(Dropout(0.1))
    conv1d.add(Flatten())
    # 7 dense layers with decreasing width — very deep for this problem size
    conv1d.add(Dense(100, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(100, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(100, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(70, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(50, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(30, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(1, activation='sigmoid'))
    conv1d.compile(loss='binary_crossentropy',
                   optimizer=optimizer, metrics=['accuracy'])

    return conv1d


def lstm_autoencoder(xshape1, xshape2, optimizer):
    """LSTM autoencoder: encode sequence → bottleneck → decode → classify.

    Encoder: LSTM(100) → LSTM(50) compresses the sequence into a fixed vector.
    RepeatVector broadcasts this vector back to sequence length.
    Decoder: LSTM(50) → LSTM(100) reconstructs, then Dense(1) classifies
    each timestep.  This architecture learns a compressed representation
    of the event sequence.
    """
    model = Sequential()
    # Encoder: compress sequence into a single vector
    model.add(LSTM(100, activation='tanh', input_shape=(
        xshape1, xshape2), return_sequences=True))
    model.add(LSTM(50, activation='tanh', return_sequences=False))  # Bottleneck
    # Repeat bottleneck vector for each timestep
    model.add(RepeatVector(xshape1))
    # Decoder: reconstruct sequence from bottleneck
    model.add(LSTM(50, activation='tanh', return_sequences=True))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    # Per-timestep classification
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def bilstm(xshape1, xshape2, optimizer):
    """4-layer Bidirectional LSTM (100→50→25→12 per direction).

    Bidirectional processing reads the event sequence both forwards and
    backwards, doubling the hidden state at each layer.  This is appropriate
    for symmetric windows where events both before and after the target
    commit are available.
    """
    model = Sequential()
    # Each Bidirectional layer doubles the output dim (e.g. 100→200 total)
    model.add(Bidirectional(LSTM(units=100, activation='tanh', name='first_lstm', return_sequences=True,
             input_shape=(xshape1, xshape2), dropout=0.2)))
    model.add(Bidirectional(LSTM(units=50, activation='tanh',
             return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(units=25, activation='tanh',
             return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(units=12, activation='tanh', dropout=0.2)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return model


def bigru(xshape1, xshape2, optimizer):
    """4-layer Bidirectional GRU (100→50→25→12 per direction).

    Same as bilstm but with GRU cells for faster training.
    """
    model = Sequential()
    model.add(Bidirectional(GRU(units=100, activation='tanh', name='first_lstm',
             input_shape=(xshape1, xshape2), return_sequences=True)))
    model.add(Bidirectional(GRU(units=50, activation='tanh',
             return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(GRU(units=25, activation='tanh',
             return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(GRU(units=12, activation='tanh', dropout=0.2)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return model


def lstm_cnn(xshape1, xshape2, optimizer):
    """CNN → LSTM → CNN hybrid: interleaved convolution and recurrence.

    First Conv1D extracts local patterns, LSTM captures sequential
    dependencies, second Conv1D refines the representation before
    flattening into the sigmoid classifier.
    """
    model = Sequential()
    model.add(Convolution1D(64, 3, input_shape=(xshape1, xshape2)))
    model.add(MaxPooling1D())
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Convolution1D(32, 3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def gru_cnn(xshape1, xshape2, optimizer):
    """CNN → GRU → CNN hybrid: same as lstm_cnn but with GRU and wider filters.

    Uses 256 filters in the first conv layer (vs 64 in lstm_cnn) and GRU
    with 300 units, making this a heavier variant.
    """
    model = Sequential()
    model.add(Convolution1D(256, 3, input_shape=(xshape1, xshape2)))
    model.add(MaxPooling1D())
    model.add(GRU(300, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Convolution1D(128, 3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# =============================================================================
# Conv1D — Farhi's original baseline architecture
# =============================================================================

def conv1d(xshape1, xshape2, optimizer):
    """Farhi's original Conv1D baseline model.

    Architecture:
        Conv1D(256, kernel=2, tanh) → MaxPool(2) → Flatten →
        Dense(1024, tanh) → Drop(0.3) → Dense(256, tanh) → Drop(0.3) →
        Dense(64, tanh) → Drop(0.3) → Dense(1, sigmoid)

    Key design choices (from Farhi 2024):
        - tanh activations (not ReLU) — works well with Adagrad
        - Adagrad optimizer at lr=0.01 — adaptive per-parameter learning rates
        - jit_compile=True — XLA compilation for faster training
        - steps_per_execution=100 — reduces Python overhead by running
          100 training steps per graph execution call

    This architecture is kept UNCHANGED from Farhi's code so that any
    performance differences in our experiments are due to the transformer
    architecture and new features, not baseline modifications.
    """
    model1 = Sequential()
    DROPOUT = 0.3
    # Single conv layer: 256 filters of width 2 (looks at pairs of adjacent events)
    model1.add(Conv1D(filters=256, kernel_size=2, activation='tanh',
               input_shape=(xshape1, xshape2)))
    # Max pooling halves the sequence length
    model1.add(MaxPooling1D(pool_size=2))
    # Flatten the 2-D feature map into a 1-D vector for dense layers
    model1.add(Flatten())
    # 3-layer MLP classifier with dropout between each layer
    model1.add(Dense(1024, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(256, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(64, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    # Output: single sigmoid for binary classification probability
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy',
                   jit_compile=True, steps_per_execution=100,
                   optimizer=Adagrad(
                       learning_rate=0.01), metrics=['accuracy'])

    return model1


# =============================================================================
# Custom Keras Layers for the Transformer
# =============================================================================
# All custom layers are decorated with @register_keras_serializable() so
# they can be saved to .keras files and loaded back without manual
# custom_objects registration.  Each implements get_config() for serialisation.

@tf.keras.utils.register_keras_serializable()
class CenterTokenPooling(Layer):
    """Extract the CENTER timestep from the encoder output.

    Used with symmetric (before_cve) windows where the target commit sits
    at the middle of the event sequence.  After the transformer processes
    all timesteps via self-attention, this layer extracts just the center
    position's representation — which has been enriched with context from
    both preceding and following events.

    Input:  (batch, seq_len, d_model)
    Output: (batch, d_model)
    """
    def call(self, inputs):
        center_index = tf.shape(inputs)[1] // 2  # Integer division for center
        return inputs[:, center_index, :]         # Select center timestep

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class LastTokenPooling(Layer):
    """Extract the LAST timestep from the encoder output.

    Used with causal (only_before) windows where the target commit is at
    the END of the event sequence.  The last position's representation
    has attended to all preceding events.

    Input:  (batch, seq_len, d_model)
    Output: (batch, d_model)
    """
    def call(self, inputs):
        return inputs[:, -1, :]                   # Select last timestep

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class PaddingMask(Layer):
    """Create a boolean mask identifying real (non-padded) timesteps.

    Left-padded timesteps are all-zero vectors.  This layer checks whether
    ANY feature in a timestep is non-zero.  The resulting mask is used by
    MaskedGlobalAveragePooling to exclude padding from the average.

    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len) — True for real tokens, False for padding
    """

    def call(self, inputs):
        # reduce_any across features: True if ANY feature is non-zero
        return tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class ExpandMaskDim(Layer):
    """Reshape mask from (batch, seq) to (batch, 1, seq) for attention layers.

    MultiHeadAttention expects the attention mask to have an extra dimension
    for broadcasting across attention heads.

    Input:  (batch, seq_len)
    Output: (batch, 1, seq_len)
    """

    def call(self, mask):
        return mask[:, tf.newaxis, :]

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class MaskedGlobalAveragePooling(Layer):
    """Average pooling that ignores left-padded (all-zero) timesteps.

    Standard GlobalAveragePooling1D would include zero-padded timesteps
    in the average, diluting the signal.  This layer uses a boolean mask
    to sum only real timesteps and divide by the actual count.

    Input:  [x: (batch, seq, d_model), mask: (batch, seq)]
    Output: (batch, d_model)
    """

    def call(self, inputs):
        x, padding_mask = inputs[0], inputs[1]
        # Convert boolean mask to float and add feature dimension for broadcasting
        mask_float = tf.cast(padding_mask, x.dtype)[:, :, tf.newaxis]
        # Sum only non-padded timesteps
        summed = tf.reduce_sum(x * mask_float, axis=1)
        # Count non-padded timesteps (clamp to 1 to avoid division by zero)
        count = tf.maximum(tf.reduce_sum(mask_float, axis=1), 1.0)
        return summed / count

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class AddPositionalEmbedding(Layer):
    """Add learned positional embeddings to the input sequence.

    Unlike sinusoidal positional encodings (Vaswani et al.), these embeddings
    are LEARNED during training.  Each position (0 to seq_len-1) gets a
    trainable d_model-dimensional vector that is added to the input.

    This gives the transformer awareness of event ORDER, which is critical
    since self-attention is inherently permutation-invariant.

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model) — with position information added
    """

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        # Embedding lookup table: position index → d_model-dimensional vector
        self.pos_embedding = keras.layers.Embedding(input_dim=seq_len, output_dim=d_model)

    def call(self, x):
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        # Look up positional embeddings and add to input
        pos_embed = self.pos_embedding(positions)
        return x + pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


@tf.keras.utils.register_keras_serializable()
class StochasticDepth(Layer):
    """Randomly drop the entire residual branch during training (layer-drop).

    With probability drop_rate, the entire output is zeroed out, effectively
    skipping this layer.  At inference time, the full output is used but
    scaled by keep_prob to maintain expected values.

    This is a regularisation technique from "Deep Networks with Stochastic
    Depth" (Huang et al., 2016).  Tested in experiments but not used in
    the final configuration (StochasticDepth is available but drop_rate
    defaults to 0.0 in _build_encoder).
    """

    def __init__(self, drop_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        if not training or self.drop_rate == 0.0:
            return x  # No-op during inference or when drop_rate=0
        keep_prob = 1.0 - self.drop_rate
        # Sample a single random value per batch item (broadcast across features)
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.cast(random_tensor < keep_prob, x.dtype)
        # Scale by 1/keep_prob to maintain expected value
        return x * binary_mask / keep_prob

    def get_config(self):
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate})
        return config


@tf.keras.utils.register_keras_serializable()
class GatherFeatures(Layer):
    """Select specific feature columns from the input tensor.

    Used by two-stream and temporal-focus architectures to split the 435
    features into temporal (~47) and static (~388) subsets, routing each
    to a specialised processing branch.

    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len, len(indices))
    """

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)

    def call(self, inputs):
        # tf.gather on axis=-1 selects specific feature columns
        return tf.gather(inputs, self.indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config["indices"] = self.indices
        return config


@tf.keras.utils.register_keras_serializable()
class ExtractStatic(Layer):
    """Extract static (per-repo) features from the center timestep only.

    Static features are identical at every timestep, so we only need to
    read them once.  This layer takes the center timestep and selects
    only the static feature columns, producing a 1-D vector per sample.

    Used by the two-stream architecture to create the static branch input.

    Input:  (batch, seq_len, n_features)
    Output: (batch, len(indices))
    """

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)

    def call(self, inputs):
        # Select center timestep (static features are the same at all timesteps)
        center = tf.shape(inputs)[1] // 2
        step = inputs[:, center, :]
        # Select only static feature columns
        return tf.gather(step, self.indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config["indices"] = self.indices
        return config


# =============================================================================
# Transformer encoder — shared builder function
# =============================================================================

def _build_encoder(x, xshape1, d_model, num_heads, ff_dim, num_layers, dropout,
                    use_conv=False):
    """Build a stack of pre-norm transformer encoder layers.

    This implements the "Pre-LayerNorm" variant of the transformer encoder
    (Xiong et al., 2020), where LayerNorm is applied BEFORE attention and
    FFN rather than after.  Pre-norm is more stable during training and
    does not require learning rate warmup as critically (though we still
    use warmup for best results).

    Architecture per layer:
        LayerNorm → MultiHeadAttention → Dropout → Residual Add
        LayerNorm → Dense(ff_dim, relu) → Dropout → Dense(d_model) → Dropout → Residual Add

    Args:
        x:          Input tensor, shape (batch, seq_len, input_features).
        xshape1:    Sequence length (needed for positional embeddings).
        d_model:    Model dimension (default 64). All internal representations
                    use this width.
        num_heads:  Number of attention heads (default 4). Each head has
                    key_dim = d_model // num_heads = 16.
        ff_dim:     Feed-forward hidden dimension (default 256). The expansion
                    factor is ff_dim/d_model = 4x.
        num_layers: Number of encoder layers to stack (default 2).
        dropout:    Dropout rate applied within attention and FFN (default 0.2).
        use_conv:   If True, add a Conv1D local feature extraction layer before
                    the transformer (tested, not used in best config).

    Returns:
        Tensor of shape (batch, seq_len, d_model) — the encoded representation.
    """
    # Project input features (e.g. 435) down to d_model dimensions (64)
    x = Dense(d_model)(x)

    # Optional: local feature extraction via Conv1D before transformer
    # This was tested to capture local patterns (adjacent event pairs)
    # but did not improve over the pure transformer in experiments.
    if use_conv:
        conv_out = Conv1D(d_model, kernel_size=3, padding="same", activation="relu")(x)
        conv_out = Dropout(dropout)(conv_out)
        x = Add()([x, conv_out])  # Residual connection

    # Add learned positional embeddings so the transformer knows event order
    x = AddPositionalEmbedding(xshape1, d_model)(x)

    # Stack num_layers encoder layers
    for _ in range(num_layers):
        # === Self-attention sub-layer (pre-norm) ===
        attn_input = LayerNormalization(epsilon=1e-6)(x)
        # Self-attention: each timestep attends to all other timesteps
        # key_dim = d_model // num_heads = 64//4 = 16 per head
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )(attn_input, attn_input)  # Self-attention: query=key=value
        attn_output = Dropout(dropout)(attn_output)
        x = Add()([x, attn_output])  # Residual connection

        # === Feed-forward sub-layer (pre-norm) ===
        ffn_input = LayerNormalization(epsilon=1e-6)(x)
        # FFN: expand to ff_dim (256), apply ReLU, project back to d_model (64)
        ffn_output = Dense(ff_dim, activation="relu")(ffn_input)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout)(ffn_output)
        x = Add()([x, ffn_output])  # Residual connection

    # Final layer normalisation (required by pre-norm architecture)
    return LayerNormalization(epsilon=1e-6)(x)


# =============================================================================
# Transformer — Primary model (our contribution)
# =============================================================================

def transformer(
    xshape1,
    xshape2,
    optimizer=None,
    d_model=64,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    dropout=0.2,
    pooling="last",
    columns=None,
    temporal_focus=False,
):
    """Build and compile the transformer encoder model for security patch detection.

    This is the PRIMARY model and our main contribution.  It uses self-attention
    to learn which events in the surrounding window are most informative for
    predicting whether the target commit is a security patch.

    Three architecture variants are supported (controlled by columns and
    temporal_focus arguments):

    1. SINGLE-STREAM (columns=None) — BEST PERFORMER:
       All 435 features are processed together.  The transformer sees both
       temporal events and static metadata at every timestep, learning to
       use metadata as context for interpreting temporal patterns.

    2. TWO-STREAM (columns provided, temporal_focus=False):
       Temporal features (~47) go through the transformer encoder.
       Static features (~388) go through a separate MLP.
       Both are concatenated before the classifier.
       Tested in Exp 5, performed worse than single-stream.

    3. TEMPORAL-FOCUS (columns provided, temporal_focus=True):
       Similar to two-stream but with a lighter static branch (32 dims
       instead of 64).  Tested in Exp 17, also performed worse.

    Args:
        xshape1:        Sequence length (number of timesteps in window).
        xshape2:        Number of features per timestep (typically 435).
        optimizer:      Keras optimizer. If None, uses AdamW(lr=1e-4, wd=1e-4).
        d_model:        Transformer model dimension (default 64).
        num_heads:      Number of attention heads (default 4).
        ff_dim:         Feed-forward hidden dim (default 256, = 4× d_model).
        num_layers:     Number of encoder layers (default 2).
        dropout:        Dropout rate (default 0.2).
        pooling:        How to reduce sequence to single vector:
                        "center" — CenterTokenPooling (for symmetric windows)
                        "last"   — LastTokenPooling (for causal windows)
                        "avg"    — MaskedGlobalAveragePooling
        columns:        Feature column names (needed for two-stream/temporal-focus).
                        If None, uses single-stream architecture.
        temporal_focus:  If True and columns provided, use temporal-focus variant.

    Returns:
        Compiled Keras Model ready for training (~180K parameters).
    """
    inputs = Input(shape=(xshape1, xshape2))

    # -------------------------------------------------------------------------
    # Variant 3: TEMPORAL-FOCUS (transformer on temporal only, light static MLP)
    # -------------------------------------------------------------------------
    if temporal_focus and columns is not None and len(columns) > 0:
        temporal_idx, static_idx = get_feature_indices(columns)

        # Transformer processes only temporal features (~47 features)
        temporal = GatherFeatures(temporal_idx)(inputs)  # (batch, seq, ~47)
        x = _build_encoder(
            temporal, xshape1, d_model, num_heads, ff_dim, num_layers, dropout
        )

        # Pool sequence to single vector based on window type
        if pooling == "center":
            x = CenterTokenPooling()(x)
        elif pooling == "last":
            x = LastTokenPooling()(x)
        else:
            x = MaskedGlobalAveragePooling()([x, PaddingMask()(temporal)])

        # Static features: lightweight projection (compressed to 32 dims)
        static = ExtractStatic(static_idx)(inputs)  # (batch, ~388)
        s = LayerNormalization(epsilon=1e-6)(static)
        s = Dense(32, activation="relu")(s)
        s = Dropout(dropout)(s)

        # Merge temporal representation + static representation
        merged = Concatenate()([x, s])       # (batch, d_model + 32)
        merged = Dropout(dropout)(merged)
        merged = Dense(128, activation="relu")(merged)
        merged = Dropout(dropout)(merged)
        merged = Dense(64, activation="relu")(merged)
        merged = Dropout(dropout)(merged)
        outputs = Dense(1, activation="sigmoid")(merged)

    # -------------------------------------------------------------------------
    # Variant 2: TWO-STREAM (transformer on temporal, MLP on static, concatenate)
    # -------------------------------------------------------------------------
    elif columns is not None and len(columns) > 0:
        temporal_idx, static_idx = get_feature_indices(columns)

        # Temporal branch: transformer on event-level features
        temporal = GatherFeatures(temporal_idx)(inputs)  # (batch, seq, ~47)
        x = _build_encoder(
            temporal, xshape1, d_model, num_heads, ff_dim, num_layers, dropout
        )

        if pooling == "center":
            x = CenterTokenPooling()(x)
        elif pooling == "last":
            x = LastTokenPooling()(x)
        else:
            x = MaskedGlobalAveragePooling()([x, PaddingMask()(temporal)])

        # Static branch: MLP on repo-level features
        static = ExtractStatic(static_idx)(inputs)  # (batch, ~388)
        s = LayerNormalization(epsilon=1e-6)(static)
        s = Dense(128, activation="relu")(s)
        s = Dropout(dropout)(s)
        s = Dense(64, activation="relu")(s)
        s = Dropout(dropout)(s)

        # Merge both branches
        merged = Concatenate()([x, s])       # (batch, d_model + 64)
        merged = Dropout(dropout)(merged)
        merged = Dense(64, activation="relu")(merged)
        merged = Dropout(dropout)(merged)
        outputs = Dense(1, activation="sigmoid")(merged)

    # -------------------------------------------------------------------------
    # Variant 1: SINGLE-STREAM (all 435 features together) — BEST PERFORMER
    # -------------------------------------------------------------------------
    else:
        # All features processed together through the transformer encoder
        x = _build_encoder(
            inputs, xshape1, d_model, num_heads, ff_dim, num_layers, dropout,
        )

        # Pool sequence to single vector
        if pooling == "center":
            x = CenterTokenPooling()(x)     # For symmetric windows
        elif pooling == "last":
            x = LastTokenPooling()(x)        # For causal windows
        else:
            x = MaskedGlobalAveragePooling()([x, PaddingMask()(inputs)])

        # Classification head: 3-layer MLP with dropout
        x = Dropout(dropout)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(dropout)(x)
        outputs = Dense(1, activation="sigmoid")(x)  # Vuln probability

    model = Model(inputs=inputs, outputs=outputs)

    # Default optimizer: AdamW with weight decay for regularisation
    # AdamW decouples weight decay from the gradient update, which works
    # better than L2 regularisation with adaptive optimisers.
    if optimizer is None:
        optimizer = AdamW(
            learning_rate=1e-4,   # Conservative LR (warmup handles the ramp)
            weight_decay=1e-4,    # Mild weight decay for regularisation
        )

    model.compile(
        loss="binary_crossentropy",   # Standard loss for binary classification
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model
