"""Generate attention weight heatmaps from saved transformer models."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from models import CenterTokenPooling, LastTokenPooling, PaddingMask
from dataset_utils import extract_dataset, pad_and_fix, split_repos, split_into_x_and_y, Aggregate

# Load model - B=20 symmetric seed 42
MODEL_PATH = "models/Aggregate.before_cve_R1_B20_meta/transformer.keras"
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
    'CenterTokenPooling': CenterTokenPooling,
    'LastTokenPooling': LastTokenPooling,
    'PaddingMask': PaddingMask,
})
seq_len = model.input_shape[1]
print(f"Model loaded. seq_len={seq_len}")

# Load data FROM CACHE
print("Loading dataset from cache...")
np.random.seed(42)
tf.random.set_seed(42)
import random
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

all_repos, _, columns = extract_dataset(
    aggr_options=Aggregate.before_cve, backs=20, metadata=True, cache=True
)
all_repos, num_vulns = pad_and_fix(all_repos)
train_size = int(0.8 * num_vulns)
val_size = int(0.1 * num_vulns)
_, test_repos, _ = split_repos(all_repos, train_size + val_size)
X_test, y_test = split_into_x_and_y(test_repos)
print(f"Test: {X_test.shape[0]} samples, shape: {X_test.shape}")

# Find attention layers and layernorm layers
attn_layers = [l for l in model.layers if 'multi_head_attention' in l.name]
ln_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.LayerNormalization)]
print(f"Attention layers: {[l.name for l in attn_layers]}")

# Get predictions
preds = model.predict(X_test, verbose=0).flatten()
pred_binary = (preds > 0.5).astype(int)
correct_vuln = np.where((y_test == 1) & (pred_binary == 1))[0]
correct_benign = np.where((y_test == 0) & (pred_binary == 0))[0]
print(f"Correct vuln: {len(correct_vuln)}, Correct benign: {len(correct_benign)}")

def compute_attention(model, x_batch, attn_layer, pre_ln_layer):
    """Compute attention weights from MHA layer weights."""
    sub_model = tf.keras.Model(inputs=model.input, outputs=pre_ln_layer.output)
    ln_out = sub_model.predict(x_batch, verbose=0)
    weights = attn_layer.get_weights()
    q_kernel, q_bias = weights[0], weights[1]
    k_kernel, k_bias = weights[2], weights[3]
    Q = np.einsum('bsi,ihk->bshk', ln_out, q_kernel) + q_bias
    K = np.einsum('bsi,ihk->bshk', ln_out, k_kernel) + k_bias
    key_dim = q_kernel.shape[-1]
    scores = np.einsum('bshk,bthk->bhst', Q, K) / np.sqrt(key_dim)
    exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return attn  # (batch, heads, seq, seq)

# Compute attention for both classes
n = min(50, len(correct_vuln), len(correct_benign))
print(f"Computing attention for {n} samples per class...")

vuln_attn = compute_attention(model, X_test[correct_vuln[:n]], attn_layers[0], ln_layers[0])
benign_attn = compute_attention(model, X_test[correct_benign[:n]], attn_layers[0], ln_layers[0])

vuln_avg = vuln_attn.mean(axis=(0, 1))
benign_avg = benign_attn.mean(axis=(0, 1))
center = seq_len // 2

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for ax, data, title in [
    (axes[0], vuln_avg, '(a) Correctly Classified Vulnerable'),
    (axes[1], benign_avg, '(b) Correctly Classified Benign'),
]:
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_title(f'{title}\n(averaged over {n} samples, 4 heads)', fontsize=11)
    ax.set_xlabel('Key position (attending to)')
    ax.set_ylabel('Query position (attending from)')
    ax.axhline(y=center, color='white', linestyle='--', alpha=0.7)
    ax.axvline(x=center, color='white', linestyle='--', alpha=0.7)
    ax.text(center, -1.8, 'target', ha='center', fontsize=8, color='red', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

diff = vuln_avg - benign_avg
vmax = max(abs(diff.min()), abs(diff.max()))
im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
axes[2].set_title('(c) Difference (Vulnerable - Benign)\n(red = more attention for vulnerable)', fontsize=11)
axes[2].set_xlabel('Key position (attending to)')
axes[2].set_ylabel('Query position (attending from)')
axes[2].axhline(y=center, color='black', linestyle='--', alpha=0.5)
axes[2].axvline(x=center, color='black', linestyle='--', alpha=0.5)
axes[2].text(center, -1.8, 'target', ha='center', fontsize=8, color='red', fontweight='bold')
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(f'Transformer Attention Patterns (Layer 1, B=20 Symmetric, Seed 42)\n'
             f'Sequence length = {seq_len}, centre position = target commit', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('figures/attention_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nSaved to figures/attention_heatmap.png")

# Print statistics
print(f"\n--- Attention from target position (row {center}) ---")
v_from = vuln_avg[center, :]
b_from = benign_avg[center, :]
print(f"Vulnerable - top 5 positions attended to:")
for pos in np.argsort(v_from)[::-1][:5]:
    print(f"  pos {pos} (target{pos-center:+d}): {v_from[pos]:.4f}")
print(f"Benign - top 5 positions attended to:")
for pos in np.argsort(b_from)[::-1][:5]:
    print(f"  pos {pos} (target{pos-center:+d}): {b_from[pos]:.4f}")

print(f"\n--- Attention TO target position (column {center}) ---")
v_to = vuln_avg[:, center]
b_to = benign_avg[:, center]
print(f"Vulnerable mean: {v_to.mean():.4f}, Benign mean: {b_to.mean():.4f}")

diff_from = v_from - b_from
print(f"\nLargest attention differences FROM target:")
for pos in np.argsort(np.abs(diff_from))[::-1][:5]:
    print(f"  pos {pos} (target{pos-center:+d}): diff={diff_from[pos]:+.4f}")
