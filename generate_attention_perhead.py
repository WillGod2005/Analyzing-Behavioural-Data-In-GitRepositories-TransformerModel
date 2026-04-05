"""Generate per-head attention heatmaps showing head specialisation."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

sys.path.insert(0, '.')
from models import CenterTokenPooling, LastTokenPooling, PaddingMask
from dataset_utils import extract_dataset, pad_and_fix, split_repos, split_into_x_and_y, Aggregate

MODEL_PATH = "models/Aggregate.before_cve_R1_B20_meta/transformer.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
    'CenterTokenPooling': CenterTokenPooling, 'LastTokenPooling': LastTokenPooling, 'PaddingMask': PaddingMask,
})
seq_len = model.input_shape[1]

np.random.seed(42); tf.random.set_seed(42); random.seed(42); os.environ['PYTHONHASHSEED']='42'
all_repos, _, columns = extract_dataset(aggr_options=Aggregate.before_cve, backs=20, metadata=True, cache=True)
all_repos, num_vulns = pad_and_fix(all_repos)
_, test_repos, _ = split_repos(all_repos, int(0.8*num_vulns)+int(0.1*num_vulns))
X_test, y_test = split_into_x_and_y(test_repos)

attn_layers = [l for l in model.layers if 'multi_head_attention' in l.name]
ln_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.LayerNormalization)]

preds = model.predict(X_test, verbose=0).flatten()
pred_binary = (preds > 0.5).astype(int)
correct_vuln = np.where((y_test == 1) & (pred_binary == 1))[0]
correct_benign = np.where((y_test == 0) & (pred_binary == 0))[0]

def compute_attention(model, x_batch, attn_layer, pre_ln_layer):
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

n = min(50, len(correct_vuln), len(correct_benign))
print(f"Computing attention for {n} samples per class...")

vuln_attn = compute_attention(model, X_test[correct_vuln[:n]], attn_layers[0], ln_layers[0])
benign_attn = compute_attention(model, X_test[correct_benign[:n]], attn_layers[0], ln_layers[0])

center = seq_len // 2
num_heads = vuln_attn.shape[1]

# --- Per-head difference plot ---
fig, axes = plt.subplots(2, 4, figsize=(20, 9))

for h in range(num_heads):
    vuln_h = vuln_attn[:, h, :, :].mean(axis=0)
    benign_h = benign_attn[:, h, :, :].mean(axis=0)

    # Top row: per-head vulnerable attention
    ax_top = axes[0, h]
    im1 = ax_top.imshow(vuln_h, cmap='YlOrRd', aspect='auto', vmin=0)
    ax_top.set_title(f'Head {h+1} — Vulnerable', fontsize=11)
    ax_top.axhline(y=center, color='white', linestyle='--', alpha=0.6, linewidth=0.8)
    ax_top.axvline(x=center, color='white', linestyle='--', alpha=0.6, linewidth=0.8)
    if h == 0:
        ax_top.set_ylabel('Query position')
    plt.colorbar(im1, ax=ax_top, fraction=0.046, pad=0.04)

    # Bottom row: per-head difference
    diff_h = vuln_h - benign_h
    vmax = max(abs(diff_h.min()), abs(diff_h.max()))
    ax_bot = axes[1, h]
    im2 = ax_bot.imshow(diff_h, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax_bot.set_title(f'Head {h+1} — Difference', fontsize=11)
    ax_bot.axhline(y=center, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_bot.axvline(x=center, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    ax_bot.set_xlabel('Key position')
    if h == 0:
        ax_bot.set_ylabel('Query position')
    plt.colorbar(im2, ax=ax_bot, fraction=0.046, pad=0.04)

plt.suptitle('Per-Head Attention Patterns (Layer 1, B=20 Symmetric, Seed 42)\n'
             'Top: Vulnerable samples | Bottom: Difference (Vulnerable − Benign, red = more attention for vulnerable)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('figures/attention_perhead.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved to figures/attention_perhead.png")

# Print per-head statistics
print(f"\n--- Per-Head Analysis (attention FROM target, row {center}) ---")
for h in range(num_heads):
    vuln_h = vuln_attn[:, h, :, :].mean(axis=0)
    benign_h = benign_attn[:, h, :, :].mean(axis=0)
    v_from = vuln_h[center, :]
    b_from = benign_h[center, :]
    diff = v_from - b_from

    # Entropy of attention distribution
    v_entropy = -np.sum(v_from * np.log(v_from + 1e-10))
    b_entropy = -np.sum(b_from * np.log(b_from + 1e-10))
    max_entropy = np.log(seq_len)

    print(f"\nHead {h+1}:")
    print(f"  Vuln entropy: {v_entropy:.3f} / {max_entropy:.3f} ({v_entropy/max_entropy*100:.1f}% of max)")
    print(f"  Benign entropy: {b_entropy:.3f} / {max_entropy:.3f} ({b_entropy/max_entropy*100:.1f}% of max)")

    # Top attended positions for vulnerable
    top_v = np.argsort(v_from)[::-1][:3]
    print(f"  Vuln top-3 attended: {[(f'pos {p} (t{p-center:+d}): {v_from[p]:.4f}') for p in top_v]}")

    # Largest differences
    top_d = np.argsort(np.abs(diff))[::-1][:3]
    print(f"  Largest diffs: {[(f'pos {p} (t{p-center:+d}): {diff[p]:+.4f}') for p in top_d]}")

    # Check if head is "local" (concentrates near target) or "global" (spread out)
    nearby = v_from[max(0,center-3):center+4].sum()
    print(f"  Attention within ±3 of target: vuln={nearby:.3f}, benign={b_from[max(0,center-3):center+4].sum():.3f}")
