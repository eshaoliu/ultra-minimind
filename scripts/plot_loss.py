"""Parse training log and plot loss curves.

Usage: python scripts/plot_loss.py [log_glob] [out_png]
Defaults: out/pretrain_*.log -> out/loss_curve.png
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_glob = sys.argv[1] if len(sys.argv) > 1 else 'out/pretrain_*.log'
out_png = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(log_glob.replace('*', 'all'))[0] + '.png'
log_files = sorted(glob.glob(log_glob))
if not log_files:
    sys.exit('no log file found for ' + log_glob)

pattern = re.compile(
    r'Epoch:\[(\d+)/\d+\]\((\d+)/\d+\), loss: ([\d.]+), logits_loss: ([\d.]+),.*?lr: ([\d.e-]+)'
)

steps, losses, logits_losses, lrs = [], [], [], []
for log_file in log_files:
    with open(log_file, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch, step, loss, logits_loss, lr = m.groups()
                steps.append((int(epoch) - 1) * 10**6 + int(step))
                losses.append(float(loss))
                logits_losses.append(float(logits_loss))
                lrs.append(float(lr))

if not steps:
    sys.exit('no loss records yet (training still preprocessing)')

# steps restart at 1 each epoch; use cumulative index for a continuous curve
x = range(1, len(steps) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(x, losses, label='loss', color='tab:red')
ax1.plot(x, logits_losses, label='logits_loss', color='tab:blue', alpha=0.7)
ax1.set_ylabel('loss')
run_name = os.path.basename(log_glob).split('_*')[0]
ax1.set_title(f'{run_name} loss ({len(steps)} records, latest={losses[-1]:.4f})')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(x, lrs, color='tab:green')
ax2.set_ylabel('lr')
ax2.set_xlabel('log record (#)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_png, dpi=120)
print(f'saved {out_png}, {len(steps)} records, latest loss={losses[-1]:.4f}')
