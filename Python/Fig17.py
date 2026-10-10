import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
from scipy.signal import medfilt
from scipy.stats import linregress

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PR_FILE           = r"PR_Ex12.txt"
MIN_STABLE_SAMPLES = 20      # minimum consecutive samples to count as a plateau
STABLE_TOL         = 0.01   # tolerance as a fraction of total pressure range

# Font sizes (matching project convention)
FONT_SIZE_AXIS  = 28   # xlabel / ylabel
FONT_SIZE_TICK  = 22   # tick labels
# ══════════════════════════════════════════════════════════════════════════════

# ── Font setup (Times New Roman, bold – same as main.py) ────────────────────
try:
    plt.rcParams['font.family']       = 'serif'
    plt.rcParams['font.serif']        = ['Times New Roman']
    plt.rcParams['font.size']         = FONT_SIZE_TICK
    plt.rcParams['font.weight']       = 'bold'
    plt.rcParams['axes.labelweight']  = 'bold'
    plt.rcParams['xtick.labelsize']   = FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize']   = FONT_SIZE_TICK
    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['mathtext.rm']       = 'Times New Roman'
    plt.rcParams['mathtext.it']       = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf']       = 'Times New Roman:bold'
    plt.rcParams['mathtext.default']  = 'bf'
    findfont(FontProperties(family=['Times New Roman']), fallback_to_default=False)
    print("SUCCESS: 'Times New Roman' font found.")
except Exception:
    print("WARNING: 'Times New Roman' not found – a fallback font will be used.")

# ── Read file ────────────────────────────────────────────────────────────────
with open(PR_FILE, 'r') as f:
    lines = f.readlines()

shots = 0
start_idx = 0
for i, line in enumerate(lines):
    if "SHOTS" in line:
        try:
            shots = int(line.split(":")[1].strip())
        except Exception:
            pass
    if "Pressure" in line:
        start_idx = i + 1
        break

p_raw = []
for line in lines[start_idx:]:
    s = line.strip()
    if s:
        try:
            p_raw.append(float(s))
        except Exception:
            pass

p_raw = np.array(p_raw)

# ── Filter to valid pressure range & smooth ──────────────────────────────────
p_filtered = p_raw[(p_raw >= 0.2) & (p_raw <= 10.0)]
p_smooth   = medfilt(p_filtered, kernel_size=31)

# ── Plateau detection (same algorithm as test_pressure.py) ──────────────────
total_range = np.ptp(p_smooth)
tol         = STABLE_TOL * total_range

plateaus = []   # list of (start_idx, end_idx, median_value)
i = 0
while i < len(p_smooth):
    ref = p_smooth[i]
    j   = i
    while j < len(p_smooth) and abs(p_smooth[j] - ref) <= tol:
        j += 1
    run_len = j - i
    if run_len >= MIN_STABLE_SAMPLES:
        median_val = np.median(p_smooth[i:j])
        if plateaus and abs(median_val - plateaus[-1][2]) <= tol * 2:
            plateaus[-1] = (plateaus[-1][0], j - 1,
                            np.median(p_smooth[plateaus[-1][0]:j]))
        else:
            plateaus.append((i, j - 1, median_val))
        i = j
    else:
        i += 1

shot_pressures = np.array([p[2] for p in plateaus])[:shots]
shot_numbers   = np.arange(1, len(shot_pressures) + 1)

print(f"Extracted {len(shot_pressures)} shot pressures from '{PR_FILE}'")

# ── Linear trendline ─────────────────────────────────────────────────────────
slope, intercept, r, *_ = linregress(shot_numbers, shot_pressures)
r2     = r ** 2
x_fit  = np.linspace(shot_numbers[0], shot_numbers[-1], 300)
y_fit  = slope * x_fit + intercept
print(f"Trendline: slope={slope:.4f}, intercept={intercept:.4f}, R²={r2:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot
ax.scatter(shot_numbers, shot_pressures,
           color='steelblue', s=80, zorder=3,
           edgecolors='navy', linewidths=1.2,
           label='Measured pressure')

# Trendline only (no equation label)
ax.plot(x_fit, y_fit,
        color='crimson', linewidth=2, linestyle='--',
        label='Trendline')

ax.set_xlabel(r'Shot',
              fontsize=FONT_SIZE_AXIS,
              fontname='Times New Roman',
              fontweight='bold')
ax.set_ylabel(r'$\mathit{P}_0\ \mathrm{(Torr)}$',
              fontsize=FONT_SIZE_AXIS,
              fontname='Times New Roman',
              fontweight='bold')

ax.set_xticks(shot_numbers)

fig.tight_layout()
plt.show()
