# Parameter Sensitivity Sweep — TPF-2 Signal-Recovery Chain

This repository contains the code used to perform a parameter sensitivity sweep on the signal-recovery processing chain for the TPF-2 plasma physics data. 

The primary goal of this analysis is to demonstrate that the processing steps—specifically the Savitzky-Golay filtering, positive signal clipping, and numerical integration procedures—**do not distort the underlying plasma physics parameters** (e.g., pinch time, peak current, and pinch duration).

---

## Key Findings

The parameter sweep involves systematically varying the signal processing parameters across 75 combinations per shot for all 20 shots:
- **Savitzky-Golay polynomial order** (2, 3, 4)
- **Pulse-segment window width** (5, 7, 11, 15, 21 samples)
- **Positive clipping threshold** (20% to 60% of baseline)

**Results:**
- The induced variation in $t_{pinch}$ due to data processing is on average **0.83 ns** (maximum 5.2 ns), which is negligible compared to the shot-to-shot physical variation of **424.6 ns**.
- The variation in $I_{peak}$ is on average **0.22 %** (maximum 0.43 %), significantly smaller than the physical shot-to-shot variation of **1.90 %**.
- As validated mathematically, clipping threshold variations (20–60%) strictly affect only the positive overshoot of the signal, leaving the physical negative dip defining the pinch completely unaffected.
- An independent verification using raw, unfiltered data to locate the pinch time yields consistent results (within −0.07 ± 1.21 ns) for 17 out of 20 shots. The remaining 3 shots highlight the necessity of the filter, as the raw-data estimator erroneously locks onto the subsequent 28.6 ns period of the Rogowski coil ringing.

All extracted values are independently cross-checked against the analytical model $I_0 = V_0 \sqrt{C_0 / L_0}$, yielding agreement within **0.41 %**.

---

## Requirements

Ensure you have the following Python packages installed:
```bash
pip install numpy scipy matplotlib pandas
```

---

## How to Run the Analysis

To execute the entire sensitivity analysis pipeline from start to finish, run the main wrapper script:

```bash
python main.py
```

This script will sequentially execute:
1. **Parameter Sweep**: Runs the processing grid across all combinations and all 20 shots. (Takes ~20 seconds).
2. **Figure Generation**: Generates comprehensive visualizations for the supplementary material (`FigS1` to `FigS7`).
3. **Verification Suite**: Runs 45 automated assertions to prove that the extracted quantities are physically sound and that the signal processing behaves exactly as mathematically defined.

*(Note: Data reading of `PD_Ex12.txt` is cached as `PD_Ex12_shots.npy` during the first run. Subsequent runs will load in < 1 second.)*

---

## File Structure

| File | Description |
|---|---|
| `main.py` | Wrapper script to execute the entire pipeline sequentially. |
| `pipeline.py` | Core signal-processing algorithm containing the filter, clipping, and integration logic. |
| `run_sweep.py` | Executes the parameter grid sweeps (Grid A, B, C) over all 20 shots and saves raw results to CSV. |
| `make_figures.py` | Visualizes the sensitivity analysis and outputs to `figures/*.png` and `*.pdf`. |
| `verify.py` | A comprehensive 45-check verification suite ensuring robustness and algorithm consistency. |
| `results/*.csv` | Directory for the generated raw result data. |
| `figures/` | Directory containing all generated plots. |

## Important Definitions

- **$t_{pinch}$**: The time at which `baseline − signal` reaches its maximum within the 2–4 µs window. The baseline is a linear fit to the pre-smoothed signal in the same window.
- **Pinch duration**: The Full-Width at Half-Maximum (FWHM) of the current dip, determined via linear interpolation (decoupled from the 80 ps sampling grid).
- **$I_{peak}$**: The maximum value of the integrated current over the entire record.
- **Coil self-resonance**: The peak of the FFT (Hann window) of the raw signal over the interval $t_{pinch} + 0.05$ to $t_{pinch} + 0.45$ µs, evaluated above 10 MHz.
