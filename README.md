# TINT TPF-2: raw data and analysis code

Data and code supporting the manuscript *Integrated Control and Power System Engineering for Thailand Plasma Focus-2*.

- **Authors:** Pattarachai Bor-in, Arlee Tamman, Pasit Wonghabut, Sarai Lekchaum, Kitsakorn Locharoenrat
- **Contact:** Kitsakorn Locharoenrat, kitsakorn.lo@kmitl.ac.th

## Layout

```
MATLAB/
  Experiment.zip             Oscilloscope records: Experiment/DAY <m-d-yyyy>/ExNN/
  Time_FFT_SNR_Analysis.mlx
  Time_FFT_Overlay.mlx
  Coil_Current_Analysis.mlx
  PF_Axial_Analysis.mlx
  Workspace.mat              dI/dt records, 1 MΩ and 50 Ω terminations
  Workspace2.mat             Integrated current, 20-shot sequence
Python/
  PD_Ex12.txt                Raw dI/dt, 20-shot sequence (PXIe-5185)
  PR_Ex12.txt                Chamber pressure, same sequence
  Fig17.py                   Pressure per shot
  DataPulseAnalyzer.py       Signal-processing chain, single shot
  SNR_Analysis.py            5 V-referenced interference ratio and record screening
  snr_results/               Screening output
  TPF2_Signal_Sensitivity_Analysis/
    pipeline.py, run_sweep.py, make_figures.py, verify.py, main.py
    results/                 Sweep output
    figures/                 Created by make_figures.py
```

## Data

### Oscilloscope records (`MATLAB/Experiment.zip`)
Tektronix MSO44, 6.25 GS/s, 125 000 points, −2 to 18 µs, triggered on CH1 at 1.04 V.
- Each record has `DataLog_XXXX_Analog.csv` (waveforms) and `DataLog_XXXX_HardCopy.png` (screen capture).
- Each `ExNN` folder also has `DataLog_Log.txt` (logger events) and, in most folders, `DataLog_Measurements.csv` (oscilloscope max/min measurements).

| Column | Channel | Quantity | Unit |
|---|---|---|---|
| 1 | — | time | s |
| 2 | CH1 | V_coil, Rogowski-coil output (∝ dI/dt) | V |
| 3 | CH2 | I_DG, diagnostic-ground current (Pearson 2877, 1 V/A) | A |
| 4 | CH3 | V_A0, microcontroller analog input A0 (HVPS) | V |
| 5 | CH4 | V_A1, microcontroller analog input A1 (MFC) | V |

| Folder | Configuration | Used in |
|---|---|---|
| DAY 2-24-2025 | before modification | Table S6.3 |
| DAY 3-13-2025 | before modification | Table S6.3; Figs 13(a), 23(a) (Ex03/DataLog_0008) |
| DAY 3-17-2025 | before modification | Figs 14(a), 24(a) (Ex09/DataLog_0000) |
| DAY 4-8-2025 | after modification | Figs 14(b), 24(b), S7.1 (Ex22/DataLog_0000) |
| DAY 4-9-2025 | after modification | Table S6.3; Figs 13(b), 23(b) (Ex02/DataLog_0000) |
| DAY 4-10-2025 | after modification | Table S6.3 |

### 20-shot sequence (`Python/PD_Ex12.txt`, `Python/PR_Ex12.txt`)
NI PXIe-5185, 12.5 GS/s, 50 Ω input, 20 consecutive shots at 11 kV.
- `PD_Ex12.txt`: header, then a single column of dI/dt samples (V): 20 shots × 375 000 samples, in shot order.
- `PR_Ex12.txt`: same header, then a single column of chamber pressure samples (Torr) logged during the sequence.

Header fields:
- **Voltage SV:** dimensionless set-point code that the control software sends to the HVPS through the PWM-to-analog interface. It is not a voltage. The value 12 corresponds to an 11 kV charging voltage.
- **Charger time, Trigger WAIT, PS WAIT:** timing settings of the automated firing cycle.

### Screening output (`Python/snr_results/`)
- `snr_per_shot.csv`: one row per record.
  - Columns: day, file, `status` (used / not selected / rejected), `reject_reason`, ratio in dB for CH3 (V_A0) and CH4 (V_A1), V_rms and V_pp.
  - Rows for DAY 4-8-2025 are not used in Table S6.3 (see Supplementary Section S6).
- `snr_summary.csv`: per-day mean, sample standard deviation and n.

Settings: window −2 to 18 µs; first 20 accepted records per day.
A record is rejected if it is flat-topped (≥ 5 samples at ≥ 95 % of full scale), if its probe attenuation differs from `1X,1X,10X,10X`, or if its record length differs from 125 000.

## Figures and tables

| Item | Data | Script |
|---|---|---|
| Figs 13, 23 | DAY 3-13-2025, DAY 4-9-2025 | `MATLAB/Time_FFT_SNR_Analysis.mlx` |
| Figs 14, 24, S7.1 | DAY 3-17-2025, DAY 4-8-2025 | `MATLAB/Time_FFT_Overlay.mlx` |
| Fig 17 | `Python/PR_Ex12.txt` | `Python/Fig17.py` |
| Figs 18, 19 | `Python/PD_Ex12.txt` | `TPF2_Signal_Sensitivity_Analysis/pipeline.py` (nominal settings) |
| Fig 20 | `MATLAB/Workspace.mat` | `MATLAB/Coil_Current_Analysis.mlx` |
| Figs 21, 22 | `Python/PD_Ex12.txt` | `Python/DataPulseAnalyzer.py` |
| Fig 27, Table 3 | `MATLAB/Workspace2.mat` | `MATLAB/PF_Axial_Analysis.mlx` |
| Table S5.1 | `TPF2_Signal_Sensitivity_Analysis/results/` | `run_sweep.py` |
| Fig S5.1 | `Python/PD_Ex12.txt` (shot 12) | `make_figures.py` → `figures/FigS6_waveform_overlay` |
| Table S6.3 | `MATLAB/DAY*` | `Python/SNR_Analysis.py` → `snr_results/` |

`make_figures.py` also writes supplementary diagnostic figures that do not appear in the article.
Lee Model Code results (Figs 15, 25, 26) are not included; the model inputs are listed in Table 2 of the article.

## Processing settings (nominal)
- Savitzky–Golay: initial window 51 samples; order 3
- Segment windows: pulse 11 samples; flank 1001 samples
- Pulse detection: hysteresis 50/70 %
- Positive clipping: 40 % of the baseline deviation
- Integration: cumulative trapezoidal, initial value 0, M = 314 pH, × 8 conductors

## Requirements
- **Python ≥ 3.9:** numpy, scipy, matplotlib, pandas
- **MATLAB ≥ R2021b:** Signal Processing Toolbox

```
cd Python/TPF2_Signal_Sensitivity_Analysis && python main.py
cd Python && python Fig17.py
cd MATLAB && python ../Python/SNR_Analysis.py --root . --select 20 --window=-2e-6,18e-6
```

Before running, extract `MATLAB/Experiment.zip` and move the six `DAY*` folders from `Experiment/` into `MATLAB/`. Run the MATLAB Live Scripts from `MATLAB/`.
