# TINT TPF-2: raw data and analysis code

Data and code supporting the manuscript *Integrated Control and Power System Engineering for Thailand Plasma Focus-2* (Scientific Reports, under review).

- **Authors:** Pattarachai Bor-in, Arlee Tamman, Pasit Wonghabut, Sarai Lekchaum, Kitsakorn Locharoenrat
- **Contact:** Kitsakorn Locharoenrat, kitsakorn.lo@kmitl.ac.th

## Repository layout

```
LICENSE                              MIT licence (code)
README.md
MATLAB/
  ExperimentData/                    Oscilloscope records, one folder per measurement day
    DAY <m-d-yyyy>/
      ExNN/                          One acquisition run
        DataLog_XXXX_Analog.csv      Waveforms of one record (4 channels)
        DataLog_XXXX_HardCopy.png    Screen capture of the same record
        DataLog_Log.txt              Logger events (most runs)
        DataLog_Measurements.csv     Oscilloscope max/min measurements (most runs)
  Time_FFT_SNR_Analysis.mlx          Figs 13, 23: time traces, FFT and interference ratio of one record
  Time_FFT_Overlay.mlx               Figs 14, 24, S7.1: before/after overlays
  Coil_Current_Analysis.mlx          Fig 20: 1 MΩ vs 50 Ω termination
  PF_Axial_Analysis.mlx              Fig 27, Table 3: axial-phase analysis
  Workspace.mat                      dI/dt records, 1 MΩ and 50 Ω terminations
  Workspace2.mat                     Integrated current, 20-shot sequence
Python/
  PD_Ex12.txt                        Raw dI/dt, 20-shot sequence (PXIe-5185)
  PR_Ex12.txt                        Chamber pressure, same sequence
  Fig17.py                           Fig 17: pressure per shot
  DataPulseAnalyzer.py               Figs 21, 22: signal-processing chain, single shot
  SNR_Analysis.py                    5 V-referenced interference ratio and record screening
  snr_results/
    snr_per_shot.csv                 One row per record, with screening status and reason
    snr_summary.csv                  Per-day mean, sample standard deviation and n
  TPF2_Signal_Sensitivity_Analysis/
    main.py                          Runs the sweep, figures and checks in sequence
    pipeline.py                      Processing chain used for Figs 18, 19
    run_sweep.py                     Parameter sweep (Table S5.1)
    make_figures.py                  Sweep figures (Fig S5.1)
    verify.py                        Consistency checks
    figstyle.py                      Plot style
    results/                         Sweep output
    figures/                         Created by make_figures.py
```

## Data

### Oscilloscope records (`MATLAB/ExperimentData/`)
Tektronix MSO44, 6.25 GS/s, 125 000 points, −2 to 18 µs, triggered on CH1 at 1.04 V.

Columns of `DataLog_XXXX_Analog.csv`:

| Column | Channel | Quantity | Unit |
|---|---|---|---|
| 1 | — | time | s |
| 2 | CH1 | V_coil, Rogowski-coil output (∝ dI/dt) | V |
| 3 | CH2 | I_DG, diagnostic-ground current (Pearson 2877, 1 V/A) | A |
| 4 | CH3 | V_A0, microcontroller analog input A0 (HVPS) | V |
| 5 | CH4 | V_A1, microcontroller analog input A1 (MFC) | V |

| Folder | Configuration | Runs | Records | Used in |
|---|---|---|---|---|
| DAY 2-24-2025 | before modification | 7 | 27 | Table S6.3 |
| DAY 3-13-2025 | before modification | 4 | 25 | Table S6.3; Figs 13(a), 23(a) (Ex03/DataLog_0008) |
| DAY 3-17-2025 | before modification | 13 | 58 | Figs 14(a), 24(a) (Ex09/DataLog_0000) |
| DAY 4-8-2025 | after modification | 38 | 275 | Figs 14(b), 24(b), S7.1 (Ex22/DataLog_0000) |
| DAY 4-9-2025 | after modification | 6 | 75 | Table S6.3; Figs 13(b), 23(b) (Ex02/DataLog_0000) |
| DAY 4-10-2025 | after modification | 25 | 157 | Table S6.3 |

Total: 93 runs, 617 records.

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
  - Covers DAY 2-24, 3-13, 4-8, 4-9 and 4-10. Rows for DAY 4-8-2025 are not used in Table S6.3 (see Supplementary Section S6).
- `snr_summary.csv`: per-day mean, sample standard deviation and n.

Settings: window −2 to 18 µs; first 20 accepted records per day.
A record is rejected if it is flat-topped (≥ 5 samples at ≥ 95 % of full scale), if its probe attenuation differs from `1X,1X,10X,10X`, or if its record length differs from 125 000.

## Figures and tables

| Item | Data | Script |
|---|---|---|
| Figs 13, 23 | `ExperimentData/DAY 3-13-2025`, `DAY 4-9-2025` | `MATLAB/Time_FFT_SNR_Analysis.mlx` |
| Figs 14, 24, S7.1 | `ExperimentData/DAY 3-13-2025`, `DAY 4-9-2025`, `DAY 3-17-2025`, `DAY 4-8-2025` | `MATLAB/Time_FFT_Overlay.mlx` |
| Fig 17 | `Python/PR_Ex12.txt` | `Python/Fig17.py` |
| Figs 18, 19 | `Python/PD_Ex12.txt` | `TPF2_Signal_Sensitivity_Analysis/pipeline.py` (nominal settings) |
| Fig 20 | `MATLAB/Workspace.mat` | `MATLAB/Coil_Current_Analysis.mlx` |
| Figs 21, 22 | `Python/PD_Ex12.txt` | `Python/DataPulseAnalyzer.py` |
| Fig 27, Table 3 | `MATLAB/Workspace2.mat` | `MATLAB/PF_Axial_Analysis.mlx` |
| Table S5.1 | `TPF2_Signal_Sensitivity_Analysis/results/` | `run_sweep.py` |
| Fig S5.1 | `Python/PD_Ex12.txt` (shot 12) | `make_figures.py` → `figures/FigS6_waveform_overlay` |
| Table S6.3 | `MATLAB/ExperimentData/` | `Python/SNR_Analysis.py` → `snr_results/` |

`make_figures.py` also writes supplementary diagnostic figures that do not appear in the article.
Lee Model Code results (Figs 15, 25, 26) are not included. They were produced with the third-party RADPF V5.16 package (S. Lee); the model inputs are listed in Tables 1 and 2 of the article.

## Processing settings (nominal)
- Savitzky–Golay: initial window 51 samples; order 3
- Segment windows: pulse 11 samples; flank 1001 samples
- Pulse detection: hysteresis 50/70 %
- Positive clipping: 40 % of the baseline deviation
- Integration: cumulative trapezoidal, initial value 0, M = 314 pH, × 8 conductors

## Requirements
- **Python ≥ 3.9:** numpy, scipy, matplotlib, pandas
- **MATLAB ≥ R2021b:** Signal Processing Toolbox

## Running

Python scripts are run from their own folder, because they open the data files by relative path:

```
cd Python
python Fig17.py
python DataPulseAnalyzer.py
python SNR_Analysis.py --root ../MATLAB/ExperimentData --select 20 --window=-2e-6,18e-6

cd TPF2_Signal_Sensitivity_Analysis
python main.py
```

- `SNR_Analysis.py` writes to `Python/snr_results/` and replaces the files supplied there. Run over the whole folder, it also processes DAY 3-17-2025, which is not part of the supplied `snr_results/`; use `--day "<folder>" --append` to process selected days only.
- The first run of the sweep caches `PD_Ex12.txt` as `PD_Ex12_shots.npy`.

MATLAB Live Scripts are opened and run from `MATLAB/`. `Time_FFT_SNR_Analysis.mlx` and `Time_FFT_Overlay.mlx` read the records from `MATLAB/ExperimentData/`.

## Licence
- Code (`.py`, `.mlx`): MIT License, see `LICENSE`.
- Data (oscilloscope records, `PD_Ex12.txt`, `PR_Ex12.txt`, `.mat` files and result tables): CC BY 4.0.

## Citation
Please cite the article above and the archived version of this repository on Zenodo.
