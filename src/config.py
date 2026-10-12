"""Paths used by all scripts and notebooks."""
import os
_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_here)
RAW = os.path.join(ROOT, 'data', 'raw')
RES = os.path.join(ROOT, 'results')          # intermediate tables, json, npy
FIG = os.path.join(ROOT, 'figures', 'main')  # manuscript figures
FIGS = os.path.join(ROOT, 'figures', 'supplementary')
XLS_TP = os.path.join(RAW, 'tibetan_plateau_water_sediment_climate.xls')
XLS_YZ = os.path.join(RAW, 'yangtze_annual.xlsm')
for d in (RES, FIG, FIGS):
    os.makedirs(d, exist_ok=True)
os.chdir(RES)  # all intermediate files are read/written in results/
