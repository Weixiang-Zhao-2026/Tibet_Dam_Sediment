"""Run the full analysis and draw all figures.  Usage:  python run_all.py
Runtime: about 1 min."""
import runpy, os, time, importlib.util
# Check that required packages are installed before running anything
import importlib
_req = {'numpy':'numpy','pandas':'pandas','scipy':'scipy','statsmodels':'statsmodels','matplotlib':'matplotlib',
        'PIL':'pillow','xlrd':'xlrd','openpyxl':'openpyxl','mpl_toolkits.basemap_data':'basemap-data'}
_missing = [pip for mod, pip in _req.items() if importlib.util.find_spec(mod.split('.')[0]) is None
            or (mod == 'mpl_toolkits.basemap_data' and importlib.util.find_spec(mod) is None)]
if _missing:
    raise SystemExit('Missing packages: ' + ' '.join(_missing) + '\nInstall with:  pip install ' + ' '.join(_missing))
print('All required packages found.')

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
STEPS = ['s01_load_yangtze', 's02_plateau_pathways', 's03_plateau_Is_anomaly',
         's04_downstream_budgets', 's05_capacity_scoping',
         's06_subreach_tests', 's07_plateau_diagnostics', 's08_downstream_structural',
         's09_lambda_bootstrap_crossperiod', 's10_indus_identifiability',
         's11_plateau_reach_budgets', 's12_table_B1',
         'f01_study_domain', 'f02_framework_lambda', 'f03_downstream', 'f04_f06_plateau_figures']
for s in STEPS:
    t0 = time.time(); print(f'--> {s}', flush=True)
    runpy.run_path(os.path.join(SRC, s + '.py'), run_name='__main__')
    print(f'    done in {time.time()-t0:.0f} s', flush=True)
print('All outputs written to results/ and figures/.')
