import os.path

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

STYLE_FP = os.path.join(os.path.dirname(__file__), 'style.mplstyle')
