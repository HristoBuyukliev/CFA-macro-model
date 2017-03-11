import pandas as pd
import numpy as np


def generate_year_list(start = 1960, stop=2016):    
    yrs = []
    for yr in range(start, stop+1):
        yrs.append("{0}".format(yr))
    return yrs

def find_null_percentage(panel):
    null = np.sum(panel.isnull().values)
    return null/float(panel.values.size)
        
def rmse(y_true, y_pred):
    output_errors = (y_true - y_pred) ** 2
    non_nan_terms = np.count_nonzero(~np.isnan(output_errors))
    output_errors = np.nan_to_num(output_errors)
    return (np.sum(output_errors)/non_nan_terms)**0.5

def add_years_to_panel(panel, years):
    panel.swap_axes

#assumes country-year-indicator axes
def interpolate(panel):
    for country in panel.axes[0]:
        df = panel[country]
        df.ix['1960'].fillna(0, inplace=True)
        df = df.interpolate()
        panel[country] = df