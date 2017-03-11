import pandas as pd
import numpy as np

class Normalizer():
    def __init__(self, panel):
        self.transform_dict = {}
        for indicator in panel.axes[2]:
            values = panel.ix[:,:,indicator].values
            values = values.flatten().astype(float)
            values = values[~np.isnan(values)]
 
            if values.max() > 1000 or values.min() < -1000:
                self.transform_dict[indicator] = {'type' : 'sign log'}
                new_values = np.log(np.abs(values)+1) * np.sign(values)
            else:
                new_values = values
                self.transform_dict[indicator] = {'type' : 'none'}
            mu = new_values.mean()
            sigma = new_values.std()
            self.transform_dict[indicator]['mu'] = mu
            self.transform_dict[indicator]['sigma'] = sigma

    def normalize(self, panel):
        new_panel = panel.copy()
        for indicator in new_panel.axes[2]:
            new_values = new_panel.ix[:,:, indicator]
            if self.transform_dict[indicator]['type'] == 'sign log':
                new_values = np.log(np.abs(new_values)+1) * np.sign(new_values)
            new_values -= self.transform_dict[indicator]['mu']
            new_values /= self.transform_dict[indicator]['sigma']
            new_panel.ix[:,:,indicator] = new_values
        return new_panel

    def renormalize(self, panel):
        new_panel = panel.copy()
        for indicator in new_panel.axes[2]:
            new_values = new_panel.ix[:,:, indicator]
            new_values *= self.transform_dict[indicator]['sigma']
            new_values += self.transform_dict[indicator]['mu']
            if self.transform_dict[indicator]['type'] == 'sign log':
                new_values = (np.exp(np.abs(new_values)) - 1) * np.sign(new_values)
            new_panel.ix[:,:,indicator] = new_values
        return new_panel
