import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(indicator, panels, lf = lambda x: x):
    f, subplots = plt.subplots(len(panels))
    for ix, panel in enumerate(panels):
        values = panel.ix[:,:,indicator].values
        values = values.flatten().astype(float)
        values = values[~np.isnan(values)]
        values = lf(values)
        print indicator
        print 'min: ', values.min(), ' max: ', values.max(), "median: ", np.median(values)
        subplots[ix].hist(values, bins=100, normed=True)
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def gaussian(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def visualize_confidence(panel, from_year=2016):
    f, subplots = plt.subplots(2,2)
    f.tight_layout()
    # f.set_size_inches(12,8)
    ti = ['GDP growth (annual %)','Bank nonperforming loans to total gross loans (%)','Interest rate spread (lending rate minus deposit rate, %)','Real interest rate (%)']
    for indicator, subplot in zip(ti, subplots.flatten()):
        print indicator
        ts = panel['2000':, 'Bulgaria', indicator]
        std = np.std(ts)
        ci = [0 for _ in range(15)] + [i for i in range(1,13)] 
        random_noise = np.tile(np.linspace(-std,std,120), (27,1)).T*np.array(ci)
        seaborn.tsplot(data=preds['2000':, 'Bulgaria', indicator].values+random_noise, ci=[68,95], time=range(2000,2027), ax=subplot)
        subplot.set_title(indicator)
    plt.savefig('all_together')
