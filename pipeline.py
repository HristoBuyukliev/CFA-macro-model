import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from normalize import Normalizer
from visualize import *
from helper_functions import *
from models import *
import seaborn


def run_pipeline():

    #get training data
    training_data = pd.read_csv('worldbank-data/WDI_Data.csv')
    training_data.set_index(['Country Name', 'Indicator Name'], inplace=True)

    #convert to panel
    panel = training_data.to_panel()
    panel.drop(['Indicator Code', 'Country Code'], axis=0, inplace=True)
    panel = panel.swapaxes(0, 1)

    indicators_to_use = [
        'Agriculture, value added (% of GDP)',
        'Industry, value added (% of GDP)',
        'Services, etc., value added (% of GDP)',
        'Domestic credit provided by financial sector (% of GDP)',
        'GDP growth (annual %)',
        'GDP (current US$)',
        'Expense (% of GDP)',
        'Inflation, consumer prices (annual %)',
        'Inflation, GDP deflator (annual %)',
        'Total debt service (% of exports of goods, services and primary income)',
        'Current account balance (BoP, current US$)',
        'External balance on goods and services (% of GDP)',
        'Health expenditure, total (% of GDP)',
        'Tax revenue (% of GDP)',
        'Gross capital formation (% of GDP)',
        'Gross savings (% of GDP)',
        'Net investment in nonfinancial assets (% of GDP)',
        'Bank capital to assets ratio (%)',
        'Bank nonperforming loans to total gross loans (%)',
        'Broad money (% of GDP)',
        'Commercial bank branches (per 100,000 adults)',
        'Deposit interest rate (%)',
        'Real interest rate (%)',
        'Risk premium on lending (lending rate minus treasury bill rate, %)',
        'Total reserves (includes gold, current US$)',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)',
        'Interest rate spread (lending rate minus deposit rate, %)'
        ]
    print len(indicators_to_use), 'indicators used'
    panel = panel[:,:,indicators_to_use]

    target_variables = [
        'Agriculture, value added (% of GDP)',
        'Industry, value added (% of GDP)',
        'Services, etc., value added (% of GDP)',
        'GDP growth (annual %)',
        'Inflation, GDP deflator (annual %)',
        'Gross capital formation (% of GDP)',
        'Gross savings (% of GDP)',
        'Bank capital to assets ratio (%)',
        'Bank nonperforming loans to total gross loans (%)',
        'Deposit interest rate (%)',
        'Real interest rate (%)',
        'Risk premium on lending (lending rate minus treasury bill rate, %)',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)',
        'Interest rate spread (lending rate minus deposit rate, %)'
    ]
    #drop useless countries such as samoa, lesoto and so on.
    useful_countries = []
    for country in panel.axes[0]:
        if find_null_percentage(panel[country,:,:]) < 0.7:
            useful_countries.append(country)
    panel = panel.ix[useful_countries,:,:]

    normalizer = Normalizer(panel)
    normalized_panel = normalizer.normalize(panel)

    # #visualize normalization:
    # for indicator in normalized_panel.axes[2]:
    #     plot_hist(indicator, [panel, normalized_panel])

    # select train data
    years_to_validate = 1
    years_to_predict  = 10
    years_train = generate_year_list(stop=2016-years_to_validate)
    years_val = generate_year_list(start=2016-years_to_validate+1)
    years_predict = generate_year_list(start=2017, stop=2016+years_to_predict)
    train_panel = normalized_panel[:, years_train, :].copy()


    # fill missing values:
    # either banal mean or median filling
    # or sampling with a generative bidirectional LSTM - see https://arxiv.org/abs/1306.1091

    generative_model = dense_generative_model(train_panel, hidden_layers=[120],epochs=100)
    sampled_filled_values = iterative_fill(generative_model, train_panel, normalizer, iterations=50, burn_in=10)
    train_panel.update(sampled_filled_values, overwrite=False)
    # or
    # train_panel.fillna(0, inplace=True)
    # or
    # train_panel = iterative_fill_bLSTM(train_panel)
    # or
    # filled_panel = fill_missing_bLSTM(train_panel, epochs=100)
    # train_panel.update(filled_panel, overwrite=False)
    # or
    # interpolate(train_panel)


    # create 1-step-ahead model
    epochs = 200
    hl = [100,100]
    print "ARCHITECTURE:", hl
    print 'EPOCHS:', epochs
    X_train = train_panel[:,years_train,:][:,:-1,:]
    y_train = train_panel[:,years_train,:][:,1:,:]
    model = dense_gradient_model(X_train, y_train, 
                                hidden_layers=hl, 
                                d=0.2, 
                                patience=50,
                                epochs=epochs)

    # finally, predict
    for start, year in enumerate(years_val+years_predict):
        predictions = model.predict(train_panel[:,start+1:,:].values)[:,-1,:]
        train_panel = train_panel.swapaxes(0,1)
        new_year_df = pd.DataFrame(data=predictions,index=train_panel.axes[1], columns=y_train.axes[2])
        train_panel[year] = new_year_df
        train_panel = train_panel.swapaxes(0,1)
    print "score:", rmse(normalized_panel[:,years_val,target_variables].values, 
                         train_panel[:,years_val,target_variables].values)

    #revert to original scale and distributions
    train_panel = normalizer.renormalize(train_panel)

    #convert to dataframe, and write relevant information to file
    target_countries = ['Bulgaria', 'Cyprus', 'Albania']
    train_panel = train_panel.swapaxes(0,1)
    df = train_panel[:,target_countries,target_variables].to_frame(filter_observations=False)
    df.to_csv('Predictions.csv')


if __name__ == "__main__":
    run_pipeline()