import numpy as np
import pandas as pd
from random import sample
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Lambda, GaussianNoise, merge, SimpleRNN
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from visualize import plot_loss


def sparse_gradient_model(X_train, y_train, y_train_nans, epochs=5000):
    # corrected_outputs sum
    #     masked_outputs mult
    #         unmasked_outputs -> noisy_inputs -> lstm -> main_input
    #         non_nans (non_nans)
    #     unfilled_targets mult
    #         target_inputs
    #         nans
    print X_train.shape, y_train.shape, y_train_nans.shape
    y_train_non_nans = 1 - y_train_nans
    n_samples, n_timesteps, n_feat = X_train.shape

    #assist inputs
    non_nans      = Input(shape=(n_timesteps, 7), name='is y non nan')
    target_inputs = Input(shape=(n_timesteps, 7), name='unfilled targets')
    nans          = Input(shape=(n_timesteps, 7), name='is y nan')

    #core model
    main_input    = Input(shape=(n_timesteps, n_feat), name='main_input')
    lstm = Bidirectional(LSTM(90, dropout_W = 0.2, dropout_U = 0.3, return_sequences=True))(main_input)
    lstm2 = Bidirectional(LSTM(40, dropout_W = 0.2, dropout_U = 0.1, return_sequences=True))(lstm)
    lstm_merge = merge([main_input, lstm2], mode='concat')
    unmasked_outputs = TimeDistributed(Dense(7))(lstm_merge)

    #bonus shenanigans
    masked_outputs   = merge([unmasked_outputs, non_nans], mode='mul')
    unfilled_targets = merge([target_inputs, nans], mode = 'mul')
    corrected_outputs = merge([masked_outputs, unfilled_targets], mode='sum')
    unmasked_model = Model(input=main_input, output=unmasked_outputs)
    masked_model   = Model(input=[main_input, non_nans, target_inputs, nans], output=corrected_outputs)
    unmasked_model.compile(optimizer='rmsprop', loss='mse')
    masked_model.compile(optimizer='rmsprop', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    history = masked_model.fit([X_train, y_train_non_nans, y_train, y_train_nans], y_train, nb_epoch=epochs, 
        validation_split=0.1, callbacks = [early_stopping])
    return history, unmasked_model

def corrupt_generator(panel, years_ahead=0, indicators='all', include_y_nans=False, corrupt_years=5):
    while True:
        new_panel = panel.copy()
        years_to_corrupt = sample(new_panel.axes[1], 5)
        new_panel.ix[:,years_to_corrupt,:] = np.nan
        len_years = panel.shape[1]
        X = panel.fillna(0)[:,:len_years-years_ahead,:].values
        if indicators == 'all':
            y = panel.fillna(0)[:,years_ahead:,:].values
            y_nan = np.isnan(panel[:,years_ahead:,:].values)
        else:
            y = panel.fillna(0)[:,years_ahead:,target_indicators].values
            y_nan = np.isnan(panel[:,years_ahead:,target_indicators].values)
        if include_y_nans:
            yield (X, (y, y_nan))
        else:
            yield (X, y)

def dense_gradient_model(X_panel, y_panel, epochs=5000, hidden_layers = [120], d=0.2, patience=30):
    X = X_panel.fillna(0).values
    y = y_panel.fillna(0).values
    n_samples, n_timesteps, n_feat = X.shape
    main_input = Input(shape=(n_timesteps, n_feat), name='main_input')
    layers = [main_input]
    for hl in hidden_layers:
        layers.append(Bidirectional(LSTM(hl, 
                                        return_sequences=True, 
                                        dropout_W = d)
                                    )(layers[-1]))
    final_layer = Bidirectional(LSTM(y.shape[-1], 
                                return_sequences=True, 
                                dropout_W = d, 
                                dropout_U = d), 
                            merge_mode='sum')(layers[-1])
    outputs = merge([final_layer, main_input], mode='sum')
    model = Model(input=main_input, output = outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    early_stopping = EarlyStopping(patience=patience)
    history = model.fit(X, y, nb_epoch = epochs, validation_split=0.1, callbacks=[early_stopping], verbose=0)
    return model



def dense_generative_model(X_panel, epochs=5000, hidden_layers = [120], d=0.2):
    X = X_panel.fillna(0).values
    n_samples, n_timesteps, n_feat = X.shape
    main_input = Input(shape=(n_timesteps, n_feat), name='main_input')
    layers = [main_input]
    for hl in hidden_layers:
        layers.append(Bidirectional(LSTM(hl, return_sequences=True, dropout_W = d, dropout_U = d, name='first'))(layers[-1]))
    outputs = Bidirectional(LSTM(X.shape[-1], return_sequences=True, dropout_W = d, dropout_U = d, name='second'), merge_mode='sum')(layers[-1])
    model = Model(input=main_input, output = outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    early_stopping = EarlyStopping(patience=20)
    generator = corrupt_generator(X_panel[:189,:,:], years_ahead = 0)
    valid_generator = corrupt_generator(X_panel[189:,:,:], years_ahead = 0)
    history = model.fit_generator(corrupt_generator(X_panel), samples_per_epoch=210, nb_epoch = epochs, validation_data=valid_generator, nb_val_samples=22, callbacks=[early_stopping])
        #(X, y, nb_epoch=epochs, validation_split=0.1, callbacks=[early_stopping])
    return model

def sparse_generative_model(X_panel, epochs=5000, hidden_layers = [120], d=0.2):
    # corrected_outputs sum
    #     masked_outputs mult
    #         unmasked_outputs -> noisy_inputs -> lstm -> main_input
    #         non_nans (non_nans)
    #     unfilled_targets mult
    #         target_inputs
    #         nans
    n_samples, n_timesteps, n_feat = X_panel.shape

    #assist inputs
    non_nans      = Input(shape=(n_timesteps, n_feat), name='is y non nan')

    #core model
    main_input = Input(shape=(n_timesteps, n_feat), name='main_input')
    layers = [main_input]
    for hl in hidden_layers:
        layers.append(Bidirectional(LSTM(hl, return_sequences=True, dropout_W = d, dropout_U = d, name='first'))(layers[-1]))
    outputs = Bidirectional(LSTM(X.shape[-1], return_sequences=True, dropout_W = d, dropout_U = d, name='second'), merge_mode='sum')(layers[-1])
    unmasked_model = Model(input=main_input, output = outputs)

    #bonus shenanigans
    masked_outputs   = merge([unmasked_outputs, non_nans], mode='mul')
    masked_model   = Model(input=[main_input, non_nans, target_inputs, nans], output=masked_outputs)
    unmasked_model.compile(optimizer='rmsprop', loss='mse')
    masked_model.compile(optimizer='rmsprop', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    history = masked_model.fit([X_train, y_train_non_nans, y_train, y_train_nans], y_train, nb_epoch=epochs, 
        validation_split=0.1, callbacks = [early_stopping])
    return history, unmasked_model

def parsimonious_model(X_train, y_train, epochs=5000):
    n_samples, n_timesteps, n_feat = X_train.shape
    shape = X_train.shape
    print X_train.shape, y_train.shape
    main_input = Input(shape=(n_timesteps, n_feat), name='main_input')
    lstms_1 = [Bidirectional(LSTM(30, dropout_W = 0.2, dropout_U = 0.2, return_sequences=True))(main_input) for ln in range(6)]
    lstms_2 = [Bidirectional(LSTM(15, dropout_W = 0.2, dropout_U = 0.2, return_sequences=True))(lstm_1) for lstm_1 in lstms_1]
    merge_l = merge(lstms_2, mode='concat')
    outputs = TimeDistributed(Dense(7), name='output dense layer')(merge_l)
    model = Model(input=main_input, output=outputs)
    print model.count_params()
    model.compile(optimizer='rmsprop', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(X_train, y_train, nb_epoch=epochs, 
        validation_split=0.1, callbacks = [early_stopping])
    return history, model

#iterative sampling
def iterative_fill(model, panel, normalizer, iterations=1000, noise=1.0, burn_in=200):
    gibbs_panel = panel.fillna(0).copy()
    scores_averaged = []
    averaged_panel = panel.fillna(0).copy()/iterations
    for iteration in range(burn_in):
        X_train = gibbs_panel.values
        X_train += np.random.normal(0, noise, X_train.shape)
        X_train = model.predict(X_train)
        overwritten_panel = pd.Panel(data=X_train, items=panel.axes[0], major_axis=panel.axes[1], minor_axis=panel.axes[2])
        gibbs_panel = panel.copy()
        gibbs_panel.update(overwritten_panel, overwrite=False)

    for iteration in range(iterations):
        X_train = gibbs_panel.values
        X_train += np.random.normal(0, noise, X_train.shape)
        X_train = model.predict(X_train)
        overwritten_panel = pd.Panel(data=X_train, items=panel.axes[0], major_axis=panel.axes[1], minor_axis=panel.axes[2])
        gibbs_panel = panel.copy()
        gibbs_panel.update(overwritten_panel, overwrite=False)        
        averaged_panel = averaged_panel.add(gibbs_panel/(iterations))

    return averaged_panel

def parsimonious_generative_model(panel, epochs=5000):
    X_train = panel.fillna(0).values
    n_samples, n_timesteps, n_feat = X_train.shape
    main_input = Input(shape=(n_timesteps, n_feat), name='main_input')
    lstms_1 = [Bidirectional(LSTM(20, dropout_W = 0.1, dropout_U = 0.1, return_sequences=True))(main_input) for ln in range(4)]
    lstms_2 = [Bidirectional(LSTM(10, dropout_W = 0.1, dropout_U = 0.1, return_sequences=True))(lstm_1) for lstm_1 in lstms_1]
    lstms_3 = [Bidirectional(LSTM(20, dropout_W = 0.1, dropout_U = 0.1, return_sequences=True))(lstm_2) for lstm_2 in lstms_2]    
    merge_l = merge(lstms_3, mode='concat')
    outputs = TimeDistributed(Dense(n_feat))(merge_l)
    model = Model(input=main_input, output = outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    print model.count_params()
    early_stopping = EarlyStopping(patience=20)
    history = model.fit(X_train, X_train, nb_epoch=epochs, validation_split=0.1, callbacks=[early_stopping])
    return model
    


def fill_missing_bLSTM(panel, epochs=100):
    non_nans = 1-np.isnan(panel.values)
    X_train = panel.fillna(0).values
    n_samples, n_timesteps, n_feat = X_train.shape
    main_input = Input(shape=(n_timesteps, n_feat), name='main_input')
    lstm = Bidirectional(LSTM(120, return_sequences=True))(main_input)
    unmasked_outputs = TimeDistributed(Dense(n_feat))(lstm)
    bool_input = Input(shape=(n_timesteps, n_feat), name='isnan_inputs')
    masked_outputs = merge([unmasked_outputs, bool_input], mode='mul')
    model = Model(input=[main_input, bool_input], output=masked_outputs)
    print model.count_params()
    unmasked_model = Model(input=main_input, output = unmasked_outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    early_stopping = EarlyStopping(patience=20)
    history = model.fit([X_train, non_nans], X_train, nb_epoch=epochs, validation_split=0.1, callbacks=[early_stopping])
    plot_loss(history)
    unmasked_model.compile(optimizer='rmsprop', loss='mae')
    X_train = unmasked_model.predict(X_train)
    fpanel = pd.Panel(data=X_train, items=panel.axes[0], major_axis=panel.axes[1], minor_axis=panel.axes[2])
    return fpanel

def iterative_fill_bLSTM(panel, epochs=5000, iterations=5):
    nans = np.isnan(panel.values) + 0
    ians = 1-nans
    X_train = panel.fillna(0).values
    for iteration in range(iterations):
        n_samples, n_timesteps, n_feat = X_train.shape
        main_input = Input(shape=(n_timesteps, n_feat), name='main_input')
        lstm = Bidirectional(LSTM(120, dropout_W = 0.5, dropout_U = 0.2, return_sequences=True))(main_input)
        # lstm2 = Bidirectional(LSTM(60, dropout_W = 0.5, dropout_U = 0.2, return_sequences=True))(lstm)
        unmasked_outputs = TimeDistributed(Dense(n_feat))(lstm)
        nan_input = Input(shape=(n_timesteps, n_feat), name='is nan inputs')
        ian_input = Input(shape=(n_timesteps, n_feat), name='is a num inputs')
        masked_outputs = merge([unmasked_outputs, ian_input], mode='mul')
        only_original_outputs = merge([main_input, nan_input], mode='mul')
        final_output = merge([masked_outputs, only_original_outputs], mode='sum')
        model = Model(input=[main_input, ian_input, nan_input], output=final_output)
        model.compile(optimizer='rmsprop', loss='mse')
        early_stopping = EarlyStopping(patience=20)
        history = model.fit([X_train, ians, nans], X_train, nb_epoch=epochs, validation_split=0.1, callbacks=[early_stopping])
        plot_loss(history)

        unmasked_model = Model(input=main_input, output = unmasked_outputs)
        unmasked_model.compile(optimizer='rmsprop', loss='mse')
        X_train = unmasked_model.predict(X_train)
    fpanel = pd.Panel(data=X_train, items=panel.axes[0], major_axis=panel.axes[1], minor_axis=panel.axes[2])
    return fpanel

