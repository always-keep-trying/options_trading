
import numpy as np
import pandas as pd


def prepare_data(full_data,data_ind_start, data_ind_end, horizon, look_back_lag, y_var_name='Z_Score'):
    assert any(full_data.isna().values.any(axis=0))==False, "There are NaNs in the data"
    input_data = full_data.iloc[data_ind_start:data_ind_end+1,:].copy()

    y_train = input_data.loc[:,y_var_name].values
    y_train = y_train.reshape(-1,1)
    x_train = input_data.iloc[:,1:].values

    true_Y = full_data.loc[full_data.index>max(input_data.index),:].head(horizon).loc[:,[y_var_name]]

    assert true_Y.shape[0] == horizon, "Y True length should match the horizon length"

    true_Y.loc[:,'pred'] = np.nan
    x_predict = input_data.iloc[-1,0:look_back_lag].values.reshape(1,-1)
    return x_train, y_train, x_predict, true_Y

def recursive_strategy(model, x_train, y_train, x_predict, true_Y, horizon, look_back_lag):
    # fit once
    model.fit(x_train, y_train)

    # predict 1st time
    y_predict = model.predict(x_predict)
    true_Y_copy = true_Y.copy()
    true_Y_copy.iloc[0,1] = y_predict
    # predict the next (horizon-1) times
    for i in range(1,horizon):
        x_predict = np.concatenate((y_predict[0],x_predict[0][0:(look_back_lag-1)])).reshape(1,-1)
        y_predict = model.predict(x_predict)
        true_Y_copy.iloc[i,1] = y_predict
    return true_Y_copy

def direct_recursive_strategy(model, x_train, y_train, x_predict, true_Y, horizon, look_back_lag):
    # fit each time and insert the prediction from
    model.fit(x_train, y_train)

    # predict first time
    y_predict = model.predict(x_predict)
    true_Y_copy = true_Y.copy()
    true_Y_copy.iloc[0,1] = y_predict
    # predict the next (horizon-1) times
    for i in range(1,horizon):
        import pdb
        pdb.set_trace()
        x_predict = np.concatenate((y_predict[0],x_predict[0][0:(look_back_lag-1)])).reshape(1,-1)

        # update train data
        x_train = np.concat([x_train,x_predict],axis=0)
        y_train = np.concat([y_train, y_predict], axis=0)
        # update model
        model.fit(x_train, y_train)
        y_predict = model.predict(x_predict)
        true_Y_copy.iloc[i,1] = y_predict
    return true_Y_copy
