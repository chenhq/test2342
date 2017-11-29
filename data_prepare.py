import numpy as np
import pandas as pd
from talib.abstract import *
import random


def construct_features(ohlcv, construct_features_func):
    return construct_features_func(ohlcv)


def construct_features1(ohlcv):
    data = pd.DataFrame(index=ohlcv.index)
    data['close'] = ohlcv['close']
    data['open_close'] = (ohlcv['open'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['high_close'] = (ohlcv['high'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['low_close'] = (ohlcv['low'] - ohlcv['close']) * 100 / ohlcv['close'].shift(1)
    data['pct_chg'] = ((ohlcv['close'] / ohlcv['close'].shift(1)) - 1) * 100
    data = data.fillna(0)

    next_ma5 = SMA(ohlcv, timeperiod=5).shift(-5)
    data['label'] = 1
    data.loc[data['close'] < next_ma5, 'label'] = 0
    data.loc[data['close'] > next_ma5, 'label'] = 2

    ma15_volume = ohlcv['volume'].rolling(15).mean()
    data['volume'] = ohlcv['volume'] / ma15_volume
    data['volume'] = data['volume'].fillna(1)

    return data


def categorical_func_factory(num_class):
    def categorical_func(x):
        cls = []
        for i in range(num_class):
            if i == x:
                cls.append(1.0)
            else:
                cls.append(0.0)
        return cls
    return categorical_func


def to_categorical(data, column, func):
    num_class = len(data[column].unique())
    data[column] = data[column].copy().map(func(num_class))
    return data


def data_split_by_idx(data, idxes):
    pass


def split_by_sample(data, split_dict, minimum_size):
    result = {}
    length = len(data)
    all = 0
    for key in split_dict:
        all += split_dict[key]

    num_batch = int(length / minimum_size)
    batch_location = list(range(num_batch))
    random.shuffle(batch_location)

    for key in split_dict:
        selected_num_batch = int(num_batch * split_dict[key] / all)
        selected_batch_location = batch_location[-selected_num_batch:]
        del batch_location[-selected_num_batch:]
        selected_location = [range(minimum_size * i, minimum_size * (i + 1)) for i in selected_batch_location]
        selected_location = np.array(selected_location).reshape(-1)
        selected_location.sort()
        result[key] = data.iloc[selected_location]
    return result


def reform_X_Y(data, batch_size, timesteps, target_field='label'):
    size = len(data)
    if size % (int(batch_size) * int(timesteps)) != 0:
        print("data size not match, size: {0}, batch_size: {1}, timesteps: {2}".format(size, batch_size, timesteps))
        return None, None

    columns = data.columns.tolist()
    if target_field not in columns:
        return None, None
    x_columns = columns
    x_columns.remove(target_field)
    X, Y0 = data.loc[:, x_columns].values, data.loc[:, target_field].values
    X = X.reshape((-1, timesteps, X.shape[1]))
    Y = np.array([np.array(y) for y in Y0])
    Y = Y.reshape((-1, timesteps, Y.shape[1]))
    return X, Y


def split_data_set(data, minimum_size=128, train_ratio=3, test_ratio=1, validate_ratio=1):
    result = split_by_sample(data, {"train": train_ratio, "validate": validate_ratio, "test": test_ratio}, minimum_size)
    train = result['train']
    validate = result['validate']
    test = result['test']

    return train, validate, test