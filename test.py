import numpy as np
import pandas as pd
from data_prepare import *
import matplotlib.pyplot as plt
from params_select import *
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval

# market = pd.read_csv("~/cs_market.csv", parse_dates=["date"], dtype={"code": str})
market = pd.read_csv("E:\market_data/cs_market.csv", parse_dates=["date"], dtype={"code": str})
all_ohlcv = market.drop(["Unnamed: 0", "total_turnover", "limit_up", "limit_down"], axis=1)
all_ohlcv = all_ohlcv.set_index('date')
ohlcv = all_ohlcv[all_ohlcv["code"] == "000725.XSHE"].drop("code", axis=1)

features = construct_features(ohlcv, construct_features1)

features_categorical = to_categorical(features.copy(), 'label', categorical_func_factory)

space = default_space
objective_func = construct_objective(features_categorical)
trials = Trials()
best = fmin(objective_func, space, algo=tpe.suggest, max_evals=40, trials=trials)
print(best)



