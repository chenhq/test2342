from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, rand, space_eval
from keras.initializers import Orthogonal, glorot_uniform, he_uniform, lecun_uniform
from data_prepare import *
from model import *
from log_history import *

default_space = {
    'time_steps': hp.choice('time_steps', [8, 16, 32]),
    'batch_size': hp.choice('batch_size', [2, 4, 8]),
    'epochs': hp.choice('epochs', [200, 500, 1000, 1500, 2000]),

    'units1': hp.choice('units1', [8, 16, 32, 64]),
    'units2': hp.choice('units2', [8, 16, 32, 64]),
    'units3': hp.choice('units3', [8, 16, 32, 64]),

    'is_BN_1': hp.choice('is_BN_1', [False, True]),
    'is_BN_2': hp.choice('is_BN_2', [False, True]),
    'is_BN_3': hp.choice('is_BN_3', [False, True]),

    'lr': hp.uniform('lr', 0.0001, 0.001),
    'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'recurrent_dropout': hp.choice('recurrent_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'initializer': hp.choice('initializer', [glorot_uniform(seed=123)])
}


def construct_objective(data):
    def objective(params):
        print(params)
        train, validate, test = split_data_set(data, params['batch_size'] * params['time_steps'], 3, 1, 1)
        X_train, Y_train = reform_X_Y(train, params['batch_size'], params['time_steps'])
        X_validate, Y_validate = reform_X_Y(validate, params['batch_size'], params['time_steps'])
        X_test, Y_test = reform_X_Y(test,params['batch_size'], params['time_steps'])
        model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1])
        history = LogHistory()
        model.fit(X_train, Y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  verbose=0,
                  validation_data=(X_validate, Y_validate),
                  shuffle=False,
                  callbacks=[history])
        history.loss_plot('epoch')

        loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=params['batch_size'])
        # profit_pred, _, _ = utils.profit(model.predict(X_validate), cls_validate)
        # utils.plot_curve(result_file, Y_validate, model.predict(X_validate), cls_validate, 0)
        # utils.plot_curve(result_file, Y_test, model.predict(X_test), cls_test, 1)
        # return {'loss': -profit_pred[-1], 'status': STATUS_OK}
        return {'loss': loss_and_metrics[0], 'status': STATUS_OK}
    return objective


# space = default_space
# objective_func = construct_objective(features_categorical)
# trials = Trials()
# best = fmin(objective_func, space, algo=tpe.suggest, max_evals=40, trials=trials)
# print(best)