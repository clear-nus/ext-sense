from functools import partial

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import confusion_matrix


from skorch import NeuralNetRegressor, NeuralNetClassifier

import torch
import torch.nn as nn

import numpy as np
import json

torch.manual_seed(100)
np.random.seed(100)

def evaluate(experiment_name, device='cuda:2'):
    
    with open('evaluators.json') as f:
        all_evaluators = json.load(f)
    
    experiments = {}
    
    my_evaluators = {
        '_evaluate_tool_svm':_evaluate_tool_svm,
        '_evaluate_tool_mlp':_evaluate_tool_mlp,
        '_evaluate_tool_rnn':_evaluate_tool_rnn,
        '_evaluate_tool_neusingle_svm': _evaluate_tool_neusingle_svm,
        '_evaluate_classifier_svm':_evaluate_classifier_svm,
        '_evaluate_classifier_mlp':_evaluate_classifier_mlp,
        '_evaluate_classifier_rnn':_evaluate_classifier_rnn,
    }
    
    for s in all_evaluators:
        experiments[s['function_name']] = partial(my_evaluators[s['evaluator']], **s['args'])
    
    if experiment_name not in experiments: raise Exception('Experiment not found')
                
    test_loss_mean, test_loss_std = experiments[experiment_name]()
    
    print('Result for {:s}: {:0.4f} ± {:0.4f}'.format(experiment_name, test_loss_mean, test_loss_std))

    
class RNNModule(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(RNNModule, self).__init__()
        
        self.rnn = nn.GRU(input_dim, 16, batch_first=True)
        self.linear1 = nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, output_dim)

    def forward(self, X):
        
        X, _ = self.rnn(X)
        X = torch.squeeze(X[:, -1, :])
        X = torch.relu(self.linear1(X))
        X = self.linear2(X)
        
        return X

#    ______          _             _                 
#   |  ____|        | |           | |                
#   | |____   ____ _| |_   _  __ _| |_ ___  _ __ ___ 
#   |  __\ \ / / _` | | | | |/ _` | __/ _ \| '__/ __|
#   | |___\ V / (_| | | |_| | (_| | || (_) | |  \__ \
#   |______\_/ \__,_|_|\__,_|\__,_|\__\___/|_|  |___/
#                                                    


def _create_evaluator(estimator, param_grid, scoring, cv=4, N=5, callback=None, cm_name=None):
    
    gs_estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=3, refit=True)
    
    def evaluate(X, y, verbose=True):
        
        test_losses = np.zeros(N)
        
        cms = []
        for n in range(N):
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=n)
            gs_estimator.fit(X_train, y_train)
            test_loss = -gs_estimator.score(X_test, y_test)
            test_losses[n] = test_loss
            
            if callback is not None: callback(gs_estimator, X_test, y_test)
            if verbose: print('Iteration {:d} | Test Loss = {:0.4f}'.format(n, test_loss))

            if cm_name is not None:
                cms.append( confusion_matrix(y_test, gs_estimator.predict(X_test)) )

        import pickle
        pickle.dump(cms, open(f'{cm_name}.pkl' ,'wb'))

        return np.mean(test_losses), np.std(test_losses)

    return evaluate


#    _____                                            ______                _   _                 
#   |  __ \                                          |  ____|              | | (_)                
#   | |__) |___  __ _ _ __ ___  ___ ___  ___  _ __   | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
#   |  _  // _ \/ _` | '__/ _ \/ __/ __|/ _ \| '__|  |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#   | | \ \  __/ (_| | | |  __/\__ \__ \ (_) | |     | |  | |_| | | | | (__| |_| | (_) | | | \__ \
#   |_|  \_\___|\__, |_|  \___||___/___/\___/|_|     |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                __/ |                                                                            


def _load_data(task, tool_type, frequency, transformation, signal_type):
    
    data_dir = f'data/convoluted/kernel_{task}_{tool_type}_{frequency}.npz'
    
    npzfile = np.load(data_dir)
    
    if signal_type == 'neuhalf':
        # left sensor
        X = np.concatenate( [npzfile['signals'][:, : , 0:40], npzfile['signals'][:, : , 80:120]], 2)
        y = npzfile['labels'] * 100
    elif signal_type == 'all':
        X = npzfile['signals']
        y = npzfile['labels'] * 100
        
    if transformation == 'default':
        X = np.reshape(X, (X.shape[0], -1))
        y = y.ravel()

    if transformation == 'tensor':
        X = torch.Tensor( X )
        y = torch.Tensor( np.reshape(y, (-1, 1)) )
        
    if transformation == 'single':
        X = npzfile['signals']
        y = npzfile['labels'] * 100
        X = np.reshape(X, (X.shape[0], X.shape[1], -1, 80 ))
        X = np.swapaxes(X, 1, 3)
        X = np.reshape(X, (X.shape[0], 80, -1))
        y = y.ravel()

    return X, y


def _evaluate_tool_svm(task, tool_type, frequency, signal_type, kernel):

    X, y = _load_data(task, tool_type, frequency=frequency, transformation='default', signal_type=signal_type)
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    
    estimator = SVR(kernel=kernel, max_iter=5000)
    evaluate = _create_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_tool_mlp(task, tool_type, frequency, signal_type):

    X, y = _load_data(task, tool_type, frequency=frequency, transformation='default', signal_type=signal_type)
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }
    
    estimator = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=100)
    evaluate = _create_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_tool_rnn(task, tool_type, frequency, signal_type, device):
    
    X, y = _load_data(task, tool_type, frequency=frequency, transformation='tensor', signal_type=signal_type)
    #X = X.to(device)
    #y = y.to(device)
    
    param_grid = { 'lr': [0.001, 0.003, 0.01] }
    
    estimator = NeuralNetRegressor(RNNModule,
                                   module__input_dim=X.shape[2],
                                   module__output_dim=1,
                                   iterator_train__shuffle=True,
                                   max_epochs=1000,
                                   train_split=False,
                                   device=device,
                                   verbose=0)
    
    evaluate = _create_evaluator(estimator,
                                 param_grid,
                                 'neg_mean_absolute_error',
                                 ShuffleSplit(n_splits=1, test_size=.25))
    
    return evaluate(X, y)



def _evaluate_tool_neusingle_svm(task, tool_type, frequency, signal_type, kernel):
    
    X, y = _load_data(task, tool_type, frequency=frequency, transformation='single', signal_type=signal_type)


    best_taxel = 0
    best_test_loss_mean = float('inf')
    best_test_loss_std = float('inf')
    
    with open(f'results/nuskin_sing_{task}_{tool_type}_{frequency}_{kernel}.csv', 'w') as file:
    
        for taxel in range(1, 81):
            
            param_grid = { 'C': [1, 3, 10, 30, 100] }
    
            estimator = SVR(kernel=kernel, max_iter=5000)
            evaluate = _create_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
            test_loss_mean, test_loss_std = evaluate(X[:, taxel-1, :], y)
            file.write(f'{taxel},{test_loss_mean},{test_loss_std}\n')

            if test_loss_mean < best_test_loss_mean:

                best_taxel = taxel
                best_test_loss_mean = test_loss_mean
                best_test_loss_std = test_loss_std

            print('Result for taxel {:02d}: {:0.4f} ± {:0.4f}'.format(taxel, test_loss_mean, test_loss_std), flush=True)
    
    print(f'Best performing taxel is {best_taxel}')
    
    return best_test_loss_mean, best_test_loss_std


#     _____ _               _  __ _           _   _                ______                _   _                 
#    / ____| |             (_)/ _(_)         | | (_)              |  ____|              | | (_)                
#   | |    | | __ _ ___ ___ _| |_ _  ___ __ _| |_ _  ___  _ __    | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
#   | |    | |/ _` / __/ __| |  _| |/ __/ _` | __| |/ _ \| '_ \   |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#   | |____| | (_| \__ \__ \ | | | | (_| (_| | |_| | (_) | | | |  | |  | |_| | | | | (__| |_| | (_) | | | \__ \
#    \_____|_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_|  |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                                                                                                              


def _load_classifier_data(task, tool_type, frequency, transformation, signal_type):
    
    data_dir = f'data/convoluted/kernel_{task}_{tool_type}_{frequency}.npz'
    npzfile = np.load(data_dir)
    
    
    if signal_type == 'neuhalf':
        # left sensor
        X = np.concatenate( [npzfile['signals'][:, : , 0:40], npzfile['signals'][:, : , 80:120]], 2)
        y = npzfile['labels']
    elif signal_type == 'all':
        X = npzfile['signals']
        y = npzfile['labels'] 
        
        
    if transformation == 'default':
        X = np.reshape(X, (X.shape[0], -1))
        y = y.ravel()

    if transformation == 'tensor':
        X = torch.Tensor( X )
        y = torch.Tensor( np.reshape(y, (-1)) )
        y = y.type(torch.LongTensor)
        
    if transformation == 'single':
        X = npzfile['signals']
        y = npzfile['labels']
        X = np.reshape(X, (X.shape[0], X.shape[1], -1, 80 ))
        X = np.swapaxes(X, 1, 3)
        X = np.reshape(X, (X.shape[0], 80, -1))
        y = y.ravel()

    return X, y


def _evaluate_classifier_svm(task, tool_type, frequency, signal_type, kernel):

    X, y = _load_classifier_data(task, tool_type, frequency=frequency, transformation='default', signal_type=signal_type)
    
    param_grid = {
        'C': [0.1, 1, 3, 10, 30, 100, 200, 500]
    }

    kernel_name = 'svmlinear' if kernel == 'linear' else 'svmrbf'
    cm_name = f'results/food_nuskin_kernel_{kernel_name}_{frequency}'
    
    estimator = SVC(kernel=kernel, max_iter=5000)
    evaluate = _create_evaluator(estimator, param_grid, 'accuracy', N=20, cm_name=cm_name)
    
    return evaluate(X, y)


def _evaluate_classifier_mlp(task, tool_type, frequency, signal_type):

    X, y = _load_classifier_data(task, tool_type, frequency=frequency, transformation='default', signal_type=signal_type)
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }

    cm_name = f'results/food_nuskin_kernel_mlp_{frequency}'
    
    estimator = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=100)
    evaluate = _create_evaluator(estimator, param_grid, 'accuracy', N=20, cm_name=cm_name)
    
    return evaluate(X, y)


def _evaluate_classifier_rnn(task, tool_type, frequency, signal_type, device):

    X, y = _load_classifier_data(task, tool_type, frequency=frequency, transformation='tensor', signal_type=signal_type)
    
        
    param_grid = { 'lr': [0.001, 0.003, 0.01] }
    
    estimator = NeuralNetClassifier(RNNModule,
                                   module__input_dim=X.shape[2],
                                   module__output_dim=len(torch.unique(y)),
                                   criterion = nn.CrossEntropyLoss,
                                   iterator_train__shuffle=True,
                                   max_epochs=1000,
                                   train_split=False,
                                   device=device,
                                   verbose=0)

    cm_name = f'results/food_nuskin_kernel_rnn_{frequency}'
    
    evaluate = _create_evaluator(estimator,
                                 param_grid,
                                 'accuracy',
                                 ShuffleSplit(n_splits=1, test_size=.25),
                                 cm_name=cm_name)
    
    return evaluate(X, y)

#     _____ _      _____    _____       _             __               
#    / ____| |    |_   _|  |_   _|     | |           / _|              
#   | |    | |      | |      | |  _ __ | |_ ___ _ __| |_ __ _  ___ ___ 
#   | |    | |      | |      | | | '_ \| __/ _ \ '__|  _/ _` |/ __/ _ \
#   | |____| |____ _| |_    _| |_| | | | ||  __/ |  | || (_| | (_|  __/
#    \_____|______|_____|  |_____|_| |_|\__\___|_|  |_| \__,_|\___\___|
#                                                                      

if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) == 2: evaluate(sys.argv[1])
    if len(sys.argv) == 3: evaluate(sys.argv[1], sys.argv[2])
