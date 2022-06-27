import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from torch.optim import Adam
from models import MLP, NAC, NALU
import copy
import time

np.random.seed(42)
NUM_LAYERS = 4
HIDDEN_DIM = 8
FUNCS = ['relu6', 'softsign', 'tanh', 'sigmoid', 'selu', 'elu', 'relu', 'none']
ARITHMETIC_OPERATIONS = {
    'a+b': lambda x, y: x + y,
    'a-b': lambda x, y: x - y,
    'a*b': lambda x, y: x * y,
    'a/b': lambda x, y: x / y,
    'a^2': lambda x, y: x ** 2,
    'sqrt(a)': lambda x, y: x ** 0.5
}


def gen_data(size, func):
    X = np.random.permutation(size * 2).reshape(-1, 2).astype(float)
    X[X == 0] = size * 2
    y = func(X[:, 0], X[:, 1])
    return X, y


def train(model, X, y, epochs):
    learned_model = copy.deepcopy(model)
    optimizer = Adam(learned_model.parameters(), lr=0.001)
    for _ in range(epochs):
        predicted = learned_model(torch.tensor(X, dtype=torch.float)).flatten()
        loss = F.mse_loss(predicted, torch.tensor(y, dtype=torch.float))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return learned_model.to(device='cpu')


if __name__ == '__main__':
    models = [
        NALU(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1
        ),
        NAC(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1
        )
    ]
    models += [
        MLP(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
            activation=f
        ) for f in FUNCS
    ]

    results = {}
    for op_name, operation in ARITHMETIC_OPERATIONS.items():
        X, y = gen_data(size=1000, func=operation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

        train_mean_avg, test_mean_avg = [], []
        for _ in range(100):
            random_model = MLP(
                num_layers=NUM_LAYERS,
                in_dim=2,
                hidden_dim=HIDDEN_DIM,
                out_dim=1,
                activation='relu6'
            )
            y_train_predicted = random_model(torch.tensor(X_train, dtype=torch.float)).flatten().detach().numpy()
            y_test_predicted = random_model(torch.tensor(X_test, dtype=torch.float)).flatten().detach().numpy()
            train_ma = np.abs(y_train - y_train_predicted).mean()
            test_ma = np.abs(y_test - y_test_predicted).mean()
            train_mean_avg.append(train_ma)
            test_mean_avg.append(test_ma)
        train_mean_avg = np.mean(train_mean_avg)
        test_mean_avg = np.mean(test_mean_avg)

        d = {'interpolation': {}, 'extrapolation': {}}
        for model in models:
            learned_model = train(model, X_train, y_train, 100000)
            y_train_predicted = learned_model(torch.tensor(X_train, dtype=torch.float)).flatten().detach().numpy()
            y_test_predicted = learned_model(torch.tensor(X_test, dtype=torch.float)).flatten().detach().numpy()
            train_ma = np.abs(y_train - y_train_predicted).mean()
            test_ma = np.abs(y_test - y_test_predicted).mean()
            print(f'Operation: {op_name}, Model: {model.__name__}, '
                  f'Train loss: {train_ma/train_mean_avg*100}, Test loss: {test_ma/test_mean_avg*100}')
            d['interpolation'][model.__name__] = train_ma / train_mean_avg * 100
            d['extrapolation'][model.__name__] = test_ma / test_mean_avg * 100
        results[op_name] = d

    with open('results.txt', 'w') as f:
        interp_titles = 'Interpolation:\n' + ' ' * 8 + '\t'.join([
            'NALU', 'NAC', 'relu6', 'softsign',
            'tanh', 'sigmoid', 'selu', 'elu', 'relu', 'MLP'
        ]) + '\n'
        f.write(interp_titles)
        for operation_name in results:
            f.write('\t'.join([
                operation_name + ' ' * (7 - len(operation_name))
            ]))
            for model_name, result in results[operation_name]['interpolation'].items():
                f.write('{:4.1f}'.format(result) + '\t')
            f.write('\n')

        extrap_titles = '\nExtrapolation:\n' + ' ' * 8 + '\t'.join([
            'NALU', 'NAC', 'relu6', 'softsign',
            'tanh', 'sigmoid', 'selu', 'elu', 'relu', 'MLP'
        ]) + '\n'
        f.write(extrap_titles)
        for operation_name in results:
            f.write('\t'.join([
                operation_name + ' ' * (7 - len(operation_name))
            ]))
            for model_name, result in results[operation_name]['extrapolation'].items():
                f.write('{:4.1f}'.format(result) + '\t')
            f.write('\n')
