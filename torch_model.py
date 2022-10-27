import pandas as pd
import numpy as np
from src import Data_analysis_tools as Tools
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

features = Tools.load_pickle('./ml_models/features_for_prediction_new')

data = pd.read_csv('validation_data_first_half_2015_2021.csv', index_col=0)

traduction = {
        'W': 1,
        'L': 0
        }

data['W/L_Home'] = data['W/L_Home'].map(traduction)

X = data[features]


assert not(X.isnull().values.any()), 'Nan Values in data'

case = np.isinf(X).values.sum()
assert case is not None, 'Inf Values in data'

X = np.array(X)
y = np.array(data['W/L_Home'])
print(f'Initial dimensions: {X.shape[1]}')
# pca = PCA(n_components='mle')
# X = pca.fit_transform(X)
# print(f'Dimensions after PCA: {X.shape[1]}')

n_inputs = X.shape[-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import torch
from torch.nn import Linear, ReLU, Sequential, BCELoss
from torch.optim import Adam, SGD

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# number of neurons in each layer
input_num_units = X_train.shape[1]
hidden_num_units = 2056
output_num_units = 1

# set remaining variables
epochs = 100
learning_rate = 0.001

TRAIN = True

if TRAIN:

    # define model
    model = Sequential(Linear(input_num_units, hidden_num_units),
                       ReLU(),
                       # Linear(hidden_num_units, hidden_num_units),
                       # ReLU(),
                       # Linear(hidden_num_units, hidden_num_units),
                       # ReLU(),
                       torch.nn.Dropout(p=0.9, inplace=False),
                       Linear(hidden_num_units, output_num_units), torch.nn.Sigmoid())

    # loss function
    loss_fn = BCELoss()

    # define optimization algorithm
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        avg_cost = 0

        pred = model(X_train)
        pred_val = model(X_test)

        # get loss
        loss = loss_fn(pred, y_train.unsqueeze(1))
        loss_val = loss_fn(pred_val, y_test.unsqueeze(1))
        train_losses.append(loss)
        val_losses.append(loss_val)

        # perform backpropagation
        loss.backward()
        optimizer.step()
        avg_cost = avg_cost + loss.data

        if epoch % 2 != 0:
            print(epoch + 1, avg_cost)


from sklearn.metrics import confusion_matrix
y_proba_nn = model(X_test)
y_proba_nn = y_proba_nn.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()
cond = y_proba_nn >= 0.5
y_pred_nn = np.where(cond, 1, 0)
print(f'\nf1_score nn: {Tools.f1_eval(y_pred_nn, y_test)}\n')
print(confusion_matrix(y_test, y_pred_nn))