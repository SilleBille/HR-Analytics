import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import Transformation as tranf
import sklearn.metrics as eval_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


data_path = "../Data/"
data = pd.read_csv(data_path + "HR_comma_sep.csv")
cost_mat = [[0, 2], [50, -5]]
print(data.head())

y_label = 'left'

data['satisfaction_level'] = tranf.z_normalize(data['satisfaction_level'])
data['last_evaluation'] = tranf.z_normalize(data['last_evaluation'])
data['number_project'] = tranf.z_normalize(data['number_project'])
data['average_montly_hours'] = tranf.z_normalize(data['average_montly_hours'])
data['time_spend_company'] = tranf.z_normalize(data['time_spend_company'])

ch = 2

if(ch == 1):
    data['sales'] = data['sales'].astype('category').cat.codes
    data['salary'] = data['salary'].astype('category').cat.codes
else:
    data = tranf.ordinal_to_number(data, 'salary', ['low', 'medium', 'high'])
    data = tranf.nominal_to_binary_vec(data, 'sales')

print(data.dtypes)
x_labels = [feat for feat in data.columns.values if not feat == y_label]
# x_train, y_train, x_test, y_test = split(data[x_labels], data[y_label], 20)
y = data[y_label]

x_train, x_test, y_train, y_test = train_test_split(data[x_labels], y, test_size=0.20, random_state=123, stratify=y)
input_dim = x_train.shape[1]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(type(x_train))


def build_and_evaluate_model(nodes, x_t, y_t):
    np.random.seed(7)

    # Initializing a sequential model
    model = Sequential()

    # Hidden layer of size = nodes (number of neurons), with activation function = relu
    model.add(Dense(nodes, input_dim=input_dim, activation='relu'))

    # Output layer of size 1, and activation = 'sigmoid' (0,1)
    model.add(Dense(1, activation='sigmoid'))

    # loss function is Mean squared error, using adam optimizer, and metric to display as accuracy
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # Train the model using [x,train, y_train] dataset
    hist = model.fit(np.array(x_train), np.array(y_train), epochs=500, batch_size=1000, verbose=0)

    # Test the model with cross_validation set.
    scores = model.evaluate(np.array(x_t), np.array(y_t))
    pred = (model.predict(np.array(x_t)) > 0.5).astype(int)
    print("\nnumber of hidden nodes = %d   %s: %.2f%%" % (nodes, model.metrics_names[1], scores[1] * 100))
    print(model.metrics_names[0], " : ", scores[0])
    return pred


y_pred = build_and_evaluate_model(50, x_test, y_test)

print("Confusion Matrix")
conf_mat = confusion_matrix(y_test, y_pred)

print()
print('total fraction:')
print("1s = ", sum(y_test == 1))
print("0s = ", sum(y_test == 0))
tn, fp, fn, tp = conf_mat.ravel()
print(conf_mat)
print("============")
print("Accuracy = ", eval_model.accuracy_score(y_test, y_pred) * 100)
print("Precision = ", eval_model.precision_score(y_test, y_pred) * 100)
print("Recall/Sensitivity = ", eval_model.recall_score(y_test, y_pred) * 100)
print("Specificity = ", tn / (tn + fp))
print("F1 score = ", eval_model.f1_score(y_test, y_pred) * 100)
print('Total Cost', np.multiply(conf_mat, cost_mat).sum())
