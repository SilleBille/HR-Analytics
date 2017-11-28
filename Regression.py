import pandas as pd
import numpy as np
import sklearn.metrics as eval_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import Transformation as tranf


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

print("==============================Logistic Reg==================")
dec_model = LogisticRegression()

fitted_model = dec_model.fit(x_train, y_train)
print(fitted_model)

y_pred = dec_model.predict(x_test)
print("accuracy = ", eval_model.accuracy_score(y_test, y_pred) * 100)

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


'''
with open("decision_tree.txt", 'w') as f:
    tree.export_graphviz(dec_model, out_file=f)
'''
print()
