import pandas as pd
import numpy as np
import math


def ordinal_to_number(data, label, val_list):
    ord_2_num = {val: i + 1 for i, val in enumerate(val_list)}
    data[label] = data[label].apply(ord_2_num.get)
    return data


def nominal_to_binary_vec(data, label):
    feat = data[label]
    data = data[[lab for lab in data.columns.values if lab != label]]
    bin_vec = pd.get_dummies(feat)
    bin_vec.columns = [label + "-" + col for col in bin_vec.columns.values]
    return pd.concat([data, bin_vec], axis=1)


def z_normalize(feat):
    feat = (feat - feat.mean()) / feat.std()
    return feat


def log_transformation(data):
    return data.apply(np.log2)


def entropy(y):
    Hy = 0
    for y_val in y.unique():
        prob = sum(y == y_val) / len(y)
        Hy += -prob * math.log(prob, 2)
    return Hy


def conditional_entropy_cat(x, y):
    Hy = 0
    for x_val in x.unique():
        Prob = sum(x == x_val) / len(x)
        Hy += Prob * entropy(y.loc[x == x_val])
    return Hy
