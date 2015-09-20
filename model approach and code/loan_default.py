# Boston College CSCI3346 Data Mining Spring 2015
# Final Project: Loan Default Prediction
# Author: Ziyuan Chen
# Teammates: Haotian Chen, Yang Zhou, Tianyu Xiang

import numpy as np
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def load_data():
    train = np.genfromtxt(open('train_v2.csv', 'rb'), delimiter=',', skip_header=1)
    test = np.genfromtxt(open('test_v2.csv', 'rb'), delimiter=',', skip_header=1)
    # clean data
    train = clean_data(train)
    test = clean_data(test)
    # separate instances and target
    xs = train[:, range(1, 770)]
    ys = train[:, -1]
    ts = test[:, range(1, 770)]
    # clear out two features causing overflow
    xs = np.delete(xs, 388, 1)
    ts = np.delete(ts, 388, 1)
    xs = np.delete(xs, 616, 1)
    ts = np.delete(ts, 616, 1)
    # make a binary target
    ysb = np.zeros(len(ys))
    ysb[ys > 0] = 1
    return xs, ys, ysb, ts

def clean_data(data):
    means = np.nanmean(data, axis=0)
    nan_index = np.where(np.isnan(data))
    data[nan_index] = means[nan_index[1]]
    return data

from sklearn.ensemble import ExtraTreesClassifier

# select 154 features using ExtraTreesClassifier
def feature_selection(xs, ys, ts):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(xs, ys)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    xs_selected = xs[:, indices[:150]]
    ts_selected = ts[:, indices[:150]]
    xs_selected = add_golden_features(xs, xs_selected)
    ts_selected = add_golden_features(ts, ts_selected)
    return xs_selected, ts_selected

# see Tianyu's code for computation of indices of these features
def add_golden_features(datas, tops):
    fs = np.array([datas[:, 520] - datas[:, 519]]).T
    fs = np.hstack((fs, np.array([datas[:, 520] + datas[:, 519]]).T))
    fs = np.hstack((fs, np.array([datas[:, 520] - datas[:, 271]]).T))
    fs = np.hstack((fs, np.array([datas[:, 520] + datas[:, 268]]).T))
    fs = np.hstack((fs, tops[:, 0:tops.shape[1]]))
    return fs

def single_stage_decision_tree(xs, ys, ts):
    xs_selected, ts_selected = feature_selection(xs, ys, ts)
    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf = clf.fit(xs_selected, ys)
    Y = clf.predict(ts_selected)
    output_prediction(Y)

def two_stage_gradient_boosting(xs, ys, ysb, ts):
    # classification stage
    xsb, tsb = feature_selection(xs, ysb, ts)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.3, min_samples_split=30, min_samples_leaf=5)
    clf.fit(xsb, ysb)
    Y_bin = clf.predict(tsb)
    # regression stage
    ind_tsr = np.where(Y_bin > 0)[0]
    ts = ts[ind_tsr]
    ind_defaults = np.where(ys > 0)[0]
    xs = xs[ind_defaults]
    ysr = ys[ind_defaults]
    xsr, tsr = feature_selection(xs, ysr, ts)
    reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, min_samples_split=30, min_samples_leaf=5, loss='lad')
    reg.fit(xsr, ysr)
    Y_defaults = reg.predict(tsr)
    Y = np.zeros(210944)
    Y[ind_tsr] = Y_defaults
    output_prediction(Y)

def output_prediction(Y):
    f = open('output.csv', 'w')
    f.write('id,loss\n')
    for i in range(len(Y)):
        if Y[i] > 100:
            Y[i] = 100
        elif Y[i] < 0:
            Y[i] = 0
        f.write(str(i+105472) + ',' + str(np.float(Y[i])) + '\n')
    f.close()
            
def main():
    two_stage_gradient_boosting(load_data())

