import numpy as np
import random
try:
    import cPickle as pickle
except ImportError:
    import pickle

from sklearn.calibration import calibration_curve

import argparse

import pystruct
import pystruct.models as crfs
from pystruct import learners
from pystruct.utils import SaveLogger

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC

from recal.common import predict_raw_proba, calibration_plot, calibration_metrics, _bucket_sizes, calibration_error, sharpness, brier_score, accuracy
from recal.recalibrators import CalibratedClassifier
from recal.structured_learners import GraphCRFPredictor
from recal.feature_extractors import AllMarginalsBestClassFeaturizer


def output_Graph(X, trained_model, filename):
    model = trained_model.model
    w = trained_model.w
    # up: n_nodes,n_states
    up = model._get_unary_potentials(X, w)
    # pp: n_edges, n_states, n_states
    pp = model._get_pairwise_potentials(X, w)
    n_nodes = len(X[0])
    n_edges = len(X[1])
    n_states = len(up[0])
    f = open(filename, 'w')
    f.write(str(n_nodes)+' '+str(n_edges)+' '+str(n_states)+'\n')
    # output unary potential
    for i in range(len(up)):
        f.write(str(i)+' ')
        for j in up[i]:
            f.write(str(j)+' ')
        f.write('\n')

    # output pairwise potential
    for i in range(len(pp)):
        f.write(str(X[1][i][0])+' '+str(X[1][i][1])+'\n')
        for j in range(n_states):
            for k in range(n_states):
                f.write(str(pp[i][j][k])+' ')
            f.write('\n')


# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data-path', default='../data')
parser.add_argument('-o', '--out-path', default='../results')

args = parser.parse_args()

# ----------------------------------------------------------------------------

data_train = pickle.load(open(args.data_path + "/voc/data_train.pickle"))
data_val = pickle.load(open(args.data_path + "/voc/data_val.pickle"))

X_train_raw, y_train_raw = data_train['X'], data_train['Y']
X_test_raw, y_test_raw = data_val['X'], list(data_val['Y'])
# X_test, y_test = X_test[:500], y_test[:500]

X_train, y_train = X_train_raw[:600], y_train_raw[:600]
X_vt = X_train_raw[600:] + X_test_raw
y_vt = y_train_raw[600:] + y_test_raw

# ----------------------------------------------------------------------------

C = 0.01
n_states = 21

# ----------------------------------------------------------------------------

raw_marginal_ssvm = pickle.load(open('../models/edge_features_one_slack_trainval_N600_0.010000.pickle'))

model = raw_marginal_ssvm.model
graph_marginal_ssvm = GraphCRFPredictor(raw_marginal_ssvm, model)

print len(raw_marginal_ssvm.w)
up = np.exp(model._get_unary_potentials(X_train[1], raw_marginal_ssvm.w))
print up.shape

# output_Graph(X_train[0], raw_marginal_ssvm, 'pgm_input')

Y_map = pystruct.inference.inference_ad3(
            model._get_unary_potentials(X_train[0], raw_marginal_ssvm.w),
            model._get_pairwise_potentials(X_train[0], raw_marginal_ssvm.w),
            X_train[0][1]
            )
print Y_map
exit()
# X_val, y_val = X_train[600:964], y_train[600:964]

# ----------------------------------------------------------------------------

svm = LinearSVC(dual=False, C=.1)
svm.fit(np.vstack([x[0] for x in X_train[:600]]), np.hstack(y_train[:600]))

# ----------------------------------------------------------------------------
# Feature combinations to try

labels = ('Uncalibrated', 
          'Unstructured SVM scores', 
          'Pseudomargins', 
          'Pseudomargins, other MAP features', 
          'Marginals, MAP/marg. concordance', 
          'All features')

feature_groups = (
        ['uncalibrated'],
        ['indiv-clf-support'],
        ['pseudomargins'],
        ['label-count', 'neighbor-concordance', 'label-type', 'pseudomargins'],
        ['map-concordance', 'margins'],
        ['label-type', 'neighbor-concordance', 'label-count', 'indiv-clf-support', 'margins']
    )

# ----------------------------------------------------------------------------

# calibrated_marginal_ssvm1 = CalibratedClassifier(graph_marginal_ssvm, method='', 
#                                                 featurizer=AllMarginalsBestClassFeaturizer(
#                                                     features=['indiv-clf-support'], indiv_clf=svm
#                                                 ))
# calibrated_marginal_ssvm1.fit_val(X_val, y_val, recalibrator=DecisionTreeClassifier(min_samples_leaf=50))

def _train_ssvm(features, X_in, y_in):
    calibrated_graph_ssvm = CalibratedClassifier(graph_marginal_ssvm, 
                                                    featurizer= AllMarginalsBestClassFeaturizer(
                                                            features=features, indiv_clf=svm),
                                                    )
    calibrated_graph_ssvm.fit_val(X_in, y_in, 
                            recalibrator=DecisionTreeClassifier(min_samples_leaf=50))
    return calibrated_graph_ssvm

def _uncal(X_test, y_test, test_marginals):
    regular_featurizer = AllMarginalsBestClassFeaturizer(graph_marginal_ssvm)
    # event pool
    Y_pred = graph_marginal_ssvm.predict(X_test)
    reg_z_new = []
    for Y_i, Y_best_i in zip(y_test, Y_pred):
        for y_i, y_best_i in zip(Y_i, Y_best_i):
            if y_i == y_best_i:
                reg_z_new.append(1)
            else:
                reg_z_new.append(0)
    # reg_z = regular_featurizer.labelize(X_test, y_test)

    # f1 = open('1','w')
    # f1.write(' '.join([str(z) for z in reg_z_new]))
    # f2 = open('2','w')
    # f2.write(' '.join([str(z) for z in reg_z]))
    # f1.close()
    # f2.close()
    reg_z_pred = np.array([np.max(m) for M in test_marginals for m in M])
    return reg_z_new, reg_z_pred

y_true_all = [list() for f in feature_groups]
y_pred_all = [list() for f in feature_groups]

for t in xrange(1):
    L = range(len(X_vt))
    random.shuffle(L)
    X_val, y_val = [X_vt[i] for i in L[:400]], [y_vt[i] for i in L[:400]]
    X_test, y_test = [X_vt[i] for i in L[400:]], [y_vt[i] for i in L[400:]]
    X_test, y_test = X_vt[400:], y_vt[400:]

    test_marginals = graph_marginal_ssvm.predict_marginals(X_test)
    # print test_marginals
    # exit()
    # print len(test_marginals[0])
    Y_pred_test = [np.argmax(m,axis=1) for m in test_marginals]
    # print len(Y_pred_test[0])
    for i, features in enumerate(feature_groups):
        if features == ['uncalibrated']:
            z, z_pred = _uncal(X_test, y_test, test_marginals)
        else:
            cal_ssvm = _train_ssvm(features, X_val, y_val)
            calibrated_featurizer = AllMarginalsBestClassFeaturizer(cal_ssvm)
            z = calibrated_featurizer.labelize(X_test, y_test, Y_best=Y_pred_test)
            z_pred = cal_ssvm.predict_proba(X_test, test_marginals)[:,1]
        y_true_all[i].extend(z)
        y_pred_all[i].extend(z_pred)

exit()
with open(args.out_path + '/graph-crf-features-marginals-964val.data', 'w') as f:
    for y_true, y_pred, label in zip(y_true_all, y_pred_all, labels):
        f.write('%s\t%f\t%f\t%f\t%f' % (label, brier_score(y_pred, y_true), calibration_error(y_pred, y_true), sharpness(y_pred, y_true), accuracy(y_pred, y_true)))
        p_emp, p_true = calibration_curve(y_true, y_pred, n_bins=10)
        for p1, p2, b in zip(p_emp, p_true, _bucket_sizes(y_pred)):
            f.write('\t%f,%f,%d' % (p1,p2,b))
        f.write('\n')

all_probs = zip(y_true_all, y_pred_all)