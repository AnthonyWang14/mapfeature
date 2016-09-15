from __future__ import division
import inspect
import warnings
import copy

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted
# from sklearn.cross_validation import _check_cv, KFold
from sklearn.cross_validation import KFold

from sklearn.metrics.classification import _check_binary_probabilistic_predictions

# recalibration methods
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression

# marked for deprecation:
from feature_extractors import single_class_features


class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, featurizer, method='knn-classification', cv=10, n_classes=None):
        self.clf = clf
        self.method = method
        self.featurizer = featurizer
        self.cv = cv
        self.n_classes = n_classes

    def _calibration_methods(self, N):
        if self.method == 'knn':
            return [_CalibrationMethod(KNeighborsRegressor(n_neighbors=k))
                    for k in [50, 100, 150, 200, 250]]
                    # for k in [25, 50, 75, 100, 150, 200]]
                    # for k in [100]] 100 for structured, 200 for multiclass
                    # for k in xrange(1, int(N/4), 10)]
        elif self.method == 'knn-classification':
            return [_CalibrationMethod(KNeighborsClassifier(n_neighbors=k))
                    for k in [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]]
        elif self.method == 'decision-tree':
            return [_CalibrationMethod(DecisionTreeClassifier(min_samples_leaf=n))
                    for n in [3, 5, 10, 20, 40, 60, 80, 100, 125, 150,]]
                    # for k in [3]]
        elif self.method == 'isotonic':
            return [_CalibrationMethod(IsotonicRegression(out_of_bounds='clip'))]
        else:
            raise ValueError("Invalid calibration method: %s" % self.method)

    def _calibration_method(self, N):
        """ Same as above, but return just one method """
        pass

    def _prediction_errors(self, z_true, z_pred):
        N = z_pred.shape[0]
        scores = z_pred[np.arange(N), z_true]
        
        return (1-scores)**2

    def fit(self, X, y, recalibrator=None):
        N = X.shape[0]
        
        # clf = self.clf.fit(X, y)
        featurizer = self.featurizer
        featurizer.set_clf(self.clf)
        if recalibrator:
            calibration_method = _CalibrationMethod(recalibrator)
        else:
            calibration_method = _CalibrationMethod(KNeighborsClassifier(n_neighbors=150))

        self.calibrator_ = _Calibrator(self.clf, calibration_method, featurizer, n_classes=self.n_classes)
        self.calibrator_.fit(X, y)

        return self

    def fit_val(self, X_val, y_val, recalibrator=None):
        """ Assumes the model is already trained """  
        featurizer = self.featurizer
        featurizer.set_clf(self.clf)
        # print 'n_classes', self.n_classes
        if recalibrator:
            calibration_method = _CalibrationMethod(recalibrator)
        else:
            calibration_method = _CalibrationMethod(KNeighborsClassifier(n_neighbors=150))
        # calibration_method = _CalibrationMethod(IsotonicRegression(out_of_bounds='clip'))
        self.calibrator_ = _Calibrator(self.clf, calibration_method, featurizer, n_classes=self.n_classes)
        self.calibrator_.fit(X_val, y_val)

        return self

    def fit_val_cv(self, X_val, y_val, recalibrator=None):
        """ Assumes the model is already trained """  
        featurizer = self.featurizer
        featurizer.set_clf(self.clf)
        N = X_val.shape[0]

        n_cv = 10

        kf = KFold(N, n_folds=n_cv)
        calibration_methods = self._calibration_methods(N)
        errors = np.zeros((len(calibration_methods), n_cv))

        for f, (train, test) in enumerate(kf):
            F = featurizer.featurize(X_val[train])
            z = featurizer.labelize(X_val[train], y_val[train])

            # print '...', f
            
            # for calibrator_method in ... :
            for i, calibration_method in enumerate(calibration_methods):
                # construct a calibrator
                calibrator = _Calibrator(self.clf, calibration_method, featurizer, n_classes=self.n_classes)
                calibrator.fit(X_val[train], y_val[train])

                # evaluate it on the test set
                F_test = featurizer.featurize(X_val[test])
                z_test = featurizer.labelize(X_val[test], y_val[test])
                z_pred = calibrator.predict_proba(X_val[test])

                errors[i, f] = np.mean(self._prediction_errors(z_test, z_pred))
                # print f, i, errors[i, f]

        final_errors = np.mean(errors, axis=1)
        best_i = np.argmin(final_errors)
        # print best_i
        calibration_method = calibration_methods[best_i]

        self.calibrator_ = _Calibrator(self.clf, calibration_method, featurizer, n_classes=self.n_classes)
        self.calibrator_.fit(X_val, y_val)

        return self

    # def fit_choose_on_val(self, X_tv, y_tv):
    def fit_choose_on_val(self, X_train, y_train, X_val, y_val):
        """ Assume the model is already trained. Pick best recalibrator on validation set """
        # N_tv = X_tv.shape[0]
        # X_train, y_train = X_tv[:N_tv], y_tv[:N_tv]
        # X_val, y_val = X_tv[N_tv:], y_tv[N_tv:]
        N = X_train.shape[0]

        clf = self.clf

        clf.fit(X_train,y_train)

        calibration_methods = self._calibration_methods(N)
        errors = np.zeros(len(calibration_methods),)

        featurizer = self.featurizer
        featurizer.set_clf(clf)

        for i, calibration_method in enumerate(calibration_methods):
            calibrator = _Calibrator(clf, calibration_method, featurizer, n_classes=self.n_classes)
            calibrator.fit(X_train, y_train)

            # evaluate it on the val set
            F_val = featurizer.featurize(X_val)
            z_val = featurizer.labelize(X_val, y_val)
            z_pred = calibrator.predict_proba(X_val)

            errors[i] = np.mean(self._prediction_errors(z_val, z_pred))
            print i, errors[i]

        best_i = np.argmin(errors)
        print best_i
        calibration_method = calibration_methods[best_i]
        
        # train the method on the entire set

        X_tv = np.concatenate((X_train, X_val))
        y_tv = np.concatenate((y_train, y_val))

        self.clf.fit(X_train, y_train)
        featurizer = self.featurizer
        featurizer.set_clf(self.clf)
        self.calibrator_ = _Calibrator(self.clf, calibration_method, featurizer, n_classes=self.n_classes)
        self.calibrator_.fit(X_train,y_train)

        return self

    def fit_cv(self, X, y):
        N = X.shape[0]

        # pick the best calibration method by cross-validation

        kf = KFold(N, n_folds=self.cv)
        calibration_methods = self._calibration_methods(N)
        errors = np.zeros((len(calibration_methods), self.cv))

        for f, (train, test) in enumerate(kf):
            clf = copy.deepcopy(self.clf)
            clf.fit(X[train,:], y[train])
            # clf = self.clf.fit(X[train,:], y[train])
            featurizer = self.featurizer
            featurizer.set_clf(clf)
            F = featurizer.featurize(X[train])
            z = featurizer.labelize(X[train], y[train])

            print '...', f
            
            # for calibrator_method in ... :
            for i, calibration_method in enumerate(calibration_methods):
                # construct a calibrator
                calibrator = _Calibrator(clf, calibration_method, featurizer, n_classes=self.n_classes)
                calibrator.fit(X[train], y[train])

                # evaluate it on the test set
                F_test = featurizer.featurize(X[test])
                z_test = featurizer.labelize(X[test], y[test])
                z_pred = calibrator.predict_proba(X[test])

                errors[i, f] = np.mean(self._prediction_errors(z_test, z_pred))
                print f, i, errors[i, f]

        final_errors = np.mean(errors, axis=1)
        best_i = np.argmin(final_errors)
        print best_i
        calibration_method = calibration_methods[best_i]
        
        # train the method on the entire set

        self.clf.fit(X, y)
        featurizer = self.featurizer
        featurizer.set_clf(self.clf)
        self.calibrator_ = _Calibrator(self.clf, calibration_method, featurizer, n_classes=self.n_classes)
        self.calibrator_.fit(X,y)

        return self

    def predict_proba(self, X, marginals=None):
        check_is_fitted(self, ["calibrator_", "clf"])

        return self.calibrator_.predict_proba(X, marginals=marginals)

    def predict(self, X):
        return self.clf.predict(X)

    def decision_function(self, X):
        if hasattr(self.clf, 'decision_function'):
            return self.clf.decision_function(X)
        elif hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(X)
        else:
            raise ValueError("Base classifier seems to not have "
                             "a decision funciton equivalent")


class _Calibrator(object):
    def __init__(self, base_estimator, calibration_method, featurizer, n_classes=None):
        self.base_estimator = base_estimator # must be trained
        self.featurizer = featurizer
        self.method = calibration_method
        self.n_classes = n_classes

    def _featurize(self, X, marginals=None):
        print "in _Calibrator, set map alg in featurlizer"
        return self.featurizer.featurize(X, marginals, map_alg='trw')

    def _labelize(self, X, y):
        return self.featurizer.labelize(X, y)

    def fit(self, X, y):
        F = self._featurize(X)
        z = self._labelize(X, y)
        self.observed_classes = np.unique(z)
        # of course there are 2 classes for z
        # print self.observed_classes

        self.method.fit(F,z)

        if not self.n_classes:
            self.n_classes = len(self.observed_classes)

        return self

    def predict_proba(self, X, marginals=None):
        F = self._featurize(X, marginals=marginals)
        P_pred = self.method.predict_proba(F)
        print "self.method", self.method
        N = P_pred.shape[0]
        
        P = np.zeros((N, self.n_classes))
        P[:, self.observed_classes] = self.method.predict_proba(F)
        
        return P

class _CalibrationMethod(object):
    """
    Wraps recalibration algorithms like KNN
    """
    def __init__(self, clf):
        super(_CalibrationMethod, self).__init__()
        self.clf = clf

    def fit(self, X,y):
        if isinstance(self.clf, IsotonicRegression):
            assert len(X.shape) == 1 or X.shape[1] == 1
            X = X.reshape(X.shape[0],)
        else:
            if len(X.shape) == 1:
                N = X.shape[0]
                X = X.reshape((N,1))
        self.clf.fit(X, y)

        return self

    def predict_proba(self, X):
        if len(X.shape) == 1:
            N = X.shape[0]
            X = X.reshape((N,1))
        if isinstance(self.clf, KNeighborsRegressor):
            z1 = self.clf.predict(X)
            z0 = 1-z1
            return np.vstack((z0,z1)).T
        elif isinstance(self.clf, KNeighborsClassifier):
            return self.clf.predict_proba(X)
        elif isinstance(self.clf, IsotonicRegression):
            assert len(X.shape) <= 1 or X.shape[1] == 1
            X = X.reshape(X.shape[0],)
            z1 = self.clf.predict(X)
            z0 = 1-z1
            return np.vstack((z0,z1)).T
        elif isinstance(self.clf, DecisionTreeClassifier):
            return self.clf.predict_proba(X)
        else:
            raise ValueError("Invalid recalibrator type")

# ----------------------------------------------------------------------------
# Predicting features per-class
# This is a slight modification of the sk-learn class

class _CalibratedClassifier(object):
    def __init__(self, base_estimator, method='sigmoid'):
        self.base_estimator = base_estimator
        self.method = method

    def _preproc(self, X):
        n_classes = len(self.classes_)
        if hasattr(self.base_estimator, "decision_function"):
            df = self.base_estimator.decision_function(X)
            if df.ndim == 1:
                df = df[:, np.newaxis]
        elif hasattr(self.base_estimator, "predict_proba"):
            df = self.base_estimator.predict_proba(X)
            if n_classes == 2:
                df = df[:, 1:]
        else:
            raise RuntimeError('classifier has no decision_function or '
                               'predict_proba method.')

        idx_pos_class = np.arange(df.shape[1])

        return df, idx_pos_class

    def fit(self, X, y, sample_weight=None):
        lb = LabelBinarizer()
        Y = lb.fit_transform(y)
        self.classes_ = lb.classes_

        df, idx_pos_class = self._preproc(X)
        self.calibrators_ = []

        for k, this_df in zip(idx_pos_class, df.T):
            F = single_class_features(self.base_estimator, X, k)
            if self.method == 'knn':
                calibrator = KNeighborsRegressor(n_neighbors=20)
                calibrator.fit(np.matrix(F).T, Y[:, k])
            elif self.method == 'sigmoid':
                calibrator = _SigmoidCalibration()
            else:
                raise ValueError('method should be "sigmoid" or '
                                 '"isotonic". Got %s.' % self.method)
            # calibrator.fit(this_df, Y[:, k], sample_weight)
            self.calibrators_.append(calibrator)

        return self

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))

        df, idx_pos_class = self._preproc(X)

        for k, this_df, calibrator in \
                zip(idx_pos_class, df.T, self.calibrators_):
            if n_classes == 2:
                k += 1
            proba[:, k] = calibrator.predict(np.matrix(this_df).T)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        # XXX : for some reason all probas can be 0
        proba[np.isnan(proba)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba