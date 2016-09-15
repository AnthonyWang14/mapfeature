import numpy as np

import enchant
dictionary=enchant.Dict("en_US")
abc = "abcdefghijklmnopqrstuvwxyz"

from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

from recal.common import predict_raw_proba

# ----------------------------------------------------------------------------
# Feature generators

class BestClassFeaturizer(object):
	"""Produces targets and features from a classifier"""
	def __init__(self, clf=None):
		# super(BestClassFeaturizer, self).__init__()
		self.clf = clf

	def set_clf(self, clf):
		self.clf = clf
	
	def featurize(self, X, marginals=None):
		F = _delta_wrt_best(self.clf, X)
		return F

	def labelize(self, X, y):
		return best_class(self.clf, X, y)

class MultiClassFeaturizer(object):
	"""Produces targets and features from a classifier"""
	def __init__(self, clf):
		# super(MultiClassFeaturizer, self).__init__()
		self.clf = clf
	
	def featurize(self, X):
		F = _all_deltas(self.clf, X)
		return F

	def labelize(self, X, y):
		return y

# ----------------------------------------------------------------------------
# Structured feature generators

class StructuredBestClassFeaturizer(object):
	"""Produces targets and features from a classifier"""
	def __init__(self, clf=None, features=['margins'], indiv_clf=None):
		# super(BestClassFeaturizer, self).__init__()
		self.features = features
		self.clf = clf
		self.indiv_clf = indiv_clf

	def set_clf(self, clf):
		self.clf = clf
	
	def featurize(self, X, marginals=None):
		F = list()
		N = len(X)

		Y1 = self.clf.predict(X)
		S1 = self.clf.predict_proba(X, Y1)

		if 'worst-marginal' in self.features:
			marginals = self.clf.predict_marginals(X)

		for feature in self.features:
			if feature == 'margins':
				Y2 = self.clf.predict_2nd_best(X)
				S2 = self.clf.predict_proba(X, Y2)
				F.append(np.array([s1 - s2 for (s1,s2) in zip(S1, S2)]))
			elif feature == 'probabilities':
				F.append(S1)
			elif feature == 'valid-word':
				F.append(np.array([_true_word(y) for y in Y1]))
			elif feature == 'indiv-clf-support':
				F.append(np.array(self._indiv_support(X)))
			elif feature == 'worst-marginal':
				L = list()
				for i, x in enumerate(X):
					sorted_marginals = np.sort(marginals[i],axis=1)
					marginal_margins = sorted_marginals[-1] - sorted_marginals[-2]
					L.append(np.min(marginal_margins))
				F.append(np.array(L))
			elif feature == 'map-concordance':
				Y_md = self.clf.predict_via_marginal_decoding(X)
				F.append(np.array([1 if all(y_md == y_map) else 0 for y_md,y_map in zip(Y_md,Y1)]))
			elif feature == 'word-length':
				F.append(np.array([len(x) for x in X]))
			else:
				raise ValueError("Invalid feature type requested: %s" % feature)

		return np.hstack([f.reshape(N,1) for f in F])

	def labelize(self, X, Y):
		Y_best = self.clf.predict(X)
		return np.array([1 if all(y == y_best) else 0 for y,y_best in zip(Y,Y_best)])

	def _indiv_support(self, X):
		scores = list()
		for X_i in X:
			indiv_P = predict_raw_proba(self.indiv_clf, X_i)
			if len(indiv_P.shape) == 1:
				raise Exception()
			else:
				sorted_P = np.sort(indiv_P,axis=1)
				scores.append(np.min(sorted_P[-1] - sorted_P[-2]))
		return scores

class MarginalMultiClassFeaturizer(object):
	"""Produces targets and features from a classifier"""
	def __init__(self, marginal=0):
		# super(BestClassFeaturizer, self).__init__()
		self.marginal = marginal

	def set_clf(self, clf):
		self.clf = clf
	
	def featurize(self, X):
		M = self.clf.predict_marginals(X)
		return np.array([m[self.marginal] for m in M])

	def labelize(self, X, Y):
		return np.array([y[self.marginal] for y in Y])

class MarginalBestClassFeaturizer(object):
	"""Produces targets and features from a classifier"""
	def __init__(self, clf, marginal=0):
		# super(BestClassFeaturizer, self).__init__()
		self.clf = clf
		self.marginal = marginal
	
	def featurize(self, X):
		M = self.clf.predict_marginals(X)
		M_sorted = [np.sort(m[self.marginal]) for m in M]
		return np.array([m[-1] - m[-2] for m in M_sorted])

	def labelize(self, X, Y):
		i = self.marginal
		Y_best = self.clf.predict(X)
		return np.array([1 if y[i] == y_best[i] else 0 for y,y_best in zip(Y,Y_best)])

class AllMarginalsMultiClassFeaturizer(object):
	"""Produces targets and features from a classifier"""
	def __init__(self):
		# super(BestClassFeaturizer, self).__init__()
		pass

	def set_clf(self, clf):
		self.clf = clf
	
	def featurize(self, X):
		return np.array([M_i for M_x in self.clf.predict_marginals(X) for M_i in M_x])

	def labelize(self, X, Y):
		return np.array([y_i for y in Y for y_i in y])

class AllMarginalsBestClassFeaturizer2(object):
	"""Produces targets and features from a classifier"""
	def __init__(self):
		# super(BestClassFeaturizer, self).__init__()
		pass

	def set_clf(self, clf):
		self.clf = clf
	
	def featurize(self, X):
		marginals = self.clf.predict_marginals(X)
		L = list()
		for marginal in marginals:
			for marginal_row in marginal:
				m_row_sorted = np.sort(marginal_row)
				L.append(m_row_sorted[-1] - m_row_sorted[-2])
		
		return np.array(L)

	def labelize(self, X, Y):
		Y_best = self.clf.predict(X)
		return np.array([1 if y_i == y_best_i else 0 
							for Y_i ,Y_best_i in zip(Y,Y_best) 
							for y_i, y_best_i in zip(Y_i ,Y_best_i)])

class AllMarginalsBestClassFeaturizer(object):
	"""
	Features for determining whether the best marginal (obtained via marginal decoding)
	is correct or not.
	"""

	def __init__(self, clf=None, features=['margins'], indiv_clf=None):
		# super(BestClassFeaturizer, self).__init__()
		self.features = features
		self.indiv_clf=indiv_clf
		self.clf = clf

	def set_clf(self, clf):
		self.clf = clf
	
	def featurize(self, X, marginals=None, map_alg='ad3'):
		assert self.clf is not None

		F = list()
		if not marginals:
			marginals = self.clf.predict_marginals(X)
		Y_pred = [np.argmax(m,axis=1) for m in marginals]

		Y_map = None
		if 'map-concordance' in self.features:
			Y_map = self.clf.predict_map(X, map_alg)

		if 'pseudomargins' in self.features:
			pseudomarginals = self.clf.pseudomarginals(X, Y_map)

		if 'label-count' in self.features:
			label_counts = dict()
			for i, y in enumerate(Y_pred):
				label_counts[i] = dict()
				for y_j in y:
					if y_j not in label_counts[i]: label_counts[i][y_j] = 0
					label_counts[i][y_j] += 1

		if 'lbp-margins' in self.features:
			bp_marginals = [self.clf._marginals_via_bp(x) for x in X]

		if 'gibbs-margins' in self.features:
			gibbs_marginals = [self.clf._marginals_via_gibbs(x) for x in X]

		for i, marginal in enumerate(marginals):
			
			if 'indiv-clf-support' in self.features:
				if isinstance(X[0], tuple):
					indiv_P = predict_raw_proba(self.indiv_clf, X[i][0])
				else:
					indiv_P = predict_raw_proba(self.indiv_clf, X[i])

			for j, marginal_row in enumerate(marginal):
				# enumerate every sample
				num_classes = len(marginal_row)
				best_m = np.argmax(marginal_row)
				F_row = list()
				for feature in self.features:
					if feature == 'probabilities':
						m_row_sorted = np.sort(marginal_row)
						F_row.append(m_row_sorted[-1])
					elif feature == 'margins':
						m_row_sorted = np.sort(marginal_row)
						F_row.append(m_row_sorted[-1] - m_row_sorted[-2])
					elif feature == 'pseudomargins':
						sorted_pm = np.sort(pseudomarginals[i][j])
						F_row.append(sorted_pm[-1] - sorted_pm[-2])	
					elif feature == 'map-concordance':
						F_row.append(1 if Y_map[i][j] == best_m else 0)
					elif feature == 'indiv-clf-support':
						F_row.append(indiv_P[j][best_m])
					elif feature == 'label-type':
						f = np.zeros((num_classes,))
						f[best_m] = 1
						F_row.extend(f)
					elif feature == 'bad-labels':
						f = 1 if best_m in [23, 10, 3] else 0
						F_row.append(f)
					elif feature == 'neighbor-concordance':
						F_row.append(_neighbor_concordance(j, Y_pred[i], X[i][1]))
					elif feature == 'label-count':
						F_row.append(label_counts[i][Y_pred[i][j]])
					elif feature == 'lbp-margins':
						sorted_m = np.sort(bp_marginals[i][j])
						F_row.append(sorted_m[-1] - sorted_m[-2])	
					elif feature == 'gibbs-margins':
						sorted_m = np.sort(gibbs_marginals[i][j])
						F_row.append(sorted_m[-1] - sorted_m[-2])	
					else:
						raise ValueError("Invalid feature type requested: %s" % feature)
				F.append(F_row)

		return np.vstack(F)

	def labelize(self, X, Y, Y_best=None):
		assert self.clf is not None

		if not Y_best:
			Y_best = self.clf.predict(X)
		return np.array([1 if y_i == y_best_i else 0 
							for Y_i ,Y_best_i in zip(Y,Y_best) 
							for y_i, y_best_i in zip(Y_i ,Y_best_i)])

		
# ----------------------------------------------------------------------------
# Graph CRF features:

def _neighbor_concordance(i, y, edges):
	neighbors = set()
	for v1, v2 in edges:
		if v1 == i: neighbors.add(v2)
		if v2 == i: neighbors.add(v1)

	if len(neighbors) == 0:
		return 1.0

	concordant_neighbors = [n for n in neighbors if y[n] == y[i]]
	return float(len(concordant_neighbors)) / len(neighbors)


# ----------------------------------------------------------------------------
# High level feature computation

def single_class_features(clf, X, k):
	return _delta_class(clf, X, k)

# ----------------------------------------------------------------------------
# Input representation

def best_class(clf, X, y):
	y_pred = clf.predict(X)
	z = np.array([1 if y_i == y_pred_i else 0 for y_i, y_pred_i in zip(y, y_pred)])

	return z

def all_classes(clf, X, y):
	return y

# ----------------------------------------------------------------------------
# Features for best class

def _delta_wrt_best(clf,X):
	"""
	Difference between y_best and y_2nd_best
	"""

	p = _raw_probs(clf, X)
	sorted_p = np.sort(p, axis=1)

	return sorted_p[:,-1] - sorted_p[:,-2]

# ----------------------------------------------------------------------------
# Multiclass features

def _score_features(clf, X):
	"""
	The uncalibrated probability estimates for each class
	"""

	return _raw_probs(clf, X)

def _all_deltas(clf, X):
	""" 
	For all k: difference between p(y=k) and p(y'), 
	where y' is second highest class after k
	"""

	p = _raw_probs(clf, X)
	n_pts, n_classes = p.shape

	ordered_classes = np.argsort(p, axis=1)
	deltas = np.zeros(p.shape)
	for k in xrange(n_classes):
		idx_k = np.array([l[-1] if l[-1] != k else l[-2] for l in ordered_classes])
		deltas[:,k] = p[:,k] - p[np.arange(n_pts),idx_k]

	return deltas

def _entropy():
	pass
	
def _delta_to_closest_class(clf, X):
	pass

# ----------------------------------------------------------------------------
# Other/test features

def _delta_known_y_vs_best(clf, X, y):
	""" 
	Difference between p(y) and p(y') where y' is second highest class after y.
	So if delta > 0, we are predicting well, and delta is a margin.
	y is the MAP class of each x.
	"""

	y_pred = clf.predict(X)
	p = _raw_probs(clf, X)
	n_pts, n_classes = p.shape
	
	ordered_classes = np.argsort(p, axis=1)
	idx = np.array([l[-1] if l[-1] != y_i else l[-2] 
					for l, y_i in zip(ordered_classes,y)])

	return p[np.arange(n_pts),idx] - p[np.arange(n_pts),y]

def _delta_class(clf, X, k):
	""" 
	Difference between p(y=k) and p(y') where y' is second highest class after k
	"""

	p = _raw_probs(clf, X)
	n_pts, n_classes = p.shape

	ordered_classes = np.argsort(p, axis=1)
	idx = np.array([l[-1] if l[-1] != k else l[-2] for l in ordered_classes])

	return p[:,k] - p[np.arange(n_pts),idx]


# ----------------------------------------------------------------------------
# Helpers

def _get_num_classes(x):
	if isinstance(x, tuple):
		return len(x[0])
	else:
		return len(x)

def _raw_probs(clf, X):
    if hasattr(clf, "decision_function"):
        p = clf.decision_function(X)
    elif hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
    else:
        raise RuntimeError('classifier has no decision_function or '
                           'predict_proba method.')
    return p

def _make_word(s):
	return ''.join([abc[i] for i in s])

def _true_word(w):
	word_suffix = _make_word(w)
	if word_suffix == 'cknowledgement': return 1 # hack b/c of US vs UK spelling rules
	for letter in abc:
		word = letter + word_suffix
		if dictionary.check(word):
			return 1
	return 0