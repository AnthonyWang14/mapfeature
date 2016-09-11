import numpy as np

def compute_features(clf, X, k):
	p = _raw_probs(clf, X)
	n_pts, n_classes = p.shape

	ordered_classes = np.argsort(p, axis=1)
	idx = np.array([l[0] if l[0] != k else l[1] for l in ordered_classes])

	return p[:,k] - p[np.arange(n_pts),idx]


def _raw_probs(clf, X):
    if hasattr(clf, "decision_function"):
        p = clf.decision_function(X)
    elif hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
    else:
        raise RuntimeError('classifier has no decision_function or '
                           'predict_proba method.')
    return p