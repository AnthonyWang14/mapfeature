import numpy as np
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve

# ----------------------------------------------------------------------------
# Visualization

def calibration_plot(probabilities, labels):
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot(np.linspace(0,1,20), np.linspace(0,1,20))

    for y_p_pair, label in zip(probabilities,labels):
        y, proba = y_p_pair
        p_emp, p_true = calibration_curve(y, proba, n_bins=10)
        ax1.plot(p_true, p_emp, 's-', label=label)
        ax2.hist(proba, range=(0, 1), bins=10, histtype="step", lw=2, label=label)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    plt.show()

def calibration_metrics(probabilities, labels):
    for y_p_pair, label in zip(probabilities, labels):
        y, proba = y_p_pair
        print
        print label
        print '', 'Brier score', brier_score(proba, y)
        print '', 'Calibration error', calibration_error(proba, y)
        print '', 'Sharpness', sharpness(proba, y)

# ----------------------------------------------------------------------------
# Save stuff to a file

def save_calibration_plot(probabilities, labels, plot_file, hist_file):
    out_plot = open(plot_file, 'w')
    out_hist = open(hist_file, 'w')

    for y_p_pair, label in zip(probabilities,labels):
        y, proba = y_p_pair
        p_emp, p_true = calibration_curve(y, proba, n_bins=10)
        out_line = '\t'.join(['%f,%f' % (p_e, p_t) for p_e,p_t in zip(p_emp, p_true)])
        out_plot.write('%s\t%s\n' % (label, out_line))
        out_line = '\t'.join(['%f' % p for p in proba])
        out_hist.write('%s\t%s\n' % (label, out_line))

def save_calibration_metrics(probabilities, labels, out_file):
    out = open(out_file, 'w')
    for y_p_pair, label in zip(probabilities, labels):
        y, proba = y_p_pair
        out.write('%s\tbrier-score\%f\n' % (label, brier_score(proba, y)))
        out.write('%s\tcalibration-error\%f\n' % (label, calibration_error(proba, y)))

# ----------------------------------------------------------------------------
# Extract classifier probabilities

def compute_label_proba(clf, k):
    pred_probs = predict_raw_proba(clf, X, n_classes)
    return pred_probs[:,k]

# ----------------------------------------------------------------------------
# Measurment of errors

def predict_raw_proba(clf, X):
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
    elif hasattr(clf, "decision_function"):
        p = clf.decision_function(X)
        p = (p - p.min(axis=0)) / (p.max(axis=0) - p.min(axis=0))
    else:
        raise RuntimeError('classifier has no decision_function or '
                           'predict_proba method.')

    return p

def calibration_error(proba, y):
    p_emp, p_true = calibration_curve(y, proba, n_bins=10)
    L = _bucket_sizes(proba)
    return sum([l*(pe - pt)**2 for l, pe, pt in zip(L, p_emp, p_true)]) / sum(L)

def sharpness(proba, y):
    p_emp, p_true = calibration_curve(y, proba, n_bins=10)
    L = _bucket_sizes(proba)
    y_avg = np.mean(y)
    return sum([l*(pt - y_avg)**2 for l, pt in zip(L, p_true)]) / sum(L)

def brier_score(proba, y):
    return sum([(p_i - y_i)**2 for p_i, y_i in zip(proba, y)]) / len(y)

def accuracy(proba, y):
    y_pred = [1 if p_i >= 0.5 else 0 for p_i in proba]
    return sum([1.0 if y_pred_i == y_i else 0.0 for y_pred_i, y_i in zip(y_pred, y)]) / len(y)    

# ----------------------------------------------------------------------------
# Helpers

def _bucket_sizes(p, n_bins=10):
    lengths = list()
    iv_size = 1./n_bins
    for i in xrange(n_bins):
        l = len([p_j for p_j in p if i*iv_size <= p_j <= (i+1)*iv_size])
        if l:
            lengths.append(l)
    return lengths
