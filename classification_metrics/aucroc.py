from matplotlib import pyplot as plt
from numpy import random

truth_labels = [1 if random.rand() > 0.6 else 0 for _ in range(500)]
# we generate some random predictions that would normally be obtained from the model
# If a predicted probability is higher than the threshold, it is considered to be a positive outcome
predicted_probs = [max(0, min(1, random.normal(loc=label, scale=0.3))) for label in truth_labels]

def roc_curve(truth_labels, predicted_probs):
    thresholds = [0.1 * i for i in range(11)]
    tprs, fprs = [], []
    for threshold in thresholds:
        tp = fp = tn = fn = 0  # initialize confusion matrix counts
        # for each prediction
        for i in range(len(truth_labels)):
            # calculate confusion matrix counts
            if predicted_probs[i] >= threshold:
                if truth_labels[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if truth_labels[i] == 1:
                    fn += 1
                else:
                    tn += 1
        # track the TPR and FPR for this threshold
        tprs.append(tp / (tp + fn))  # True Positive Rate (TPR)
        fprs.append(fp / (tn + fp))  # False Positive Rate (FPR)
    return tprs, fprs


tprs, fprs = roc_curve(truth_labels, predicted_probs)
plt.plot(fprs, tprs, marker='.')
plt.show()

def compute_aucroc(tprs, fprs):
    aucroc = 0
    for i in range(1, len(tprs)):
        aucroc += 0.5 * abs(fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1])
    return aucroc

aucroc = compute_aucroc(tprs, fprs)
print(f"The AUC-ROC value is: {aucroc}")  # The AUC-ROC value is: 0.9827272125066242
