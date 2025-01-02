import numpy as np

true_labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
predicted_labels = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0])

TP = np.sum((predicted_labels == 1) & (true_labels == 1))
TN = np.sum((predicted_labels == 0) & (true_labels == 0))
FP = np.sum((predicted_labels == 1) & (true_labels == 0))
FN = np.sum((predicted_labels == 0) & (true_labels == 1))

print("Confusion Matrix:\n TP: ", TP, "\tFP: ", FP, "\n FN: ", FN, "\tTN: ", TN)

'''Output:
Confusion Matrix:
 TP:  4 	FP:  2 
 FN:  2 	TN:  2
'''

def calculate_precision(TP, FP):
    return TP / (TP + FP)

def calculate_recall(TP, FN):
    return TP / (TP + FN)

precision = calculate_precision(TP, FP)
recall = calculate_recall(TP, FN)

print("Precision: ", round(precision, 2))  # 0.67
print("Recall: ", round(recall, 2))  # 0.67


