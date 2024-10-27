import numpy as np
import matplotlib.pyplot as plt

def compute_accuracy(y_true, y_pred):
  return np.sum(y_true == y_pred) / y_true.shape[0]

def compute_f1_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Avoid division by zero
    if TP + FP == 0 or TP + FN == 0:
        return 0.0

    # Calculate Precision and Recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def confusion_matrix(y_true, y_pred):
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the four components of the confusion matrix
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Create the confusion matrix
    confusion_matrix = np.array([[TN, FP],
                                 [FN, TP]])

    return confusion_matrix

def evaluate_predictions(y_true, y_pred):
  conf_matrix = confusion_matrix(y_true, y_pred)
  
  print(f"Accuracy: {compute_accuracy(y_true, y_pred)}")
  print(f"F1 score: {compute_f1_score(y_true, y_pred)}")
  
  print("Confusion Matrix: ")
  print(conf_matrix)