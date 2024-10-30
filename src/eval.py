import numpy as np
import matplotlib.pyplot as plt


def compute_accuracy(y_true, y_pred):
    """
    Computes the accuracy of predictions.

    Parameters:
    y_true (array-like): Array of true labels.
    y_pred (array-like): Array of predicted labels.

    Returns:
    accuracy: The accuracy of the predictions.
    """
    return np.sum(y_true == y_pred) / y_true.shape[0]


def compute_f1_score(y_true, y_pred):
    """
    Computes the F1 score for binary classification.

    Parameters:
    y_true (array-like): Array of true labels (binary values: 0 or 1).
    y_pred (array-like): Array of predicted labels (binary values: 0 or 1).

    Returns:
    f1: The F1 score of the predictions.
           Returns 0.0 if precision or recall is undefined.
    """
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
    """
    Constructs the confusion matrix for binary classification.

    Parameters:
    y_true (array-like): Array of true labels (binary values: 0 or 1).
    y_pred (array-like): Array of predicted labels (binary values: 0 or 1).

    Returns:
    confusion_matrix: A 2x2 numpy array, where the elements represent:
                [[True Negatives, False Positives],
                 [False Negatives, True Positives]]
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the four components of the confusion matrix
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Create the confusion matrix
    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    return confusion_matrix


def evaluate_predictions(y_true, y_pred):
    """
    Evaluates and prints key metrics for binary classification.

    Parameters:
    y_true (array-like): Array of true labels (binary values: 0 or 1).
    y_pred (array-like): Array of predicted labels (binary values: 0 or 1).

    Prints:
    - Accuracy: The overall accuracy of predictions.
    - F1 Score: The F1 score of predictions.
    - Confusion Matrix: A 2x2 confusion matrix of the predictions.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {compute_accuracy(y_true, y_pred)}")
    print(f"F1 score: {compute_f1_score(y_true, y_pred)}")

    print("Confusion Matrix: ")
    print(conf_matrix)
