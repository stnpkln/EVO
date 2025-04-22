# print accuracy of the classifier
from file_handler import get_test_data_classifier, get_test_data_filter
from params import *
from sklearn.metrics import accuracy_score, precision_score, recall_score

from windows import apply_window


def get_test_stats(predictor, X_test, y_test):
    y_pred = predictor(X_test)
    y_test = y_test
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test.flatten()
    accuracy = accuracy_score(y_test_flat, y_pred_flat)
    precision = precision_score(y_test_flat, y_pred_flat, average='weighted')
    recall = recall_score(y_test_flat, y_pred_flat, average='weighted')

    return accuracy, precision, recall, y_pred, y_test


def evaluate_classifier(classifier, noise_type):
    X_test, y_test = get_test_data_classifier(noise_type)[0] # only single test image is used
    accuracy, precision, recall, pred_mask, mask = get_test_stats(classifier, X_test, y_test)

    return accuracy, precision, recall, pred_mask, mask


def evaluate_filter(filter, noise_type):
    X_test, y_test = get_test_data_filter(noise_type)[0] # only single test image is used
    noised_image = X_test.copy()
    _, _, _, filtered_image, original_image = get_test_stats(filter, X_test, y_test)

    return filtered_image, original_image, noised_image