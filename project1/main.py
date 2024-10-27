import implementations
import model
import helpers
import data_preprocessing
import eval
import numpy as np
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training model with customizable parameters.")
    parser.add_argument('--gamma', type=float, default=0.01, help="Learning rate for model training.")
    parser.add_argument('--max_iters', type=int, default=1000, help="Maximum number of iterations for training.")
    parser.add_argument('--lambda_', type=float, default=0.1, help="Regularization parameter.")
    parser.add_argument('--sampling', type=str, default='undersampling', help="Sampling method to use.")
    parser.add_argument('--proportion', type=float, default=0.5, help="Proportion of the majority class in the new dataset.")
    parser.add_argument('--output_file', type=str, default='prediction.csv', help="Output file for predictions.")

    args = parser.parse_args()

    # Set random seed to ensure that results are deterministic
    np.random.seed(42)

    # Load and preprocess the data
    x_train, x_test, y_train = data_preprocessing.preprocess_data()
    test_ids = np.arange(x_test.shape[0]) + x_train.shape[0]

    if args.sampling == 'undersampling':
        x_train_undersampled, y_train_undersampled = model.undersample(x_train, y_train, undersampling_ratio=5)

    x_tr, x_val, y_tr, y_val = model.split_data(x_train_undersampled, y_train_undersampled, val_size=0.2)

    w, loss = model.train(x_tr, y_tr, x_val, y_val, gamma=0.1, max_iters=1000, lambda_=0, threshold=1e-6)

    # Predict the labels
    y_pred_train = model.predict_labels(w, x_train)
    y_pred = model.predict_labels(w, x_test)

    eval.evaluate_predictions(y_train, y_pred_train)

    y_pred = y_pred*2-1

    # Save the predictions
    helpers.create_csv_submission(test_ids,y_pred, 'submission.csv')
