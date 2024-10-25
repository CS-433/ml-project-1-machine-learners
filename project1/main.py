from implementations import *
from model import *
from helpers import *
from data_preprocessing import *
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

    #Load and preprocess the data
    x_train, x_test, y_train,train_ids,test_ids = preprocess_data()

    # Select the features 
    x_train, x_test = select_features(x_train, y_train, x_test)

    if args.sampling == 'oversampling':
        x_train, y_train = oversampling(x_train, y_train, args.proportion)
    elif args.sampling == 'undersampling':
        x_train, y_train = undersampling(x_train, y_train, args.proportion)
    #Train the model

    w, loss = train(x_train, y_train, lr=0.01, max_iters=1000, lambda_=0.01)

    #Predict the labels

    y_pred = predict_labels(w, x_test)
    y_pred = y_pred*2-1

    #Save the predictions

    create_csv_submission(test_ids,y_pred, 'submission.csv')


