# CS-433 Machine Learning - Project 1
Team  : machine-learners

Team members:
- Madeleine HUEBER
- Adrian MARTINEZ
- Duru BEKTAS


## Getting started
### Project description
This project is part of the course CS-433 Machine Learning at EPFL. The goal of the project is to predict if someone will have a cardiovascular disease based on a set of answers to a medical questionnaire. The dataset used comes from the Behavioral Risk Factor Surveillance System (BRFSS) and contains  for each person the answers to 330 questions. The target variable is binary and indicates if the person has a cardiovascular disease or not. 
For this project we first did some data preprocessing to clean the data and handle missing values. Then we did some feature selection to reduce the number of features. Finally, we trained a logistic regression model to predict the target variable.

More details about the project can be found in the project paper `project1.pdf`.

### Repository structure

`data/`: Directory containing datasets and feature information.

- `abnormal_feature_values.json`: JSON file specifying abnormal feature values to be addressed during data cleaning.
- `test_dataset.npy`: Test dataset in NumPy format.
- `train_dataset.npy` : Training dataset in NumPy format.
- `train_targets.npy` : Labels for the training dataset.

`src/` : Main directory for source code modules.

- `config.py` : Configuration file containing paths and general settings for the project.
- `data_cleaning.py`: Module for cleaning data and handling outliers or missing values.
- `data_preprocessing.py` : Functions for preprocessing data before training, such as normalization or encoding.
- `evaluation.py` : Module to evaluate model predictions using various metrics.
- `feature_engineering.py`: Contains functions to create and transform features.
- `feature_type_detection.py`: Detects data types of features and helps with automated preprocessing.
- `helpers.py` : Utility functions for tasks like saving files, managing logs, etc.
- `model.py`: Contains functions for training, validating, and predicting with the model.

Project Root:

- `.gitattributes` and .`gitignore`: Git configuration files for version control, specifying files to include or exclude.
- `README.md`: Project documentation with instructions on setup, usage, and structure.
- `implementations.py` : Functions from the part 1 of the project
- `run.py`: Main script to execute the full pipeline, from data loading and preprocessing to model training and evaluation.






### Installation 


To use our model on this project, you will first need to clone the repository 

```bash

git clone https://github.com/CS-433/ml-project-1-machine-learners/blob/final_branch/run.py

```

Then, you will need to install the required packages. You can do this by running the following command in the terminal:

```bash

pip install -r requirements.txt

```

## Running the model

To run our model, you will need to run the following command in the terminal:

```bash

python run.py --seed 42 --gamma 0.1 --max_iters 1000 --lambda_ 0 --undersampling_ratio 0.2


```

It will preprocess the data, train the model and output the predictions in the form of a csv file. The predictions will be saved as `submission.csv`.

### Arguments

You can specify the following arguments when running the model:

| Argument              | Description                                    | Type   | Default |
|-----------------------|------------------------------------------------|--------|---------|
| `--seed`              | Set the seed for deterministic results         | `int`  | 42      |
| `--gamma`             | Learning rate for training                     | `float`| 0.1     |
| `--max_iters`         | Maximum number of iterations                   | `int`  | 1000    |
| `--lambda_`           | Regularization parameter                       | `float`| 0       |
| `--undersampling_ratio` | Undersampling ratio to balance the classes    | `float`| 0.2     |

## Parameters exploration

To do some parameters exploration, you can run the following command in the terminal:

```bash

python test.py

```

It will test different values for the hyperparameters listed above and ouput which values of each hyperparameter give the best F-1 score using a 5-fold cross-validation.



