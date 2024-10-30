**Titanic: Machine Learning from Disaster (Kaggle) - Hyper-parameter Tuning and Cross-Validation**

This repository contains a Python script to perform hyperparameter tuning and cross-validation on multiple machine learning models for the Titanic: Machine Learning from Disaster competition on Kaggle. This project aims to find the best model and hyperparameters for predicting passenger survival, and prepares the results for submission to Kaggle.

Table of Contents:

### Installation
Clone this repository:

    git clone https://github.com/mohammad10787/Kaggle-Machine-Learning-Titanic.git
    
Install required dependencies. This code requires Python 3.7+ and the following packages:

    pip install -r requirements.txt

Optional: To install xgboost, if it isn’t included in requirements.txt:

    pip install xgboost


### Usage
- Download the data:
- Go to the Titanic: Machine Learning from Disaster page on Kaggle and download the train.csv and test.csv files.
- Place the downloaded files in the data file in the project directory.
- Run the script: main.py

### Results:
- The best model and hyperparameters will be displayed in the console.
- Cross-validation results, individual fold accuracies, and mean accuracy will be printed.
- Final predictions will be saved as submission.csv, formatted for direct submission to Kaggle.

### Code Structure

- main.py: The main script performing hyperparameter tuning, cross-validation, and model evaluation.
- requirements.txt: A list of Python libraries and versions needed to run the code.

### Pipeline Explanation
1. Pipeline Setup:
	- The code uses Pipeline from scikit-learn to streamline preprocessing and model selection.
	- Three models are evaluated: LogisticRegression, RandomForestClassifier, and XGBClassifier (from xgboost).
2. Parameter Grid:
	- A dictionary of hyperparameters is defined for each model, enabling model-specific tuning within a single grid search.
3. Hyperparameter Tuning:
	- RandomizedSearchCV is used to optimize model hyperparameters efficiently with cross-validation (KFold).
4. Model Selection and Evaluation:
	- The model with the best hyperparameters is selected and evaluated using cross_val_score for accuracy measurement.
	- Accuracy results are printed, and predictions are saved in submission.csv for easy submission.

### Data

This project uses data from the Kaggle competition Titanic: Machine Learning from Disaster. Please download train.csv and test.csv files from the competition page to use this script.

Contributing:

Feel free to open issues or submit pull requests if you’d like to contribute. Please ensure any pull requests are thoroughly tested.

