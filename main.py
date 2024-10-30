# Importing necessary libraries for data manipulation, visualization, and machine learning
import numpy as np  # For linear algebra operations
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For data processing and handling CSV files
from sklearn.compose import ColumnTransformer  # For column-wise transformations
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding and scaling features
from sklearn.model_selection import RandomizedSearchCV  # For hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier model
from sklearn.model_selection import KFold, cross_val_score  # For cross-validation
from sklearn.pipeline import Pipeline  # For creating machine learning pipelines
from sklearn.linear_model import LogisticRegression  # Logistic regression model
import xgboost as xgb  # XGBoost classifier
import warnings  # For managing warnings
from sklearn.exceptions import ConvergenceWarning  # Handling convergence warnings

# Loading training and testing datasets from CSV files
data_train = pd.read_csv("data/train.csv")  # Load training data
data_test = pd.read_csv("data/test.csv")  # Load testing data

# Displaying summary information for each dataset (column data types, non-null counts, etc.)
data_train.info(), data_test.info()

# Checking for any missing values in both datasets
data_train.isnull().sum(), data_test.isnull().sum()

# Data Engineering Section

# Visualizing the Age distribution across different passenger classes

# Plotting Age distribution for all passengers in the training data
data_train["Age"].plot(kind='hist', title="Age Distribution for All Classes")

# Plotting Age distribution for each Passenger Class (Pclass 1, 2, 3)
data_train[data_train["Pclass"] == 1]["Age"].plot(kind='hist', title="Age Distribution for Pclass 1")
data_train[data_train["Pclass"] == 2]["Age"].plot(kind='hist', title="Age Distribution for Pclass 2")
data_train[data_train["Pclass"] == 3]["Age"].plot(kind='hist', title="Age Distribution for Pclass 3")

# Displaying the histograms
plt.show()

# Calculating and printing the median Age for each combination of Passenger Class and Gender in training data
for i in range(1, 4):  # Loop through each Passenger Class (1, 2, 3)
    for s in ['male', 'female']:  # Loop through each gender (male, female)
        # Compute the median Age for the current Class and Gender combination
        tmp = data_train[(data_train['Pclass'] == i) & (data_train['Sex'] == s)]['Age'].median()

        # Display the calculated median
        print(f"The median for Pclass={i} and Sex={s} is {tmp}")

# Repeating the median Age calculation for the test dataset
for i in range(1, 4):
    for s in ['male', 'female']:
        tmp = data_test[(data_test['Pclass'] == i) & (data_test['Sex'] == s)]['Age'].median()

        print(f"The median for Pclass={i} and Sex={s} is {tmp}")

# Filling missing Age values in training and testing datasets based on median Age for each Sex and Passenger Class group
data_train['Age'] = data_train.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
data_test['Age'] = data_test.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# Filling missing values in the Embarked column with the most common embarkation point ('S') in the training data
data_train['Embarked'] = data_train['Embarked'].fillna('S')

# Identifying rows in test dataset with missing Fare values
print(data_test[data_test['Fare'].isnull()])

# Plotting Fare distribution for passengers in Passenger Class 3 in test dataset
data_test[(data_test['Pclass'] == 3)]['Fare'].plot(kind='hist')
plt.show()

# Calculating the median Fare for Passenger Class 3 and filling missing Fare values with this median
medfare = data_test[(data_test['Pclass'] == 3)]['Fare'].median()
data_test['Fare'] = data_test['Fare'].fillna(medfare)

# Feature Engineering Section

# Creating a new feature Family_Size as a sum of siblings/spouses and parents/children columns plus one (self)
data_train['Family_Size'] = data_train['SibSp'] + data_train['Parch'] + 1
data_test['Family_Size'] = data_test['SibSp'] + data_test['Parch'] + 1

# Creating a new feature Ticket_Frequency showing the count of each unique ticket in the training and test data
data_train['Ticket_Frequency'] = data_train.groupby('Ticket')['Ticket'].transform('count')
data_test['Ticket_Frequency'] = data_test.groupby('Ticket')['Ticket'].transform('count')

# Preprocessing and Feature Selection Section

# Combining training and testing data for consistency in preprocessing
data_all = pd.concat([data_train, data_test])

# Dropping columns that will not be used in the model, including the target column Survived (only present in training data)
X = data_all.drop(columns=['Survived', 'Name', 'Cabin', 'PassengerId'])

# Identifying categorical and numerical columns for further processing
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# Checking correlations of numerical columns with the Age column in the training dataset
print(data_train[numerical_cols].corr()['Age'])

y = data_train["Survived"]

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),       # Scaling numerical data
        ("cat", OneHotEncoder(), categorical_cols)       # Encoding categorical data
    ]
)

# Apply transformations to the feature set
df_all = preprocessor.fit_transform(X)
df_train = df_all[:891]
df_test = df_all[891:]

X.isnull().sum()

# Hyperparameter Tuning and Cross-Validation

# Importing required libraries for model selection, ensemble methods, linear models, and warnings
from sklearn.model_selection import RandomizedSearchCV  # For randomized hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier     # Random Forest classifier
from sklearn.model_selection import KFold, cross_val_score  # KFold for cross-validation, cross_val_score for evaluation
from sklearn.pipeline import Pipeline                  # To create pipeline of transformations and models
from sklearn.linear_model import LogisticRegression    # Logistic regression model
import xgboost as xgb                                  # XGBoost model for classification
import warnings
from sklearn.exceptions import ConvergenceWarning      # ConvergenceWarning to handle warning filtering

# Creating a pipeline with a placeholder for the classifier, allowing easy model substitution
pipeline = Pipeline(steps=[
    ('classifier', 'passthrough')  # Initial classifier set to 'passthrough' for flexibility in the pipeline
])

# Setting up KFold cross-validation with 5 splits, shuffling the data to randomize
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Defining the hyperparameter grid for RandomizedSearchCV with multiple classifiers and their respective parameters
param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000)],   # Logistic Regression with max iterations set to 1000
        'classifier__C': [0.01, 0.1, 1.0]                    # Regularization parameter options for Logistic Regression
    },
    {
        'classifier': [RandomForestClassifier()],            # Random Forest model option
        'classifier__n_estimators': [100, 110],              # Number of trees in the Random Forest
        'classifier__max_depth': [20, 25, 30],               # Maximum depth of the trees in the Random Forest
    },
    {
        'classifier': [xgb.XGBClassifier()],                 # XGBoost model option
        'classifier__n_estimators': [100, 110],              # Number of trees in XGBoost
        'classifier__learning_rate': [0.01, 0.1]             # Learning rate options for XGBoost
    }
]

# Ignoring all warnings, particularly useful for suppressing convergence warnings
warnings.filterwarnings("ignore")

# Setting up RandomizedSearchCV to find the best hyperparameters, with 25 iterations and accuracy scoring
random_search = RandomizedSearchCV(pipeline, param_grid, n_iter=25, cv=kf, scoring='accuracy')
random_search.fit(df_train, y)  # Fitting the model with the training data

# Printing the best parameters and the best score obtained from RandomizedSearchCV
print(f"Best parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")

# Extracting the best model from RandomizedSearchCV for further use
best_model = random_search.best_estimator_

# Evaluating the best model using cross-validation and printing individual fold results
cv_res = cross_val_score(best_model, df_train, y, cv=kf, scoring='accuracy')

# Displaying the accuracy of each fold and the mean accuracy across all folds
print(f"Cross-Validation Accuracies: {cv_res}")
print(f"Mean Accuracy: {cv_res.mean()}")

## Testing the best model and preparing the results for submission

# Generating predictions on the test data
predictions = best_model.predict(df_test)

# Creating a DataFrame with predictions formatted for Kaggle submission
output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})
output['Survived'] = output['Survived'].astype(int)  # Ensuring 'Survived' column is an integer
output.to_csv('submission.csv', index=False)  # Saving predictions to CSV for Kaggle submission

print("Your submission was successfully saved!")