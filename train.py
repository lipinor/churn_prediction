import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

from sklearn.metrics import (precision_recall_curve, 
                             accuracy_score,
                             auc)

import pickle


# Reading data
data = pd.read_csv('data/Churn_Modelling.csv')

data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True) # Removing the RowNumber and CustomerId columns

# Removing target from train dataset
drop_cols = [
    'Exited',
    'EstimatedSalary',
    'Balance',
    'HasCrCard'
]

X_train = data.drop(drop_cols, axis=1)

# Creating target series
y_train = data['Exited']

X_cat_cols = list(X_train.loc[:, X_train.dtypes == object].columns)
X_num_cols = list(X_train.loc[:, X_train.dtypes != object].columns)


def make_pipeline(model, df):
    """Create a pipeline for a model."""

    cat_transf = OneHotEncoder(sparse_output=False,
                               handle_unknown="ignore")

    std_scaler = StandardScaler()

    # Determining categorical and numerical columns
    df_cat_cols = list(df.loc[:, df.dtypes == object].columns)
    df_num_cols = list(df.loc[:, df.dtypes != object].columns)

    transformer = ColumnTransformer(transformers=[("cat", cat_transf, df_cat_cols),
                                                  ("num_scaler", std_scaler, df_num_cols)],
                                    remainder='passthrough'
                                   )

    steps = [("transformer", transformer),
             ("model", model)]
    
    return Pipeline(steps)


def score_model(pipeline, X, y_true):
    """Score for a given pipeline using Accuracy, ROC AUC, Precision and Recall"""
    predict = pipeline.predict(X)
    predict_proba = pipeline.predict_proba(X)[:, 1]
    precisions, recalls, _ = precision_recall_curve(y_true, predict_proba)

    metrics = {"precision_recall_auc": auc(recalls, precisions),
               "accuracy": accuracy_score(y_true, predict),
              }
    
    return metrics

def print_scores(score_dict):
    """Print the scores"""

    for key, value in score_dict.items():
        print(key, np.round(value, 4))


xgb = XGBClassifier(subsample= 0.8,
                    scale_pos_weight=1,
                    reg_lambda=0,
                    max_depth=3,
                    learning_rate=0.1,
                    gamma=1,
                    colsample_bytree=0.5,
                    random_state=15)

pipe_xgb = make_pipeline(xgb, X_train)

pipe_xgb.fit(X_train, y_train)

proba_train_xgb = pipe_xgb.predict_proba(X_train)[:,1]

score_train_xgb = score_model(pipe_xgb, X_train, y_train)

print_scores(score_train_xgb)

# Saving the model
with open('churn_model.bin', 'wb') as f_out:
    pickle.dump((pipe_xgb), f_out)