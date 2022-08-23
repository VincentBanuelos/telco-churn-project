import pandas as pd
import numpy as np
from scipy import stats
import math

import acquire
import prepare

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier






df = acquire.get_telco_data()
df = prepare.prep_telco(df)

train, validate, test = prepare.my_train_test_split(df, target = 'churn')

features = ['num_addons','partner','dependents','tenure']

X_train = train[features]
y_train = train.churn

X_val = validate[features]
y_val = validate.churn

X_test = test[features]
y_test = test.churn


def logrmodel(df):
    metrics = []

    for i in np.arange(0,1,.1):
        threshold = i
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_train)
        y_pred_proba = pd.DataFrame(y_pred_proba,
        columns = ['no_churn', 'churn'])
        
        v_y_pred_proba = model.predict_proba(X_val)
        v_y_pred_proba = pd.DataFrame(v_y_pred_proba,
        columns = ['no_churn', 'churn'])
        
        y_pred = (y_pred_proba.churn > i).astype(int)
        val_y_pred = (v_y_pred_proba.churn > i).astype(int)
        
        
        in_sample_accuracy = accuracy_score(y_train, y_pred)
        out_of_sample_accuracy = accuracy_score(y_val, val_y_pred)

        output = {
                "threshold": threshold,
                "train_accuracy": in_sample_accuracy,
                "validate_accuracy": out_of_sample_accuracy
            }

        metrics.append(output)
        
    dt = pd.DataFrame(metrics)
    dt["difference"] = dt.train_accuracy - dt.validate_accuracy
    return dt


def dtmodel(df):
    metrics = []

    for i in range(2,9):
        # Make the model
        depth = i
        model = DecisionTreeClassifier(max_depth=i)
        # Fit the model (on train and only train)
        model.fit(X_train, y_train)

        # Use the model
        # We'll evaluate the model's performance on train, first
        in_sample_accuracy = model.score(X_train, y_train)
        
        out_of_sample_accuracy = model.score(X_val, y_val)

        output = {
            "max_depth": depth,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
        }
        
        metrics.append(output)
        
    dt = pd.DataFrame(metrics)
    dt["difference"] = dt.train_accuracy - dt.validate_accuracy
    return dt.sort_values(by=['difference'])


def rfmodel(df):
    metrics = []
    max_depth = 11

    for i in range(1, max_depth):
        # Make the model
        depth = max_depth - i
        n_samples = i
        forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=123)

        # Fit the model (on train and only train)
        forest = forest.fit(X_train, y_train)

        # Use the model
        # We'll evaluate the model's performance on train, first
        in_sample_accuracy = forest.score(X_train, y_train)
        
        out_of_sample_accuracy = forest.score(X_val, y_val)

        output = {
            "min_samples_per_leaf": n_samples,
            "max_depth": depth,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
        }
        
        metrics.append(output)
        
    rfl = pd.DataFrame(metrics)
    rfl["difference"] = rfl.train_accuracy - rfl.validate_accuracy
    return rfl.sort_values(by=['difference'])

def knnmodel(df):
    models_acc = []

    for x in range(1,21):

        #make it
        knn = KNeighborsClassifier(n_neighbors=x)

        #fit it
        knn = knn.fit(X_train, y_train)

    # predict it
    # y_pred = knn.predict(X_train)
    # y_pred_val = knn.predict(X_validate)
        
        #score it
        acc = knn.score(X_train, y_train)
        acc_val = knn.score(X_val, y_val)
        difference = acc - acc_val
        models_acc.append([x, acc, acc_val,difference])
        
    knn_df = pd.DataFrame(models_acc, columns =['neighbors', 'train_accuracy',
                                    'validate_accuracy','difference'])
    return knn_df.sort_values(by=['difference'])

def score_models(X_train, y_train, X_val, y_val):
    '''
    Score multiple models on train and val datasets.
    Print classification reports to decide on a model to test.
    Return each trained model, so I can choose one to test.
    models = lr_model, dt_model, rf_model, kn_model.
    '''
    lr_model = LogisticRegression(random_state=123)
    dt_model = DecisionTreeClassifier(max_depth=2, random_state=123)
    rf_model = RandomForestClassifier(max_depth=8, min_samples_leaf=3, random_state=123)
    knn_model = KNeighborsClassifier(n_neighbors = 19)
    models = [lr_model, dt_model, rf_model, knn_model]
    for model in models:
        model.fit(X_train, y_train)
        actual_train = y_train
        predicted_train = model.predict(X_train)
        actual_val = y_val
        predicted_val = model.predict(X_val)
        print(model)
        print('')
        print('train score: ')
        print(classification_report(actual_train, predicted_train))
        print('val score: ')
        print(classification_report(actual_val, predicted_val))
        print('________________________')
        print('')
    return lr_model, dt_model, rf_model, knn_model