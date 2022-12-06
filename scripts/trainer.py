# Import Neptune
import os, joblib
import pandas as pd
import numpy as np

import wandb
from wandb import AlertLevel

wandb.login(key="cf946a0ea9f104db10794da536d8d192de788614")
wandb.init(project="bank_loan_eligibility")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score

# --- COLUMNS NAMES --- #
COLUMNS_NAMES = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'
]

categorical_fts = [
    "gender", "married", "education", "self_employed",
    "credit_history", "property_area"
]
numerical_fts_eng = ['dependents', 'loanamount', 'totalincome', 'loan_amount_term_months']
numerical_fts = ['dependents', 'applicantincome', 'coapplicantincome', 'loanamount', 'loan_amount_term']
# --------------------- #

# ---- Multi Columns Label Enconder ---- #
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
# ---------------------------------------- #

# //////////////////////////////////////////////////// #

model_path = os.path.join("./models/clf_model.joblib")
train_set_path = os.path.join("./data/train.txt")

_model = joblib.load(model_path)
_train_set = pd.read_csv(train_set_path, low_memory=False)

# ///////////////////////////////////////////////////// #

def preprocessor(data = _train_set):
    assert all(x in COLUMNS_NAMES for x in data.columns)

    data = data.drop(["Loan_ID"], axis=1)
    data = data.dropna()
    data.columns = data.columns.str.lower()
    data["dependents"] = pd.to_numeric(data["dependents"].map({"3+": "3", "0": "0", "1": "1", "2": "2"}))

    X = data.drop(["loan_status"], axis=1)
    y = data["loan_status"]

    return X, y

def splitter_balancer(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sm = SMOTE(random_state=0)

    X_train, y_train = sm.fit_resample(X_train, y_train)
    X_test, y_test = sm.fit_resample(X_test, y_test)

    return X_train, X_test, y_train, y_test

def feature_encoder(X, y, ctg_fts):
    le = LabelEncoder()

    onehot_df = pd.get_dummies(X[ctg_fts[-1]])
    label_df = MultiColumnLabelEncoder(columns = ctg_fts[:-1]).fit_transform(X)
    label_df = label_df.drop(ctg_fts[-1], axis=1)

    X = label_df.join(onehot_df)
    y = le.fit_transform(y)

    return X, y

def feature_engineering(X, y, pipe, threshold, unit_test=True):
    # Converting the scale of loan term from months to years
    X["loan_amount_term_months"] = (X["loan_amount_term"] / 12)

    # Combining applicant and co-applicant income to get the total income per application
    X["totalincome"] = X["applicantincome"] + X["coapplicantincome"]

    # Dropping the columns as we created a new column which captures the same information
    X.drop(columns=["applicantincome", "coapplicantincome"], inplace=True)

    X_train, X_test, y_train, y_test = splitter_balancer(X, y)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_probas = pipe.predict_proba(X_test)
    fts_names = X_train.columns

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    if not unit_test:
        return X, y

    if precision < threshold: wandb.alert(title="Low precision",
                                          text=f"Precision {precision} is below the acceptable threshold {threshold}",
                                          level=AlertLevel.WARN)

    assert precision >= threshold, f"Low precision, precision of {precision} is below acceptable threshold {threshold}"

    wandb.summary["precision"] = precision
    wandb.summary["recall"] = recall
    wandb.sklearn.plot_feature_importances(pipe["classifier"], fts_names)

    labels = ["One", "Zero"]
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
    wandb.sklearn.plot_roc(y_test, y_probas, labels)

def hyperparameter_optimizer(pipe, X_eng, y, threshold):
    params = {'C': 10,
        'class_weight': 'balanced',
        'dual': False,
        'multi_class': 'auto',
        'penalty': 'l2',
        'solver': 'lbfgs'
    }

    X_train, X_test, y_train, y_test = splitter_balancer(X_eng, y)

    pipe[-1].set_params(**params)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_probas = pipe.predict_proba(X_test)
    labels = ["One", "Zero"]
    fts_names = X_train.columns

    precision = precision_score(y_test, y_pred)
    # precision = 0.45
    recall = recall_score(y_test, y_pred)

    if precision < threshold: wandb.alert(title="Low precision",
                                          text=f"Precision {precision} is below the acceptable threshold {threshold}",
                                          level=AlertLevel.WARN)

    assert precision >= threshold, f"Low precision, precision of {precision} is below acceptable threshold {threshold}"

    wandb.summary["precision"] = precision
    wandb.summary["recall"] = recall
    wandb.sklearn.plot_feature_importances(pipe["classifier"], fts_names)
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
    wandb.sklearn.plot_roc(y_test, y_probas, labels)

def pipeline_constructor(num_fts_l):
    column_transformer = ColumnTransformer([
        ("scaler", StandardScaler(), num_fts_l)
    ], remainder="passthrough")

    return Pipeline([("datafeed", column_transformer), ("classifier", _model)])

def test_feature_eng_against_benchmark():
    BENCHMARK_LIMIT = 0.75
    lg_pipe = pipeline_constructor(numerical_fts_eng)

    # Preprocess raw data
    X, y = preprocessor()
    X, y = feature_encoder(X, y, ctg_fts=categorical_fts)

    # Unit testing on first feature engineering model improvement exp run #
    feature_engineering(X, y, lg_pipe, threshold=BENCHMARK_LIMIT)

# def test_hps_optimization_against_benchmark(ft_eng=False):
#     BENCHMARK_LIMIT = 0.75
#     lg_pipe = pipeline_constructor(numerical_fts_eng) if ft_eng else pipeline_constructor(numerical_fts)

#     # Preprocess raw data
#     X, y = preprocessor()
#     X, y = feature_encoder(X, y, ctg_fts=categorical_fts)

#     # Unit testing on second model improvement hyperparamter optimization feature exp run #
#     hyperparameter_optimizer(lg_pipe, X, y, threshold=BENCHMARK_LIMIT)

def test_combined_fts_against_benchmark():
    BENCHMARK_LIMIT = 0.75
    lg_pipe = pipeline_constructor(numerical_fts_eng)

    # Preprocess raw data
    X, y = preprocessor()
    X, y = feature_encoder(X, y, ctg_fts=categorical_fts)

    # Unit testing on first and second model improvement features exp run #
    X_eng, y = feature_engineering(X, y, lg_pipe, threshold=BENCHMARK_LIMIT, unit_test=False)
    hyperparameter_optimizer(lg_pipe, X_eng, y, threshold=BENCHMARK_LIMIT)
