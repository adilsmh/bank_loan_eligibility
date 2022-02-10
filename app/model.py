import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle as pck

from pydantic import BaseModel


class Client(BaseModel):
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


class loan_model:
    def __init__(self):
        self.model_fname_ = "../models/model_pkl"
        self.model = pck.load(open(self.model_fname_, 'rb'))

    def preprocessing(data_raw):
        data_raw = [data_raw.dict()]
        df = pd.DataFrame(data_raw)

        df = df.dropna()  # drop NaN / missing values
        df.columns = df.columns.str.lower()  # renaming columns for better readability

        # define continuous / categorical features
        continuous_fts = []
        categorical_fts = []

        for col in df.columns:
            if df[col].dtype == "int64" or df[col].dtype == "float64":
                continuous_fts.append(col)
            else:
                categorical_fts.append(col)

        le = LabelEncoder()
        df[categorical_fts] = df[categorical_fts].apply(le.fit_transform)

        return df

    def predict_eligibility(self, data_pcd):
        prediction = self.model.predict(data_pcd)
        probability = np.round(self.model.predict_proba(data_pcd).max(), 2)

        return prediction[0], probability
