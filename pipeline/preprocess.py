import pandas as pd

selected_features = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender',
       'EverBenched', 'ExperienceInCurrentDomain']

encoded_features = ['Education', 'City', 'Gender', 'EverBenched']

def preprocess(df):
    df = pd.get_dummies(df, columns=encoded_features)
    return df