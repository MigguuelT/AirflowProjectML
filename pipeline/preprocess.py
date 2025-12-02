import pandas as pd

selected_features = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender',
       'EverBenched', 'ExperienceInCurrentDomain']

encoded_features = ['Education', 'City', 'Gender', 'EverBenched']

target = 'LeaveOrNot'

def preprocess(df, inference=False):
    if not inference:
        df = df[selected_features + [target]]
    else:
        df = df[selected_features]
    df = pd.get_dummies(df, columns=encoded_features)
    return df