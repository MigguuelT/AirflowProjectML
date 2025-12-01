from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

target_path = 'data/projeto-ml/processed/processed.csv'

target = 'LeaveOrNot'

def make_test_train(df):
    x = df.drop(columns=[target])
    y = df[target]
    return train_test_split(x, y, test_size=0.3, random_state=125)

def train_logistic(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def train_randomforest(x_train, y_train):
    model = RandomForestClassifier(random_state=125)
    model.fit(x_train, y_train)
    return model

def train_bayes(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model
