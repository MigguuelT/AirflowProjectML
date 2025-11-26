from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

target_path = 'data/projeto-ml/processed/processed.csv'

target = 'LeaveOrNot'

def make_test_train(df):
    x = df.drop(columns=[target])
    y = df[target]
    return train_test_split(x, y, test_size=0.3, random_state=125)

def train_model(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model
