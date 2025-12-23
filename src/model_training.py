from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data
import pickle

def train_models():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(lr, f)

    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    return X_test, y_test, lr, rf
