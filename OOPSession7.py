import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

class IrisModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.species_mapping = None  # To store species name mapping

    def load_and_clean_data(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        self.species_mapping = dict(enumerate(iris.target_names))
        df["target"] = [self.species_mapping[label] for label in iris.target]

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        X = df.drop(columns=["target"])
        y = df["target"]
        return X, y

    def train_model(self):
        X, y = self.load_and_clean_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    def save_model(self, filename="iris_rf_model.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Model saved as {filename}")

    def load_model(self, filename="iris_rf_model.pkl"):
        with open(filename, "rb") as file:
            self.model = pickle.load(file)
        print(f"Model loaded from {filename}")

iris_rf = IrisModel()
iris_rf.train_model()
iris_rf.save_model()

loaded_model = IrisModel.load_model()