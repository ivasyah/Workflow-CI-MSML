import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

mlflow.set_experiment("Loan Prediction CI")

def main():
    df = pd.read_csv("loan_prediction_preprocessed.csv")

    X = df.drop(columns=["loan_approved"])
    y = df["loan_approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.sklearn.autolog()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

if __name__ == "__main__":
    main()
