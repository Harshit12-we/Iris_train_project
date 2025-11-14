import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib

DATA_PATH = "iris.csv"
feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
label_col = "Species"
RANDOM_STATE = 42
MODEL_OUTPUT = "iris_model.joblib"


def load_data(path):
    if not os.path.exists(path):
        print(f"Error: data file not found at {path}")
        sys.exit(1)
    return pd.read_csv(path)


def prepare_data(df):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    df = df.dropna(subset=feature_cols + [label_col])
    X = df[feature_cols].values
    y = df[label_col].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def build_pipeline_and_params():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    param_grid = [
        {
            "clf": [LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)],
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["lbfgs", "saga"]
        },
        {
            "clf": [RandomForestClassifier(random_state=RANDOM_STATE)],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10]
        },
        {
            "clf": [SVC(probability=True, random_state=RANDOM_STATE)],
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear"]
        }
    ]

    return pipeline, param_grid


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data(DATA_PATH)
    X, y, label_encoder = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline, param_grid = build_pipeline_and_params()

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)

    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print("5-fold CV scores:", np.round(cv_scores, 4))
    print("CV mean accuracy:", np.round(cv_scores.mean(), 4))

    joblib.dump({
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_cols
    }, MODEL_OUTPUT)

    example = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred_class = label_encoder.inverse_transform(best_model.predict(example))
    print("Example prediction:", pred_class[0])


if __name__ == "__main__":
    main()