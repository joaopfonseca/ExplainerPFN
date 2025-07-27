from mlresearch.datasets import ContinuousCategoricalDatasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def get_toy_data():
    """
    Get toy data for testing purposes.
    """
    datasets = ContinuousCategoricalDatasets(names=["GERMAN CREDIT"]).download()
    df = datasets[0][-1]
    continuous_columns = df.columns.drop("target").str.startswith("cat_")
    X, y = df.drop(columns=["target"]), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    return X_train, X_test, y_train, y_test, continuous_columns


def get_ml_pipeline(X_train, y_train, continuous_columns):
    """
    Get a machine learning pipeline for testing purposes.
    """

    clf = make_pipeline(
        ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(), continuous_columns),
                ("num", "passthrough", ~continuous_columns),
            ]
        ),
        RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    )

    clf.fit(X_train, y_train)
    return clf
