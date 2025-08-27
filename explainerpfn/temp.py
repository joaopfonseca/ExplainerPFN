from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mlresearch.datasets import BinaryDatasets


def train_random_forest(X_train_rf, y_train_rf, X_train, X_test):
    clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    clf.fit(X_train_rf, y_train_rf)

    y_train = clf.predict_proba(X_train)[:, 1]
    y_test = clf.predict_proba(X_test)[:, 1]
    return clf, y_train, y_test


def get_data(name="BANKNOTE AUTHENTICATION"):
    datasets = BinaryDatasets(names=[name]).download()
    df = datasets[name]

    df_train_rf = df.sample(500, replace=False, random_state=42)
    df_train = df.drop(df_train_rf.index).sample(500, replace=False, random_state=42)
    df_test = df.drop(df_train_rf.index).drop(df_train.index)
    return df, df_train_rf, df_train, df_test


def preprocess_data(df_train_rf, df_train, df_test):
    scaler = StandardScaler().fit(
        df_train_rf.loc[:, df_train_rf.columns.drop("target")].values
    )

    Xy = []
    for df_ in [df_train_rf, df_train, df_test]:
        df_ = df_.reset_index(drop=True).astype(float)
        df_.loc[:, df_.columns.drop("target")] = scaler.transform(
            df_.loc[:, df_.columns.drop("target")].values.astype(float)
        )
        X = df_.loc[:, df_.columns.drop("target")]
        y = df_["target"]
        Xy.append((X, y))

    (X_train_rf, y_train_rf), (X_train, _), (X_test, _) = Xy

    return (X_train_rf, y_train_rf), (X_train, None), (X_test, None)


def get_shap_values(X, model, check_additivity=True, masker=None):
    """
    Get SHAP values for the given model and data.
    """
    xai = shap.Explainer(model=model, masker=masker)

    shap_explanation_train = xai(X, check_additivity=check_additivity).values[:, :, -1]

    return shap_explanation_train


def analyze_correlation_results(
    df, df_shap, explanations, explanations_corrected, y_test, df_train
):
    correlations = {}

    # Correlation between input values and the predicted target
    correlations["input_values_vs_target"] = df.corr()["target"]

    # Correlation between shapley values and the predicted target
    correlations["shap_vs_target"] = df_shap.corr()["target"]

    # Correlation between PFN explanations and the predicted target
    df_explanations = pd.DataFrame(
        explanations.astype(float), columns=df_train.columns.drop("target")
    )
    df_explanations["target"] = y_test
    correlations["pfn_exp_vs_target"] = df_explanations.corr()["target"]

    # Correlation between the corrected PFN explanations and the predicted target
    df_explanations_corrected = pd.DataFrame(
        explanations_corrected.astype(float), columns=df_train.columns.drop("target")
    )
    df_explanations_corrected["target"] = y_test
    correlations["pfn_exp_corr_vs_target"] = df_explanations_corrected.corr()["target"]

    # Correlation between PFN explanations and SHAP values
    correlations["pfn_exp_vs_shap"] = pd.Series(
        [
            np.corrcoef(df_explanations.loc[:, i], df_shap.loc[:, i])[0, 1]
            for i in df_explanations.columns
        ],
        index=df_explanations.columns,
    )

    # Correlation between corrected PFN explanations and SHAP values
    correlations["pfn_exp_corr_vs_shap"] = pd.Series(
        [
            np.corrcoef(df_explanations_corrected.loc[:, i], df_shap.loc[:, i])[0, 1]
            for i in df_explanations_corrected.columns
        ],
        index=df_explanations_corrected.columns,
    )

    return pd.DataFrame(correlations)


def plot_explanations(df, df_explanations, df_explanations_corrected, df_shap):

    label_data = {
        "Input values": df.drop(columns="target"),
        "PFN explanations": df_explanations.drop(columns="target"),
        "Corrected PFN explanations": df_explanations_corrected.drop(columns="target"),
        "SHAP values": df_shap.drop(columns="target"),
    }

    fig, axes = plt.subplots(
        nrows=1, ncols=len(label_data), figsize=(6 * len(label_data), 6)
    )
    for ax, (label, data) in zip(axes, label_data.items()):
        data.plot.hist(
            bins=50,
            alpha=0.5,
            ax=ax,
        )
        ax.set_xlabel(f"{label} distribution")

    return fig, axes
