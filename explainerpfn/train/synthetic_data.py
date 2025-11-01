import pickle
import numpy as np
import pandas as pd
import networkx as nx

from tqdm.auto import tqdm
from explainerpfn.train.utils import _check_random_state, postprocess_synthetic_data
from explainerpfn.train._directed_acyclical_graphs import (
    generate_synthetic_data,
    redirection_sampling_dag,
    plot_dag,
    random_join,
)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import shap


class SyntheticDataGenerator:
    def __init__(
        self,
        classifier=None,
        n_train_samples=[1, 2048],
        n_test_samples=128,
        n_features=[1, 160],
        n_dags=[1, 10],
        n_nodes=[2, 10],  # Can be overriden during parameter sampling
        edge_prob=[0, 0.25],
        # init_distr=["uniform", "normal", "mixed"],
        # max_cells=75000,
        alpha=[0, 5],
        beta=[0, 5],
        successor_as_target=[True, False],
        random_state=None,
        n_jobs=-1,
    ):
        self.classifier = classifier
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.n_features = n_features
        # self.max_cells = max_cells
        self.n_dags = n_dags
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.alpha = alpha
        self.beta = beta
        self.successor_as_target = successor_as_target
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._rng = _check_random_state(self.random_state)

    def __repr__(self):
        content = {
            k: len([e for e in v if e is not None])
            for k, v in self.__dict__.items()
            if k.endswith("_")
        }
        return self.__class__.__name__ + "(" + str(content)[1:-1] + ")"

    def _sample_params(self):
        params = {
            "edge_prob": self._rng.uniform(*self.edge_prob),
            "n_train_samples": self._rng.integers(*self.n_train_samples),
            "n_test_samples": self.n_test_samples,
            "n_features": int(
                np.round(
                    self._rng.beta(a=0.95, b=10)
                    * (self.n_features[1] - self.n_features[0])
                    + self.n_features[0]
                )
            ),
            "n_dags": int(
                self._rng.beta(a=0.95, b=5) * (self.n_dags[1] - self.n_dags[0])
                + self.n_dags[0]
            ),
            "successor_as_target": self._rng.choice(self.successor_as_target),
        }

        params["n_nodes"] = self._rng.integers(*self.n_nodes, size=params["n_dags"])
        if params["n_nodes"].sum() < params["n_features"]:
            # TODO: Add warning
            params["n_nodes"] = np.ceil(
                params["n_nodes"] / params["n_nodes"].sum() * params["n_features"]
            ).astype(int)

        # Check if the same parameters were already sampled
        if (
            hasattr(self, "params_")
            and len(self.params_) > 0
            and (self.describe().iloc[:, :-1] == pd.Series(params).iloc[:-1])
            .all(axis=1)
            .any()
        ):
            return self._sample_params()

        return params

    def describe(self):
        df_params = pd.DataFrame(self.params_)
        df_params["n_nodes"] = df_params["n_nodes"].apply(sum)
        return df_params

    def plot_dag(self, dag):
        """
        `dag` can be a networkx graph or an index of the generated dag.
        """
        dag = self.dags_[dag] if isinstance(dag, int) else dag
        return plot_dag(dag)

    def _dataset(
        self,
        n_nodes,
        n_dags,
        edge_prob,
        # init_distr,
        n_train_samples,
        n_test_samples,
        n_features,
        successor_as_target,
    ):
        dag = redirection_sampling_dag(
            n_nodes=n_nodes[0], edge_prob=edge_prob, random_state=self._rng
        )
        for i in range(1, n_dags):
            dag = random_join(
                dag,
                redirection_sampling_dag(
                    n_nodes=n_nodes[i], edge_prob=edge_prob, random_state=self._rng
                ),
                edge_prob=edge_prob,
                random_state=self._rng,
            )

        data, dag_data = generate_synthetic_data(
            dag,
            # init_distr=init_distr,
            return_dag_data=True,
            random_state=self._rng,
            n_samples=n_train_samples + n_test_samples,
        )
        df = postprocess_synthetic_data(
            data,
            dag_data,
            n_features=max(4, min(n_features, len(dag.nodes))),
            target_is_successor=successor_as_target,
            random_state=self._rng,
        )

        return df, dag

    def _explanations(self, df, dag):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=self.random_state
        )

        # df_exp = df.copy().iloc[:, :-1] * np.nan

        # nodes = [int(col.split("_")[-1]) for col in df.columns]
        # relevant_nodes = [n for n in nodes if nx.has_path(dag, n, nodes[-1])]
        # relevant_mask = df.columns.map(
        #     lambda x: int(x.split("_")[-1]) in relevant_nodes
        # ).to_numpy()[:-1]
        # df_exp.loc[:, ~relevant_mask] = 0

        # if relevant_mask.sum() == 0:
        #     relevant_mask += True

        # X_train = X_train.loc[:, relevant_mask]
        # X_test = X_test.loc[:, relevant_mask]

        clf = MLPClassifier(
            max_iter=2000,
            alpha=0.05,
            learning_rate="invscaling",
            random_state=self.random_state,
        )
        clf.fit(X_train, y_train)
        # y_pred = clf.predict_proba(X.loc[:, relevant_mask])[:, -1]
        y_pred = clf.predict_proba(X)[:, -1]

        xai = shap.Explainer(model=clf.predict_proba, masker=X_train)
        # shap_explanation = xai(X.loc[:, relevant_mask]).values[:, :, 1]
        shap_explanation = xai(X).values[:, :, 1]
        # df_exp.loc[:, relevant_mask] = shap_explanation
        df_exp = pd.DataFrame(
            shap_explanation, columns=X.columns, index=X.index
        )
        return df_exp, y_pred

    def generate(self, n_datasets=1, verbose=False):

        if not hasattr(self, "params_"):
            self.params_ = []
            self.X_ = []
            self.y_ = []
            self.y_pred_ = []
            self.dags_ = []
            self.explanations_ = []

        iter_ = tqdm(range(n_datasets)) if verbose else range(n_datasets)
        for _ in iter_:
            params = self._sample_params()
            df, dag = self._dataset(**params)
            explanations, y_pred = self._explanations(df, dag)

            self.params_.append(params)
            self.X_.append(df.iloc[:, :-1])
            self.y_.append(df.iloc[:, -1])
            self.y_pred_.append(y_pred)
            self.dags_.append(dag)
            self.explanations_.append(explanations)

        return self

    def save_data(self, path):
        data = {
            "params": self.params_,
            "X": self.X_,
            "y": self.y_,
            "y_pred": self.y_pred_,
            "dags": self.dags_,
            "explanations": self.explanations_,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_data(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        if not hasattr(self, "params_"):
            self.params_ = []
            self.X_ = []
            self.y_ = []
            self.y_pred_ = []
            self.dags_ = []
            self.explanations_ = []

        self.params_ += data["params"]
        self.X_ += data["X"]
        self.y_ += data["y"]
        self.y_pred_ += data["y_pred"]
        self.dags_ += data["dags"]
        self.explanations_ += data["explanations"]
        return self
