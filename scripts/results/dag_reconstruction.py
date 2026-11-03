"""DAG-reconstruction appendix figure.

Samples synthetic datasets from known DAGs, runs zero-shot ExplainerPFN, turns
the mean absolute attribution of every (target, feature) pair into a directed
graph, and compares it with the true DAG using graph edit distance.

The GED search is exponential, so the 50 datasets are processed in parallel and
the edge sweep is restricted to the range shown in the figure (1-18 edges).

Outputs (``notebooks/paper_results/``):
    figures/dag_ground_truth.pdf
    figures/dag_predicted.pdf
    figures/dag_graph_edit_distance.pdf
    pickles/dag_reconstruction.pkl
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    FIGURES_DIR,
    PICKLES_DIR,
    add_repo_to_path,
    ensure_results_dirs,
    resolve_model_path,
)

add_repo_to_path()

from explainerpfn.base import ExplainerPFN  # noqa: E402
from explainerpfn.train import DAGGenerator  # noqa: E402

RANDOM_STATE = 42
N_DATASETS = 50
N_SAMPLES = 1000
MAX_EDGES = 18
GED_TIMEOUT = 60


def sample_dag_dataframe(generator, n_samples):
    scm = generator.generate_dataset(include_shap=False)["scm"]
    df, *_ = generator._generate_data_from_dag(scm.dag, n_samples)
    return df, scm.dag


def clean_frame(df):
    df = df.drop(columns=df.columns[-1])  # drop the target column
    df.columns = df.columns.map(lambda name: str(name).split("_")[1])
    return df.T.sort_index().T


def get_all_explanations(xai_model, df):
    explanations = {}
    for target_col in df.columns:
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        xai = deepcopy(xai_model)
        xai.fit(X, y)
        values = xai.predict(X, y)
        values = xai.apply_correction(
            y, values - values.mean(axis=0), kind=["statistical", "additive"]
        )
        explanations[target_col] = pd.DataFrame(
            values, columns=df.drop(columns=[target_col]).columns
        )
    return explanations


def edge_list(explanations):
    """Rank every (source, target) attribution pair by mean absolute value."""
    edges = []
    for target, exp_df in explanations.items():
        for source, weight in exp_df.abs().mean().items():
            if source == target:
                continue
            edges.append((float(weight), str(source), str(target)))
    edges.sort(key=lambda edge: edge[0], reverse=True)
    return edges


def dag_from_edges(edges, n_edges):
    graph = nx.DiGraph()
    for _, source, target in edges[:n_edges]:
        graph.add_edge(source, target)
    return graph


def _ged_task(task):
    gt_edges, predicted_edges, n_edges = task
    gt = dag_from_edges(gt_edges, len(gt_edges)).to_undirected()
    predicted = dag_from_edges(predicted_edges, n_edges).to_undirected()
    try:
        value = nx.graph_edit_distance(gt, predicted, timeout=GED_TIMEOUT)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float("nan") if value is None else float(value)


def edit_distances(gt_edges_list, predicted_edges_list, n_edges, pool):
    tasks = [
        (gt_edges_list[i], predicted_edges_list[i], int(n))
        for i in range(len(predicted_edges_list))
        for n in n_edges
    ]
    values = np.asarray(pool.map(_ged_task, tasks), dtype=float)
    return values.reshape(len(predicted_edges_list), len(n_edges))


def plot_dag(graph, colors=None):
    colors = colors if colors is not None else "#A0CBE2"
    plt.figure(figsize=(4, 4))
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, arrows=True, node_size=700, node_color=colors)
    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels={k: f"{v:.2f}" for k, v in labels.items()}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--max-edges", type=int, default=MAX_EDGES)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() - 2, 48))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ensure_results_dirs()
    generator = DAGGenerator(
        n_features_range=[6, 8],
        n_samples_range=[N_SAMPLES, N_SAMPLES],
        n_dags_range=[4, 4],
        nodes_per_dag_range=[2, 4],
        linear_dag_prob=1.0,
        random_state=RANDOM_STATE,
    )

    # --- Example used for the ground-truth / reconstructed panels ---
    df, dag = sample_dag_dataframe(generator, N_SAMPLES)
    df = clean_frame(df)
    model = ExplainerPFN(model_path=resolve_model_path(args.model_path), device=args.device)
    explanations = get_all_explanations(model, df)

    all_nodes = list(dag.nodes)
    nodes_in_data = [int(col) for col in df.columns]
    nodes_not_in_data = [node for node in all_nodes if node not in nodes_in_data]
    palette = {
        **{node: "#CA9E98" for node in nodes_not_in_data},
        **{node: "#A0CBE2" for node in nodes_in_data},
    }
    plot_dag(dag, colors=[palette[node] for node in dag.nodes])
    plt.savefig(FIGURES_DIR / "dag_ground_truth.pdf")
    plt.close()

    plot_dag(dag_from_edges(edge_list(explanations), 7))
    plt.savefig(FIGURES_DIR / "dag_predicted.pdf")
    plt.close()

    # --- 50 datasets: true DAGs vs attribution graphs ---
    gt_edges_list, predicted_edges_list = [], []
    print(f"Sampling {args.n_datasets} datasets and computing explanations...")
    for _ in range(args.n_datasets):
        frame, true_dag = sample_dag_dataframe(generator, N_SAMPLES)
        frame = clean_frame(frame)
        true_dag = nx.relabel_nodes(true_dag, str)
        gt_edges_list.append([(0.0, str(u), str(v)) for u, v in true_dag.edges()])
        predicted_edges_list.append(edge_list(get_all_explanations(model, frame)))

    n_edges = np.arange(1, args.max_edges + 1)
    print(f"Computing graph edit distances with {args.workers} workers...")
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        all_edit = edit_distances(gt_edges_list, predicted_edges_list, n_edges, pool)
        rng = np.random.default_rng(RANDOM_STATE)
        random_edges = [
            sorted(
                ((float(w), s, t) for (_, s, t), w in zip(edges, rng.uniform(size=len(edges)))),
                key=lambda edge: edge[0],
                reverse=True,
            )
            for edges in predicted_edges_list
        ]
        random_edit = edit_distances(gt_edges_list, random_edges, n_edges, pool)

    with open(PICKLES_DIR / "dag_reconstruction.pkl", "wb") as handle:
        pickle.dump(
            {
                "n_edges": n_edges,
                "gt_edges": gt_edges_list,
                "predicted_edges": predicted_edges_list,
                "all_edit_distances": all_edit,
                "random_edit_distances": random_edit,
            },
            handle,
        )

    sns.set_theme(style="whitegrid", font="serif", font_scale=1.6)
    for values, label in ((all_edit, "ExplainerPFN"), (random_edit, "Random")):
        frame = pd.DataFrame(values, columns=n_edges).melt(
            var_name="# Edges", value_name="Edit Distance"
        )
        sns.lineplot(
            data=frame, x="# Edges", y="Edit Distance", errorbar="sd", label=label
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dag_graph_edit_distance.pdf", dpi=300)
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
