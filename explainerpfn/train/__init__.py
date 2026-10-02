"""Training-data generation for ExplainerPFN.

Online synthetic datasets with exact do-Shapley labels (Witter et al., 2026),
built on a copy of the DiffusionExplainerPFN generator. Restored to the
exact-label regime only (d <= 15); permutation-MC labels at d > 15 are
DiffusionExplainerPFN's scaling mechanism.
"""

from explainerpfn.train.dag_generators import DAGGenerator
from explainerpfn.train.do_shapley import DoShapley
from explainerpfn.train.scm import StructuralCausalModel
from explainerpfn.train.synthetic_data import SyntheticDataGenerator, TrainingBatchIterator

__all__ = [
    "SyntheticDataGenerator",
    "TrainingBatchIterator",
    "DAGGenerator",
    "DoShapley",
    "StructuralCausalModel",
]
