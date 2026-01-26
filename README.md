# ExplainerPFN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

ExplainerPFN is a novel explainability method. This project provides **fast,
zero-shot feature importance estimation** without requiring access to the
original prediction model, making it ideal for model-agnostic explainability in
tabular data scenarios.

## Key Features

- **Zero-shot Explainability**: Predict feature contributions without retraining or access to the original model
- **Fast Inference**: Leverages TabPFN's efficient transformer architecture for rapid explanations
- **SHAP-compatible**: Uses SHAP values as ground truth for validation and benchmarking
- **Model-agnostic**: Works with any black-box model's predictions
- **Customizable Corrections**: Multiple correction methods (statistical, linear, multiplicative, additive) to improve explanation quality
- **Fine-tuning Support**: Ability to fine-tune the foundation model on custom datasets
- **Comprehensive Metrics**: Built-in evaluation metrics for explanation quality (fidelity, consistency, sensitivity)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ExplainerPFN.git
cd ExplainerPFN

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy
- pandas
- scikit-learn
- PyTorch
- TabPFN 2.1.2
- SHAP
- matplotlib
- tqdm

## Quick Start

### Basic Usage

```python
from explainerpfn.model import ExplainerPFN
import numpy as np

# Initialize the explainer
explainer = ExplainerPFN(
    n_estimators=1,
    device="auto",  # Automatically selects GPU/CPU
    fit_mode="fit_with_cache"
)

# Fit on background data
# X: feature matrix (n_samples, n_features)
# y: target values or model predictions
explainer.fit(X_train, y_train)

# Get feature importance explanations
explanations = explainer.predict(X_test, y_test)

# Apply correction (optional but recommended)
corrected_explanations = explainer.apply_correction(
    y_test, 
    explanations, 
    kind=["statistical", "additive"]
)
```

### Fine-tuning on Custom Data

Note: Fine-tuning was working at some point but may require adjustments since
some things were modified in the meantime and we haven't tested this method
since.

```python
import torch.optim as optim

# Prepare optimizer
optimizer = optim.Adam(explainer.model_.parameters(), lr=1e-4)

# Fine-tune with SHAP values as ground truth
for epoch in range(num_epochs):
    loss = explainer.finetune(optimizer, X_train, y_train, shap_values)
    print(f"Epoch {epoch}: Loss = {loss}")

# Save the fine-tuned model
explainer.save_foundation_model("path/to/model.ckpt")
```

## Architecture

ExplainerPFN follows the sequence below for generating feature importance
explanations:

```
Input (X, y, feature_idx)
    ↓
Preprocessing Pipeline
    ├── Categorical encoding
    ├── Feature distribution reshaping
    └── Target transformations
    ↓
TabPFN Transformer Encoder
    ↓
Bar Distribution Logits
    ↓
Statistical Corrections
    ↓
Feature Importance Values
```

### Key Components

- **`ExplainerPFN`**: Main class implementing the explainer interface
- **Inference Engine**: Efficient batched inference with caching
- **Preprocessing**: Customizable feature transformations and ensembling
- **Metrics Module**: Evaluation tools for explanation quality
- **Training Module**: Synthetic data generation for pre-training

## Research & Experiments

The `notebooks/` directory contains Jupyter notebooks demonstrating:

1. **SHAP value estimation**: Comparison with traditional SHAP methods
2. **Cross-dataset evaluation**: Performance across multiple tabular datasets
3. **Ablation studies**: Component-wise analysis of the method
4. **Case studies**: Real-world application examples
5. **Synthetic data generation**: DAG-based synthetic data for training
6. **Model multiplicity analysis**: Understanding explanation variance

## Project Structure

```
ExplainerPFN/
├── explainerpfn/           # Main package
│   ├── base.py            # Core ExplainerPFN class
│   ├── inference.py       # Inference engine implementation
│   ├── preprocessing.py   # Data preprocessing utilities
│   ├── config.py          # Configuration settings
│   ├── metrics/           # Evaluation metrics
│   ├── model/             # Model architecture components
│   ├── train/             # Training utilities and data generation
│   └── visualization/     # Plotting and visualization tools
├── notebooks/             # Example notebooks and experiments
├── requirements.txt       # Package dependencies
├── Makefile              # Utility commands
└── README.md             # This file
```

## Citation

If you use ExplainerPFN in your research, please cite:

```bibtex
@inproceedings{
    TBD
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [TabPFN](https://github.com/PriorLabs/TabPFN) by Prior Labs
- Uses [SHAP](https://github.com/slundberg/shap) for ground truth explanations

