"""Pre-registered Phase 0 candidate manifest (UCI survey + legacy additions)."""

# (uci_id, name, bucket_group) from the surveyed no-missing UCI datasets.
# UCI n/d/class counts are filled in by the OpenML verification step.
UCI_CANDIDATES = [
    # [3, 7)
    "Apartment for Rent Classified", "Maternal Health Risk", "Vertebral Column",
    "Wholesale customers", "Contraceptive Method Choice", "Thyroid Disease",
    "User Knowledge Modeling", "Banknote Authentication",
    "Blood Transfusion Service Center",
    "Assessing Mathematics Learning in Higher Education", "Haberman's Survival",
    "Land Mines",
    # [7, 11)
    "Forty Soybean Cultivars from Subsequent Harvests",
    "ILPD (Indian Liver Patient Dataset)", "MAGIC Gamma Telescope",
    "Page Blocks Classification", "Statlog (Heart)", "Glass Identification",
    "Statlog (German Credit Data)", "Website Phishing", "ApisTox",
    "Energy Efficiency", "HTRU2", "Yeast", "Abalone", "Auction Verification",
    "Ecoli", "NHANES Age Prediction Subset", "Raisin",
    "Rice (Cammeo and Osmancik)",
    # [11, 15)
    "Algerian Forest Fires", "Iranian Churn",
    "Risk Factor Prediction of Chronic Kidney Disease", "Thoracic Surgery Data",
    "Drug Consumption (Quantified)", "Estimation of Obesity Levels",
    "Heart Failure Clinical Records", "Wine Quality",
]

# Explicitly simulated: excluded from the pool.
SIMULATED_EXCLUSIONS = {
    "MONK's Problems": "simulated",
    "Electrical Grid Stability Simulated Data": "simulated",
}

# Legacy datasets not among the UCI candidates (eligible under n >= 200).
LEGACY_ADDITIONS = ["Annealing", "Heart Disease"]

# Names the UCI API does not resolve by exact name: pinned by UCI id instead.
UCI_ID_OVERRIDES = {
    "Land Mines": 763,
    "NHANES Age Prediction Subset": 887,
    "Estimation of Obesity Levels": 544,
}

# UCI page backing each legacy dataset that is not itself a UCI candidate.
LEGACY_UCI_NAME = {"ANNEALING": "Annealing", "HEART DISEASE": "Heart Disease"}

assert len(UCI_CANDIDATES) == 38
