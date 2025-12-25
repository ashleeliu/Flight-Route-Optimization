# Delay-Aware Itinerary Planning via Predict-Then-Optimize

For this group course project, we develop a predict-then-optimize framework for airline itinerary planning that explicitly accounts for flight delays and cancellation risk.

Using historical U.S. airline performance data, we first apply machine learning models to predict expected delay times and cancellation probabilities. These predictions are then embedded into an optimization model that selects flight itineraries minimizing expected delay, cancellation risk, and cost under user-specified constraints.

## Dataset

We use an airline dataset covering monthly aggregated flight statistics for U.S. airports and carriers from 2013–2023. Downloadable from [Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay?resource=download).

Key characteristics:
* Aggregated by (year, month, airport, carrier)
* Includes arrival delays, cancellations, diversions, and delay causes
* Focused on top 7 major U.S. carriers and major cross-country routes to reduce noise and sparsity

## Modeling Pipeline

### 1. Supervised Learning — Delay Prediction
- Target: Average arrival delay (minutes)
- Models:
  - Random Forest
  - XGBoost (log-transformed target)
- XGBoost achieved the best performance (R² ≈ 0.97)

### 2. Supervised Learning — Cancellation Risk
- Binary classification:
  - 0: ≤ 5% cancellations
  - 1: > 5% cancellations
- Random Forest classifier
- Threshold tuning used to improve recall for high-risk routes

### 3. Unsupervised Learning — Risk Profiling
- K-Means clustering on delay composition
- Identified distinct delay risk profiles (e.g., weather-dominated, carrier-dominated)
- PCA used for 2D and 3D visualization
- Clusters aligned well with true dominant delay causes (~79% accuracy)


## Optimization Model

Predicted delay and cancellation risk are integrated into a single-stage stochastic optimization model.

**Decision variables:**
- Select flight legs (carrier, origin, destination)

**Objective:**
- Minimize:
  - Expected delay
  - Cancellation risk (heavily penalized)
  - Delay-to-cost ratio

**Constraints:**
- Flow conservation (valid origin → destination path)
- Budget constraint
- At least one feasible itinerary returned

The model is implemented in Pyomo and supports interactive user input (origin, destination, budget).

## Tools & Technologies

- Python  
- Scikit-learn  
- XGBoost  
- Pyomo  
- HiGHS Solver  
- Matplotlib / Seaborn  
- LaTeX (for the final project report)


## Notes

This project was completed as part of a final project for INDENG 142A: Machine Learning & Data Analytics I at UC Berkeley.

Group members: Annie Chen, Ashlee Liu, Sukhman Sidhu, Kenny Wongchamcharoen, Charmaine Yuen
