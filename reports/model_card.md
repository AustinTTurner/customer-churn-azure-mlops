# Customer Churn Model Card

## Model

- Name: Balanced Logistic Regression
- Version: 1.0.0
- Selected hyperparameter: C = 0.1
- Decision threshold: 0.55
- Positive class: Churn (1)

## Intended Use

Demonstrate customer churn classification and a reproducible machine-learning deployment workflow.

The model is a portfolio prototype and is not approved for production customer decisions.

## Training Data

IBM Telco Customer Churn dataset, retrieved from OpenML dataset 42178.

The original dataset contains 7,043 records.
An 80/20 stratified split with random_state = 42 produced 5,634 training records and 1,409 test records.

## Development-Stage Evaluation

Threshold selection used five-fold, training-only out-of-fold predictions.

At the selected threshold of 0.55:

| Metric | Value |
|---|---:|
| Precision | 0.5456 |
| Recall | 0.7639 |
| F1 | 0.6366 |
| True positives | 1,142 |
| False positives | 951 |
| False negatives | 353 |

The illustrative business requirement was to achieve at least 75% recall while maximizing
precision among eligible thresholds.

## Evaluation Limitations

The original test set was examined during earlier model comparison and
is not an untouched final evaluation set.

Hyperparameters and the decision threshold were selected using the same
training dataset. The reported metrics are development-stage estimates,
not independent final performance.

Additional independent evaluation is REQUIRED.

## Prediction Limitations

The model uses class weighting, and its output scores have not
been separately calibrated.

Scores should not be presented as verified real-world probabilities of 
customer churn.

The model has not been validated on a different telecommunications
provider or a later time period.

## Deployment 

The versioned local package contains the fitted scikit-learn
pipeline and a JSON manifest.

The manifest records the decision threshold, required input
features, model version and development-stage metrics.