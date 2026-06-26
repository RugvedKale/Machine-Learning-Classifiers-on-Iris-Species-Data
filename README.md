# Iris Species Classification — SVM, KNN, and Random Forest

A comparative study of three classification algorithms — **SVM**, **KNN**, and **Random Forest** — applied to the classic Iris flower dataset. This was completed as **ML: TASK-03**, with Random Forest implemented as the additional algorithm beyond the required SVM and KNN.

## Overview

The Iris dataset contains 150 samples across three species (*Iris setosa*, *Iris versicolor*, *Iris virginica*), described by four features: sepal length, sepal width, petal length, and petal width. This project walks through loading and preprocessing the data, training each classifier, evaluating performance, and comparing the algorithms conceptually and empirically.

## Dataset

- **Source:** `Iris.csv` (150 rows, 6 columns — `Id`, 4 features, `Species`)
- **Classes:** 3 species, perfectly balanced at 50 samples each
- **Missing values:** None

## Workflow

1. **Preprocessing** — dropped the `Id` column, label-encoded the `Species` target, split the data 80/20 into train/test sets, and standardized features using `StandardScaler`.
2. **KNN** — trained with K=5.
3. **SVM** — trained and compared across four kernels: Linear, RBF, Sigmoid, and Polynomial.
4. **Random Forest** — trained with 100 decision trees; extracted feature importance scores.
5. **Evaluation** — accuracy scores and confusion matrices for every model, plus a bar chart comparing all six configurations.

## Results

| Model              | Test Accuracy |
|--------------------|:------------:|
| KNN (K=5)          | 100.0%       |
| SVM – Linear       | 96.67%       |
| SVM – RBF          | 100.0%       |
| SVM – Sigmoid      | 90.0%        |
| SVM – Polynomial   | 96.67%       |
| Random Forest      | 100.0%       |

**Feature importance (Random Forest):** Petal measurements dominate, together accounting for the vast majority of predictive power:

| Feature        | Importance |
|----------------|:----------:|
| PetalLengthCm  | 0.440      |
| PetalWidthCm   | 0.422      |
| SepalLengthCm  | 0.108      |
| SepalWidthCm   | 0.030      |

This confirms the well-known botanical fact that petal dimensions vary far more sharply across Iris species than sepal dimensions do.

## Key Takeaways

- **SVM (Linear/RBF) and Random Forest** matched each other at the top of the leaderboard, while the **Sigmoid kernel** clearly underperformed on this tabular dataset.
- **Feature scaling matters** for SVM and KNN (both are distance/margin-based), but is irrelevant for Random Forest, since tree splits rely on threshold comparisons rather than distances.
- **Random Forest's feature importance output** is a practical advantage SVM and KNN don't offer out of the box — useful for understanding *which* measurements actually drive predictions.
- On a small, clean, well-separated dataset like Iris, **algorithm choice matters less than preprocessing quality** — all three models perform comparably well once the data is properly prepared.

A full write-up of each algorithm's mechanics, assumptions, strengths/limitations, and when to use it is available in [`iris final report.pdf`](./iris%20final%20report.pdf).

## Repository Contents
├── Iris.csv                  # Dataset

├── iris final.ipynb          # Full notebook: preprocessing, training, evaluation, plots

├── iris final report.pdf     # Written report with conceptual analysis and comparisons

└── README.md

**## Tech Stack

- Python
- pandas, numpy
- scikit-learn (`train_test_split`, `StandardScaler`, `LabelEncoder`, `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`, confusion matrix utilities)
- matplotlib, seaborn

## Running the Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook "iris final.ipynb"
```

Make sure `Iris.csv` is in the same directory as the notebook before running.**
