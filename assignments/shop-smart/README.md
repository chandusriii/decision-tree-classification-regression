# 🛒 ShopSmart — E-Commerce Purchase Prediction

A practical **Decision Tree Classification** assignment based on the ShopSmart e-commerce session dataset.

This assignment applies the Decision Tree concepts studied in this repository to a real-world-style binary classification problem: predicting whether an online visitor completes a purchase.

---

## 🎯 Problem Statement

ShopSmart wants to predict whether a visitor is likely to make a purchase based on their browsing-session behaviour.

The assignment requires:

- Exploratory Data Analysis (EDA)
- Feature preprocessing and transformations
- Decision Tree based classification
- Handling an imbalanced target
- Evaluation using **F1 Score**
- Pruning / tree-complexity control

The assignment provides **0.55 F1 score as the benchmark**.

---

## 📊 Dataset

The dataset contains **12,330 individual user sessions** and **18 columns**.

Each row represents a visitor session.

### Target

**`Revenue`**

- `False` → visitor did not complete a purchase
- `True` → visitor completed a purchase

The uploaded dataset contains:

- **10,422** non-purchase sessions
- **1,908** purchase sessions

This makes the target **imbalanced**, which is why F1 score is used as the primary evaluation metric.

---

## 🧾 Features

| Feature | Description |
|---|---|
| `Administrative` | Number of account-management pages visited |
| `Administrative_Duration` | Time spent on administrative pages |
| `Informational` | Number of informational pages visited |
| `Informational_Duration` | Time spent on informational pages |
| `ProductRelated` | Number of product-related pages visited |
| `ProductRelated_Duration` | Time spent on product-related pages |
| `BounceRates` | Percentage of visitors leaving after one page |
| `ExitRates` | Percentage of exits from a page |
| `PageValues` | Average value of pages visited before a transaction |
| `SpecialDay` | Closeness to a special day |
| `Month` | Month of the visit |
| `OperatingSystems` | Operating system used |
| `Browser` | Browser used |
| `Region` | Geographic region |
| `TrafficType` | Source of website traffic |
| `VisitorType` | Visitor category |
| `Weekend` | Whether the visit occurred on a weekend |
| `Revenue` | **Target — whether a purchase was completed** |

---

## 🔄 Machine Learning Workflow

```
ShopSmart Dataset
       ↓
Separate Features & Target
       ↓
Identify Numerical & Categorical Features
       ↓
Train / Test Split
       ↓
Preprocessing Pipeline
       ├── Numerical → StandardScaler
       └── Categorical → OneHotEncoder
       ↓
Decision Tree Classifier
       ↓
Handle Class Imbalance
       ↓
F1 Score Evaluation
       ↓
Hyperparameter Tuning
       ↓
Compare Model Performance
```

---

## 🧹 Preprocessing

The notebook separates the target:

```python
X = df.drop(columns=["Revenue"])
y = df["Revenue"].astype(int)
```

### Numerical Features

Numerical columns are processed using:

```python
StandardScaler()
```

### Categorical Features

Categorical columns are encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing is combined with the classifier using a Scikit-Learn **Pipeline** and **ColumnTransformer**.

---

## 🌳 Decision Tree Classifier

The initial model uses:

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)
```

### Why these parameters?

| Parameter | Role |
|---|---|
| `max_depth=6` | Controls tree depth and limits excessive growth |
| `min_samples_leaf=30` | Keeps leaves from becoming too small |
| `class_weight="balanced"` | Addresses the imbalanced target |
| `random_state=42` | Makes the experiment reproducible |

The `max_depth` and `min_samples_leaf` settings also act as tree-complexity controls related to the pruning concepts covered in the main Decision Tree notes.

---

## ⚖️ Why F1 Score?

The target is imbalanced:

```
Non-Revenue → 10,422
Revenue     →  1,908
```

Accuracy alone can hide poor performance on the minority class.

The assignment therefore focuses on **F1 Score**, which combines:

- Precision
- Recall

```
F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

---

## 📈 Initial Model Results

On the held-out test set, the notebook reports:

**F1 Score: 0.6278**

Classification report:

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| 0 | 0.97 | 0.85 | 0.90 |
| 1 | 0.50 | 0.83 | 0.63 |

Confusion Matrix:

```
[[1771  313]
 [  64  318]]
```

The positive class (`Revenue=True`) has an F1 score of approximately **0.63** on the test set.

---

## 🔧 Hyperparameter Tuning

The notebook then uses **GridSearchCV** with F1 scoring.

Parameters explored:

```python
{
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [20, 30, 50]
}
```

The search uses:

- 5-fold cross-validation
- F1 as the scoring metric
- Multiple tree-depth and leaf-size combinations

### Best Cross-Validation Result

```
Best F1: 0.6344
```

Best parameters found:

```
max_depth = 4
min_samples_leaf = 50
```

> **Note:** 0.6344 is the best cross-validation F1 reported by GridSearchCV. The notebook does not subsequently report the tuned model's F1 on the held-out test set, so these two values should not be treated as the same evaluation.

---

## ✂️ Connection to Pruning

The assignment specifically asks for pruning to improve Decision Tree performance.

The notebook applies **tree-complexity control** through:

- `max_depth`
- `min_samples_leaf`

These restrict how complex the tree can become.

The main repository notes additionally cover:

### Pre-Pruning

```
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
min_impurity_decrease
```

### Post-Pruning

```
ccp_alpha
```

This assignment therefore provides a practical application of the pruning concepts studied in the main Decision Tree notes.

---

## 💻 Implementation

The complete implementation is available in:

**`shop_smart.ipynb`**

The notebook contains:

1. Dataset loading
2. Feature / target separation
3. Numerical and categorical feature detection
4. Train-test split
5. Preprocessing pipeline
6. Decision Tree Classifier
7. Class-imbalance handling
8. F1 score evaluation
9. Classification report
10. Confusion matrix
11. Hyperparameter tuning with GridSearchCV

---

## 📁 Assignment Files

```
shop-smart/
│
├── README.md
├── shop_smart.ipynb
└── shop_smart_ecommerce.csv
```

---

## 🧠 What This Assignment Demonstrates

This practical assignment connects the theory of Decision Trees with an applied classification problem.

### Concepts applied

- Supervised Learning
- Binary Classification
- Numerical & Categorical Features
- Data Preprocessing
- One-Hot Encoding
- Feature Scaling
- Decision Tree Classification
- Class Imbalance
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Tree Complexity Control
- Hyperparameter Tuning
- Cross-Validation
- GridSearchCV
- Reproducible ML Pipeline

---

## 🔗 Relation to the Main Decision Tree Notes

```
Decision Tree
      │
      ├── Classification
      │      ├── Entropy
      │      ├── Gini
      │      └── Information Gain
      │
      ├── Tree Growth
      │      └── Overfitting
      │
      ├── Pruning
      │      ├── Pre-Pruning
      │      └── Post-Pruning
      │
      └── ShopSmart Assignment
             │
             ├── Classification
             ├── Imbalanced Data
             ├── F1 Score
             ├── Tree Complexity Control
             └── Hyperparameter Tuning
```

---

## 🎯 Assignment Benchmark

The assignment specifies:

**Benchmark F1 Score: 0.55**

The notebook's initial held-out test F1 is:

**0.6278**

The GridSearchCV best cross-validation F1 is:

**0.6344**

These results are reported separately because they represent different evaluation procedures.

---

## 🚀 Key Takeaway

The ShopSmart assignment shows how a Decision Tree moves from **theory to an applied ML workflow**:

```
Problem
  ↓
Data
  ↓
Preprocessing
  ↓
Decision Tree
  ↓
Handle Imbalance
  ↓
F1 Evaluation
  ↓
Control Tree Complexity
  ↓
Hyperparameter Tuning
  ↓
Model Analysis
```

The main learning is not only building a Decision Tree, but understanding how **data preprocessing, class imbalance, evaluation metrics, and tree complexity** affect a practical classification problem.
