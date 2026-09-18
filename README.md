# 🌳 Decision Tree — Classification & Regression

A complete learning repository covering Decision Trees from fundamentals to practical implementation.

## 📚 Repository Structure

```
decision-tree-classification-regression/
│
├── README.md
│
├── classification/
│   └── decision_tree_classifier.ipynb
│
├── regression/
│   └── decision_tree_regressor.ipynb
│
├── assignments/
│   └── shop-smart/
│       ├── README.md
│       ├── shop_smart.ipynb
│       └── shop_smart_ecommerce.csv
│
├── LICENSE
└── .gitignore
```

---

# 🧭 Learning Path

```
                    DECISION TREE
                         │
             ┌───────────┴───────────┐
             │                       │
      CLASSIFICATION             REGRESSION
             │                       │
      Predict a Class          Predict a Number
             │                       │
      ┌──────┴──────┐             Variance
      │             │                 │
   Entropy         Gini        Variance Reduction
      │             │                 │
      └──────┬──────┘                 │
             │                         │
      Information Gain                │
             │                         │
             └──────────┬──────────────┘
                        │
                     SPLITTING
                        │
                   Tree Growth
                        │
                    Overfitting?
                        │
                     PRUNING
                    ┌────┴────┐
                    │         │
               Pre-Pruning  Post-Pruning
                              │
                          ccp_alpha
```

---

# 🎯 What This Repository Covers

- Decision Tree fundamentals
- Classification
- Entropy
- Gini Impurity
- Information Gain
- Tree growth
- Overfitting
- Pre-Pruning
- Post-Pruning
- Cost-Complexity Pruning
- Decision Tree Regression
- Variance
- Variance Reduction
- Scikit-Learn implementation
- Tree visualization
- Practical assignment
- Hyperparameter tuning

---

# 📂 Sections

## 1. 🎯 Classification

The classification notebook demonstrates Decision Tree Classification using the Titanic dataset.

**File:**

```
classification/decision_tree_classifier.ipynb
```

The notebook covers:

- Loading and exploring the dataset
- Handling missing values
- Encoding categorical features
- Selecting features and target
- Train-test split
- Decision Tree Classifier
- Model evaluation
- Tree visualization

---

## 2. 📊 Regression

The regression notebook demonstrates Decision Tree Regression using the Diabetes dataset.

**File:**

```
regression/decision_tree_regressor.ipynb
```

The notebook covers:

- Loading the Diabetes dataset
- Separating features and target
- Train-test split
- Decision Tree Regressor
- Prediction
- Regression evaluation
- Tree visualization
- Regression model interpretation

---

## 3. 🧪 Practical Assignment — ShopSmart

The ShopSmart assignment applies Decision Tree Classification to an e-commerce purchase prediction problem.

**Folder:**

```
assignments/shop-smart/
```

Contents:

```
assignments/shop-smart/
├── README.md
├── shop_smart.ipynb
└── shop_smart_ecommerce.csv
```

### Assignment focus

- Exploratory Data Analysis
- Feature preprocessing
- Numerical and categorical features
- Decision Tree Classification
- Imbalanced target handling
- F1 Score
- Tree complexity control
- Hyperparameter tuning
- Cross-validation

The assignment uses **12,330 individual user sessions** and predicts the `Revenue` target.

The assignment specifies an **F1 benchmark of 0.55**.

The notebook reports:

- Held-out test F1: **0.6278**
- Best 5-fold cross-validation F1: **0.6344**
- Best parameters: `max_depth=4`, `min_samples_leaf=50`

See the assignment README for the complete experiment details.

---

# 🌱 Why the Repository is Organized This Way

The structure separates the learning material into three clear stages:

```
THEORY
  ↓
CLASSIFICATION / REGRESSION
  ↓
PRACTICAL ASSIGNMENT
```

This makes it easy to move from:

**understanding the concept → implementing the algorithm → applying it to a practical problem**

---

# 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# 🧠 Core Mental Model

```
Decision Tree
│
├── Classification
│   ├── Entropy
│   ├── Gini Impurity
│   └── Information Gain
│
├── Tree Growth
│   └── Overfitting
│
├── Pruning
│   ├── Pre-Pruning
│   └── Post-Pruning
│       └── ccp_alpha
│
└── Regression
    ├── Variance
    ├── Variance Reduction
    └── DecisionTreeRegressor
```

---

# 🔑 Quick Revision

### Classification

```
Class prediction
     ↓
Entropy / Gini
     ↓
Information Gain
     ↓
Best Split
     ↓
Tree Growth
```

### Regression

```
Numerical prediction
     ↓
Variance
     ↓
Variance Reduction
     ↓
Best Split
     ↓
Tree Growth
```

### Overfitting Control

```
Tree Growth
     ↓
Complexity
     ↓
Overfitting?
     ↓
Pruning / Constraints
```

---

# 🚀 Repository Goal

This repository is designed as a **structured Decision Tree learning record** rather than a collection of disconnected notebooks.

Each section connects the concepts with practical Scikit-Learn implementation and an applied assignment.

---

## 👤 Author

**Chandu Sri**

Exploring Machine Learning, AI, and practical model development with Python.
