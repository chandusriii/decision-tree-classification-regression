# 🌳 Decision Tree — Classification & Regression

A complete learning repository covering Decision Trees from fundamentals to practical implementation, following a concept → intuition → formula → example → implementation → interview revision flow.

## 📌 Learning Flow

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

# 1. 🌳 What is a Decision Tree?

A **Decision Tree** is a supervised machine learning algorithm that makes predictions by repeatedly splitting data based on feature conditions.

It works like a sequence of questions.

Example:

```
             Age <= 30?
              /       \
            Yes        No
            /           \
      Income <= 50K?     Buy
        /      \
      Yes       No
      /          \
    Don't Buy    Buy
```

Each decision divides the dataset into smaller groups.

### Decision Trees can be used for

- **Classification** → predicting a class
- **Regression** → predicting a numerical value

---

# 2. 🎯 Decision Tree Classification

In **classification**, the target variable represents a class or category.

Examples:

- Spam / Not Spam
- Yes / No
- Disease / No Disease
- Cat / Dog

The tree tries to create groups that are as **pure** as possible.

## 🌿 Pure vs Impure Nodes

### Pure Node

A node is pure when all observations belong to the same class.

```
Yes
Yes
Yes
Yes
```

### Impure Node

A node is impure when multiple classes are present.

```
Yes
No
Yes
No
```

The goal of splitting is to reduce impurity.

---

# 3. 📉 Entropy

**Entropy** measures the impurity or uncertainty in a dataset.

### Formula

```
Entropy = -Σ pᵢ log₂(pᵢ)
```

Where:

- (pᵢ) = probability of class (i)
- (n) = number of classes

### Intuition

```
Pure node
   ↓
Low uncertainty
   ↓
Low entropy

Mixed node
   ↓
High uncertainty
   ↓
High entropy
```

For a completely pure node:

```
Entropy = 0
```

So, **lower entropy means greater purity**.

---

# 4. 🔵 Gini Impurity

**Gini Impurity** is another measure used to determine how impure a node is.

### Formula

```
Gini = 1 - Σ pᵢ²
```

Where:

- (pᵢ) = probability of class (i)

For a completely pure node:

```
Class A = 100%
Class B = 0%

Gini = 0
```

A mixed node has higher impurity.

---

# 5. 📈 Information Gain

Information Gain tells us how much a split reduces uncertainty/entropy.

### Formula

```
Information Gain
=
Entropy(parent)
-
Weighted Entropy(children)
```

### Intuition

```
Before splitting
       ↓
   High impurity
       ↓
      Split
       ↓
After splitting
       ↓
 Lower impurity
```

A useful split produces a larger reduction in impurity.

### Classification Split Flow

```
Calculate impurity
        ↓
Try possible splits
        ↓
Calculate Information Gain
        ↓
Choose useful split
        ↓
Split dataset
        ↓
Repeat recursively
```

---

# 6. 🔀 How a Decision Tree Grows

The tree grows by repeatedly finding useful splits.

```
                    Root
                     │
              Best Feature?
              /           \
            Split         Split
            /                \
         Node                Node
        /    \              /    \
      ...    ...          ...    ...
```

The process continues until a stopping condition is reached.

Without controlling the growth, the tree can become unnecessarily complex.

---

# 7. ⚠️ Overfitting in Decision Trees

A Decision Tree can keep splitting until it becomes very deep.

A very deep tree may learn the training data too closely.

```
Simple Tree
     ↓
May underfit

Very Complex Tree
     ↓
May overfit
```

This is why **pruning** is important.

---

# 8. ✂️ Pruning

**Pruning** means removing unnecessary parts of a Decision Tree to reduce overfitting and improve performance on unseen data.

There are two main types:

```
                 PRUNING
                    │
          ┌─────────┴─────────┐
          │                   │
    Pre-Pruning          Post-Pruning
          │                   │
    Stop early            Grow tree
    while building        then remove
                           unnecessary
                           branches
```

---

# 9. 🛑 Pre-Pruning

In **Pre-Pruning**, we stop the tree from growing further while it is being built.

## `max_depth`

Controls the maximum depth of the tree.

```python
DecisionTreeClassifier(max_depth=3)
```

Smaller depth generally produces a simpler tree.

## `min_samples_split`

Minimum number of samples required to split an internal node.

```python
DecisionTreeClassifier(min_samples_split=10)
```

A node must have at least 10 samples before it can be split.

## `min_samples_leaf`

Minimum number of samples that must be present in a leaf.

```python
DecisionTreeClassifier(min_samples_leaf=5)
```

This prevents leaves from becoming too small.

## `max_leaf_nodes`

Controls the maximum number of leaf nodes and limits overall tree complexity.

## `min_impurity_decrease`

A split is performed only when the impurity decreases by at least the specified amount.

This prevents insignificant splits.

---

# 10. 🌿 Post-Pruning

In **Post-Pruning**, the tree is first allowed to grow and then unnecessary branches are removed.

## `ccp_alpha`

`ccp_alpha` is used for **Cost-Complexity Pruning**.

```python
DecisionTreeClassifier(ccp_alpha=0.01)
```

Generally:

```
Higher ccp_alpha
       ↓
More aggressive pruning
       ↓
Simpler tree
```

### Easy distinction

| Pre-Pruning | Post-Pruning |
|---|---|
| Stops tree early | Tree grows first |
| Controls growth | Removes unnecessary branches |
| `max_depth` | `ccp_alpha` |
| `min_samples_split` | Cost-Complexity Pruning |
| `min_samples_leaf` | Applied after growth |

---

# 11. 🧰 Important Pruning Parameters

| Parameter | Purpose |
|---|---|
| `max_depth` | Maximum tree depth |
| `min_samples_split` | Minimum samples required to split |
| `min_samples_leaf` | Minimum samples in a leaf |
| `max_leaf_nodes` | Maximum number of leaf nodes |
| `min_impurity_decrease` | Minimum impurity decrease required for splitting |
| `ccp_alpha` | Cost-Complexity Post-Pruning |

---

# 12. 💻 Decision Tree Classifier — Scikit-Learn

Basic implementation:

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Evaluation

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
```

You can also use:

```python
model.score(X_test, y_test)
```

For a `DecisionTreeClassifier`, `.score()` returns **accuracy**.

---

# 13. 🌳 Visualizing the Tree

Decision Trees are easy to visualize.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```

This helps understand:

- Which feature was selected
- How the tree is splitting
- How deep the tree became
- What decisions lead to each prediction

---

# 14. 📏 Do Decision Trees Need Feature Scaling?

Generally, **Decision Trees do not require feature scaling**.

A tree may make decisions such as:

```
Age <= 30?
Income <= 50000?
```

The algorithm is based on threshold-based splits rather than distance calculations.

This differs from algorithms such as:

- KNN
- Logistic Regression
- SVM
- Neural Networks

where feature scaling can be important.

---

# 15. 📊 Decision Tree Regression

A Decision Tree can also solve **regression problems**.

The difference is the type of prediction.

### Classification

```
Input
 ↓
Decision Tree
 ↓
Class
```

Example:

```
Spam
Not Spam
```

### Regression

```
Input
 ↓
Decision Tree
 ↓
Numerical value
```

Example:

```
House Price = ₹65,00,000
```

---

# 16. 📉 Variance in Decision Tree Regression

For regression, the key concept is **Variance**.

Variance measures how much the target values differ from their mean.

### Formula

```
Variance = (1/n) Σ(yᵢ - ȳ)²
```

Where:

- (yᵢ) = individual target value
- (ȳ) = mean target value
- (n) = number of observations

## Example

Suppose:

```
10, 20, 30
```

Mean:

```
ȳ = 20
```

Variance:

```
[(10-20)² + (20-20)² + (30-20)²] / 3
= 66.67
```

### Intuition

```
Values close to mean
        ↓
Lower variance

Values far from mean
        ↓
Higher variance
```

---

# 17. 📉 Variance Reduction

In Decision Tree Regression, the tree looks for splits that reduce variance.

### Formula

```
Variance Reduction
=
Variance(parent)
-
Weighted Variance(children)
```

### Process

```
Parent Node
    ↓
Calculate Parent Variance
    ↓
Try a Candidate Split
    ↓
Create Child Nodes
    ↓
Calculate Child Variances
    ↓
Calculate Weighted Variance
    ↓
Calculate Variance Reduction
    ↓
Choose the split with the largest reduction
```

The goal is to create groups where target values are more similar to each other.

---

# 18. 🔢 Variance Reduction Example

Consider:

```
X     Y
1     2
2     4
3     6
4     10
```

Suppose we consider:

```
X <= 2
```

The split produces:

```
Left Child:
Y = 2, 4

Right Child:
Y = 6, 10
```

We calculate:

1. Parent variance
2. Left-child variance
3. Right-child variance
4. Weighted child variance
5. Variance reduction

The formula remains:

```
VR = Variance(parent)
     - Weighted Variance(children)
```

---

# 19. 🔍 Comparing Candidate Splits

A Decision Tree Regression problem can have multiple possible splits.

For example:

```
Split 1 → X <= 2
Split 2 → X <= 3
```

For each candidate:

```
Calculate child groups
        ↓
Calculate child variances
        ↓
Calculate weighted variance
        ↓
Calculate variance reduction
```

The split with the **largest variance reduction** is selected.

---

# 20. 💻 Decision Tree Regressor — Scikit-Learn

Basic implementation:

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Mean Squared Error

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print(mse)
```

### R² Score

```python
score = model.score(X_test, y_test)

print(score)
```

For a `DecisionTreeRegressor`, `.score()` returns **R²**.

---

# 21. ⚖️ Classifier vs Regressor

| Feature | Decision Tree Classifier | Decision Tree Regressor |
|---|---|---|
| Purpose | Classification | Regression |
| Prediction | Class | Numerical value |
| Example | Spam / Not Spam | House Price |
| Main concept | Impurity | Variance |
| Split concepts | Entropy / Gini | Variance |
| Split selection | Information Gain | Variance Reduction |
| `.score()` | Accuracy | R² |

---

# 22. 🧠 Complete Decision Tree Mental Model

```
                    DECISION TREE
                          │
             ┌────────────┴────────────┐
             │                         │
       CLASSIFICATION              REGRESSION
             │                         │
       Predict a Class           Predict a Number
             │                         │
      ┌──────┴──────┐               Variance
      │             │                   │
   Entropy         Gini          Variance Reduction
      │             │                   │
      └──────┬──────┘                   │
             │                           │
      Information Gain                   │
             │                           │
             └──────────┬────────────────┘
                        │
                      SPLIT
                        │
                  Tree Growth
                        │
                  Overfitting?
                        │
                     PRUNING
                   ┌────┴────┐
                   │         │
              Pre-Pruning Post-Pruning
                   │         │
             max_depth     ccp_alpha
             min_samples
             max_leaf_nodes
                   │
                   ↓
             Simpler Tree
```

---

# 23. 🔑 The 7 Core Questions

After completing this topic, you should be able to explain:

### 1. What is a Decision Tree?

A supervised learning algorithm that makes predictions through a sequence of feature-based decisions.

### 2. What is the difference between Classifier and Regressor?

```
Classifier → Predicts a class
Regressor  → Predicts a number
```

### 3. What is Entropy?

A measure of uncertainty/impurity in classification.

### 4. What is Gini Impurity?

A measure of impurity used to evaluate classification nodes.

### 5. What is Information Gain?

The reduction in entropy obtained from a split.

### 6. What is Pruning and why do we need it?

Pruning removes or prevents unnecessary tree growth to reduce overfitting.

### 7. What is Variance Reduction?

The reduction in target variance produced by a regression split.

---

# 24. 🛠️ Practical Skills Covered

The practical Scikit-Learn workflow is:

```
Import Model
     ↓
Prepare Data
     ↓
Train / Test Split
     ↓
Create Model
     ↓
model.fit()
     ↓
model.predict()
     ↓
Evaluate Model
     ↓
Visualize Tree
     ↓
Control Overfitting
     ↓
Pruning
```

### Classification

```python
DecisionTreeClassifier
```

### Regression

```python
DecisionTreeRegressor
```

### Important methods

```python
.fit()
.predict()
.score()
```

### Important controls

```python
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
min_impurity_decrease
ccp_alpha
```

---

# 🎯 Final Revision

```
Decision Tree
│
├── Classification
│   ├── Pure / Impure Nodes
│   ├── Entropy
│   ├── Gini Impurity
│   └── Information Gain
│
├── Tree Growth
│   └── Can lead to Overfitting
│
├── Pruning
│   ├── Pre-Pruning
│   │   ├── max_depth
│   │   ├── min_samples_split
│   │   ├── min_samples_leaf
│   │   ├── max_leaf_nodes
│   │   └── min_impurity_decrease
│   │
│   └── Post-Pruning
│       └── ccp_alpha
│
└── Regression
    ├── Continuous Output
    ├── Variance
    ├── Variance Reduction
    └── DecisionTreeRegressor
```

---

## 📂 Repository Structure

```
decision-tree-classification-regression/
│
├── README.md
├── decision_tree_classifier.ipynb
├── decision_tree_regressor.ipynb
├── LICENSE
└── .gitignore
```

---

## 🚀 What This Repository Covers

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
- Classification vs Regression
- Interview revision

---

## 📌 Key Takeaway

```
DECISION TREE
      │
      ├── Classification
      │      ├── Entropy
      │      ├── Gini
      │      └── Information Gain
      │
      ├── Regression
      │      ├── Variance
      │      └── Variance Reduction
      │
      ├── Tree Growth
      │      └── Overfitting
      │
      └── Pruning
             ├── Pre-Pruning
             └── Post-Pruning
                    └── ccp_alpha
```

The goal is not only to use a Decision Tree, but to understand **why it makes a split, how it grows, how overfitting happens, and how pruning controls the tree**.
