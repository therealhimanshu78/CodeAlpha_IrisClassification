# 🌸 Iris Flower Classification using Logistic Regression

This project classifies Iris flower species (Setosa, Versicolor, Virginica) using Machine Learning with Logistic Regression.

---

## 📌 Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## 📊 Load Dataset

```python
iris = load_iris()
X = iris.data
y = iris.target
```

---

## 🧾 Convert to DataFrame

```python
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

print("First 5 Rows of Dataset:")
print(df.head())
```

---

## 📈 Data Visualization

```python
sns.pairplot(df, hue='species')
plt.show()
```

---

## ✂️ Train Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Model Training

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 📊 Model Evaluation

```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## ✅ Conclusion

Logistic Regression successfully classifies the Iris dataset with high accuracy.

---

## 🛠 Tools & Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
