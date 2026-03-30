# Random Forest Classifier

[data cleaning](data-cleaning.md) 

One of the models we use for this project is the Random Forest Classifier. It is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## Initialization

Importing the necessary libraries and initializing the Random Forest Classifier:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
```

Other libraries such as `pandas` and `numpy` may also be imported for data manipulation and analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
Loading the dataset
```python
df = pd.read_csv('dataset/cleaned_data.csv')
```

### Formatting the Data

The cleaned data, although free of missing values, may still require formatting to be suitable for the Random Forest Classifier. The Size column still need formatting.

This will handle `'Varies with device'` and convert the Size column to numeric values.
```python
df['Size_veries'] = (df['Size'] == 'Varies with device').astype(int)
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
```

This way, we create a new column `Size_veries` to indicate whether the size varies with the device, and we convert the `Size` column to numeric values, handling any non-numeric entries gracefully.

Filling any remaining missing values in the Size column with the median value based on app category:
```python
df['Size'] = df.groupby('Category')['Size'].transform(
    lambda x: x.fillna(x.median())
)
```

What this does is it groups the data by the `Category` column and fills any missing values in the `Size` column with the median size for that specific category. This approach helps to maintain the integrity of the data while ensuring that we have a complete dataset for training our Random Forest Classifier.

```python
df['Size'].fillna(df['Size'].median(), inplace=True)
```

This is for fallback in case there are still any missing values in the `Size` column after the group-wise filling. It fills any remaining missing values with the overall median size from the entire dataset.


## The Model

Before we start, let's define the objective.

> [!IMPORTANT]
> Can we predict if an app will be a hit or a miss based on its features?

### Initialization

Let's define the Target variable
```python
df['Success'] = (df['Installs'] > 100_000).astype(int)
```

Then the features
```python
features = [
    'Category',
    'Szize_veries',
    'Size',
    'Type',
    'Price',
    'Content Rating',
]
```

Then set the variables
```python
X = df[features]
y = df['Success']
```

Encoding categorical variables
```python
X_encoded = pd.get_dummies(X, drop_first=True, columns=[
    'Category',
    'Type',
    'Content Rating',
])
```

This will convert the categorical variables into a format that can be provided to the Random Forest Classifier. The `drop_first=True` parameter is used to avoid multicollinearity by dropping the first category of each variable.

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=67)
```

### Model Training

```python
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=67, n_jobs=-1)
```

This instantiates the Random Forest Classifier with 100 trees (`n_estimators=100`), a fixed random state for reproducibility, and `n_jobs=-1` to utilize all available CPU cores for training.

```python
rf_classifier.fit(X_train, y_train)
```

This line trains the Random Forest Classifier on the training data (`X_train` and `y_train`).

### Prediction

```python
y_pred = rf_classifier.predict(X_test)
```

This line uses the trained model to make predictions on the test set (`X_test`), and the predicted labels are stored in `y_pred`. Sklearn includes `predict` method for easy prediction after the model has been trained.

### Evaluation

```python
print(classification_report(y_test, y_pred, terget_names=['Miss', 'Hit']))
```

This will print a classification report that includes precision, recall, f1-score, and support for each class (Miss and Hit). This report helps to evaluate the performance of the Random Forest Classifier on the test set.

#### Report

This will show the precision, recall, f1-score, and support.

- Precision: The ratio of correctly predicted positive observations to the total predicted positives. It indicates how many of the predicted hits were actually hits.

- Recall: The ratio of correctly predicted positive observations to the all observations in actual class. It indicates how many of the actual hits were correctly predicted.

- F1-Score: The weighted average of Precision and Recall. It is a better measure than accuracy for imbalanced datasets.

- Accuracy: The ratio of correctly predicted observations to the total observations. It is not always the best metric, especially for imbalanced datasets.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Where $TP$ is True Positives and $FP$ is False Positives.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Where $TP$ is True Positives and $FN$ is False Negatives.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Where Precision and Recall are as defined above.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where $TP$ is True Positives, $TN$ is True Negatives, $FP$ is False Positives, and $FN$ is False Negatives.

#### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

This will output the confusion matrix, which is a table that is often used to describe the performance of a classification model. It shows the counts of true positives, true negatives, false positives, and false negatives.

How to read:

|               | Predicted Miss | Predicted Hit |
|---------------|----------------|---------------|
| Actual Miss   | True Negatives ($TN$) | False Positives ($FP$) |
| Actual Hit    | False Negatives ($FN$) | True Positives ($TP$) |


### Importance of Features

```python
importances = rf_classifier.feature_importances_
feature_names = X_encoded.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
```

This will output a DataFrame that lists the features and their corresponding importance scores, sorted in descending order of importance. This helps to identify which features are most influential in predicting whether an app will be a hit or a miss.

For this project, these are the important features:
- Size: 0.59
- Size_varies: 0.10
- Type_Paid: 0.04
- Price: 0.03
- Category_GAME: 0.03



