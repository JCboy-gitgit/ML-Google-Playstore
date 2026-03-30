# Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms a high-dimensional dataset into a lower-dimensional space while preserving as much variance as possible. It does this by identifying the directions (principal components) along which the data varies the most.

PCA was use to identify the landscape of the Google Play store apps. The first two principal components were able to capture a significant amount of the variance in the data, allowing us to visualize the distribution of apps in a 2D space. This visualization revealed clusters of apps based on their categories and ratings, providing insights into the relationships between different types of apps and their popularity.

## Initialization

Import the necessary libraries and load the dataset.

```python
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_pca = pd.read_csv('googleplaystore_cleaned.csv')
```

One condition for PCA is that the data must be continuous.

Log Transform the data
```python
df_pca['Log_Installs'] = np.log1p(df_pca['Installs'] + 1)
df_pca['Log_Reviews'] = np.log1p(df_pca['Reviews'] + 1)
df_pca['Log_Size'] = np.log1p(df_pca['Size'] + 1)
```

Define the contnuous features to be used in PCA and standardize the data.

```python
contnuous_features = [
    'Log_Installs',
    'Log_Reviews',
    'Log_Size',
    'Rating',
    'Price'
]

X = df_pca[contnuous_features]
```

Normalize/Standardize the data
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

`scaler` by default uses z score normalization, which centers the data to have a mean of 0 and scales it to have a standard deviation of 1. This is important for PCA because it is sensitive to the scales of the features. By standardizing the data, we ensure that each feature contributes equally to the analysis, preventing features with larger ranges from dominating the principal components.

## Modeling

First step is to instantiate the PCA model and fit it to the data.

```python
pca = PCA()
pca.fit(X_scaled)
```

### Variance Explained

To determine how many principal components to retain, we can look at the explained variance ratio.

```python
variance_ratio = pca.explained_variance_ratio_
```

The explained variance ratio tells us how much of the total variance in the data is captured by each principal component. We can plot the cumulative explained variance to visualize how many components we need to retain to capture a certain percentage of the variance.

```python
variance_ratio_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(variance_ratio))],
    'Variance Ratio': variance_ratio,
})
```

This will output the variance ratio for each principal component, which can be used to decide how many components to keep for further analysis. Typically, we look for a cumulative variance ratio of around 80-90% to retain enough information while reducing dimensionality.

#### Cumulative Variance

This is the cumulative variance explained by the principal components.

```python
cumulative_variance = sum(variance_ratio[:2]) * 100
```

For this dataset, the `cumulative_variance` results 67.7% cumulative_variance.

### PCA Components

The PCA components are the directions in the feature space that capture the most variance. We can access the PCA components using the `components_` attribute of the PCA model.

We will output the components in a data frame for better visualization.
```python
components_df = pd.DataFrame(
    pca.components_,
    columns=contnuous_features,
    index=[f'PC{i+1}' for i in range(len(variance_ratio))]
)
```

The components reveals the eigenvectors of the covariance matrix of the data. Each row corresponds to a principal component, and each column corresponds to a feature. The values in the components indicate the contribution of each feature to the corresponding principal component. By analyzing these values, we can understand which features are most influential in capturing the variance in the data along each principal component.

### Plotting the PCA

This will project the data onto the first two principal components and create a scatterplot to visualize the distribution of apps in the reduced dimensional space.

```python
pca_2d = pca.transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1 (Viral Reach)' : pca_2d[:, 0],
    'PC2 (Premium Penalty)' : pca_2d[:, 1],
    'App_type' : df_pca['Type']
})

plt.figure(figsize=(10, 6))
sns.catplot(
    x='PC1 (Viral Reach)',
    y='PC2 (Premium Penalty)',
    hue='App_type',
    data=pca_df,
    palette={'Free' : 'cyan', 'Paid' : 'red'},
    alpha=0.7,
    s=30
)

plt.title('PCA of Google Play Store Apps')
plt.xlabel('Principal Component 1 (Viral Reach)')
plt.ylabel('Principal Component 2 (Premium Penalty)')
plt.legend(title='App Type')
plt.show()
```



