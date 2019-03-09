# Breast Cancer Classification

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Cancer data from the SKlearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# Create dataframe
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']]
                         , columns = np.append(cancer['feature_names'], ['target']))

# Visualizing the data
sns.pairplot(df_cancer, hue = 'target'
             , vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.countplot(df_cancer['target'], label = 'Count')
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
# Checking the variables' correlation
plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True)

# Creating our dataset
X = df_cancer.drop(['target'], axis = 1) # Dropping the target label columns
y = df_cancer['target']

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the SVM into the Training set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and creating Classification report
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying Grid Search to find the best model and the best parameters
from sklearn.grid_search import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           refit = True,
                           verbose = 4)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

sns.heatmap(cm, annot=True)


