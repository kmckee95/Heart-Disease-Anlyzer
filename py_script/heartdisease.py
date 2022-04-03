# import the library
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#libraries for model evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

# read the dataset
df = pd.read_csv('heart.csv')

# get categorical columns
categorical_cols= df.select_dtypes(include=['object'])

# get count of unique values for categorical columns
for cols in categorical_cols.columns:
    print(cols,':', len(categorical_cols[cols].unique()),'labels')

# categorical columns
cat_col = categorical_cols.columns

# numerical column
num_col = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']

# define X and y
X = df.drop(['HeartDisease'],axis=1)
y = df['HeartDisease']

# create a pipeline for preprocessing the dataset

num_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('std_scaler', StandardScaler()),
    ])

num_attribs = num_col 
cat_attribs = cat_col

# apply transformation to the numerical and categorical columns
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

X = full_pipeline.fit_transform(X)

# save preprocessed data
temp_df = pd.DataFrame(X)
temp_df.to_csv('processed_data.csv')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# count plot for number of heart disease(1)/No heart disease(0)
import seaborn as sns
sns.countplot(y_train,palette='OrRd')

# create a fresh model based on tuned parameters
rfc1=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 50, max_depth=7, criterion='gini')

rfc1.fit(X_train, y_train)

# Predicting the Test set results
y_pred = rfc1.predict(X_test)
print('Random forest accuracy_score:',accuracy_score(y_test,y_pred))

# Save the Model

import pickle

# save the random forest model for future use
pickle.dump(rfc1, open('rfc.pickle', 'wb'))

# save the preprocessing pipeline
pickle.dump(full_pipeline, open('full_pipeline.pickle', 'wb'))

# Load the Models for future use

rfc_saved = pickle.load(open('rfc.pickle','rb'))
full_pipeline_saved = pickle.load(open('full_pipeline.pickle','rb'))

# Visualization

target = df['HeartDisease'].replace([0,1],['Low','High'])

data = pd.crosstab(index=df['Sex'],
           columns=target)

data.plot(kind='bar',stacked=True)
plt.show()

plt.figure(figsize=(10,5))
bins=[0,30,50,80]
sns.countplot(x=pd.cut(df.Age,bins=bins),hue=target,color='r')
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=target,hue=df.ChestPainType)
plt.xticks(np.arange(2), ['No', 'Yes']) 
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x=target,hue=df.ExerciseAngina)
plt.xticks(np.arange(2), ['No', 'Yes']) 
plt.show()

# feature importance

# get important features used by model 
importances = rfc1.feature_importances_
feature_names = num_col
for i in cat_col:
    feature_names = feature_names + [i]*df[i].nunique()

import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)

forest_importances = forest_importances.groupby(level=0).first().sort_values(ascending=False)

# plot the features based on their importance in model performance.
fig, ax = plt.subplots()
forest_importances.plot.bar()
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()