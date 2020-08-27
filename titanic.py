import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from encode import *

from pandas import Series, DataFrame

filename = 'train.csv'
df = pd.read_csv(filename)
df.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

passengers = df
print(passengers.info)

for column in ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:
	print(f'Column "{column}" contains the following unique inputs: {passengers[column].unique()}')

print('\nTable of missing value counts in each column:')
print(passengers.isnull().sum())

# Sorting out missing data
passengers['Embarked'] = passengers['Embarked'].fillna(0)
passengers = passengers.drop(columns=['Cabin'])
passengers['Age'] = passengers['Age'].fillna(passengers['Age'].mean())

# Encoding
encode_pclass(passengers)
encode_embarked(passengers)
encode_sex(passengers)
encode_ticket(passengers)
passengers['Ticket'] = passengers['Ticket'].fillna(passengers['Ticket'].mean())
passengers = passengers.drop(columns=['Pclass', 'Embarked'])

dead, survived = passengers['Survived'].value_counts()
print(f"Number of Survived : {survived}")
print(f"Number of Dead : {dead}")
print(f"Percentage Dead : {round(dead/(dead+survived)*100,1)}%")


sns.countplot(x='Survived', data=passengers, palette = 'hls')
plt.title('Count Plot for Survived')
print(passengers.groupby('Survived').mean())
plt.show()

#Normalisation of remaining features
passengers_normalised = passengers.copy()
age_mean, age_std = normalise_col(passengers_normalised , 'Age')
sibsp_mean, sibsp_std = normalise_col(passengers_normalised , 'SibSp')
parch_mean, parch_std = normalise_col(passengers_normalised , 'Parch')
ticked_mean, ticket_std = normalise_col(passengers_normalised , 'Ticket')
fare_mean, fare_std = normalise_col(passengers_normalised , 'Fare')
print(passengers_normalised.groupby('Survived').mean())


def mean_visual(df):
	#Produce a bar chart showing the mean values of the explanatory variables grouped by match result. 
	fig = plt.figure(figsize=(16,12))
	df_copy = df.copy()
	df_copy = df_copy.drop(columns=['PassengerId'])
	df_copy_mean = df_copy.groupby('Survived').mean().reset_index()
	tidy = df_copy_mean.melt(id_vars='Survived').rename(columns=str.title)

	g =sns.barplot(x='Variable',y='Value' ,data=tidy, hue='Survived',palette = 'hls')
	fig.autofmt_xdate()
	plt.title("Visualisation of Mean Value by Survival Status")
	plt.xlabel("Explanatory Variable", fontsize=18)
	plt.ylabel("Normalised Value", fontsize=18)
	plt.tick_params(axis='both', which= 'major', labelsize=14)
	plt.show()

mean_visual(passengers_normalised)

def correlation_visualisation(df):
	"""
	Creates a correlation plot between each of the explanatory variables.
	"""

	plt.figure(figsize=(15,15))
	sns.heatmap(df.corr(), linewidth=0.25, annot=True, square=True, cmap = "BuGn_r", linecolor = 'w')


correlation_visualisation(passengers)
plt.title("Heatmap of Correlation for Variables")
plt.show()

passengers_final = passengers.drop(columns=['Name', 'PassengerId','class1', 'cherbourg', 'Ticket', 'queenstown', 'Fare', 'Parch']).copy()
passengers_final['bias'] = np.ones(passengers_final.shape[0])

import statsmodels.api as sm
from sklearn.model_selection import train_test_split

X=passengers_final.loc[:, passengers_final.columns != 'Survived']
y=passengers_final.loc[:, passengers_final.columns =='Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logit_model=sm.Logit(y_train.astype(int),X_train.astype(float))

result=logit_model.fit()
print(result.summary2())


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(fit_intercept=False)
logreg.fit(X_train.astype(float), np.ravel(y_train))

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print('\n', confusion_matrix)

print('\n',logreg.coef_)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test.values, logreg.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


filename = 'test.csv'
df = pd.read_csv(filename)
df.columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df['Embarked'] = df['Embarked'].fillna(0)
df = df.drop(columns=['Cabin'])
df['Age'] = df['Age'].fillna(passengers['Age'].mean())

encode_pclass(df)
encode_embarked(df)
encode_sex(df)
encode_ticket(df)
df['Ticket'] = df['Ticket'].fillna(df['Ticket'].mean())
df = df.drop(columns=['Pclass', 'Embarked'])

df_final = df.drop(columns=['Name', 'PassengerId','class1', 'cherbourg', 'Ticket', 'queenstown', 'Fare', 'Parch']).copy()
df_final['bias'] = np.ones(df_final.shape[0])

X=df_final.copy()
print('\nTable of missing value counts in each column:')
print(df_final.isnull().sum())

y_pred = logreg.predict(X)
df = pd.read_csv(filename)
df_predictions = pd.DataFrame(list(zip(df['PassengerId'],y_pred)),columns=['PassengerId','Survived']) 

df_predictions.to_csv(r'predictions.csv', index = False)