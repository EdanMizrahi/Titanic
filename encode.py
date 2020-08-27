import numpy as np
import pandas as pd
import re

def encode_pclass(df):
	# A function used to create three columns, encoding the pclass column
	class1, class2, class3= [], [], []
	for passenger in df['Pclass']:
		if passenger == 1:
			class1.append(1)
			class2.append(0)
			class3.append(0)
		elif passenger == 2:
			class1.append(0)
			class2.append(1)
			class3.append(0)
		elif passenger == 3:
			class1.append(0)
			class2.append(0)
			class3.append(1)
		else: 
			class1.append(0)
			class2.append(0)
			class3.append(0)
	df['class1'] = class1
	df['class2'] = class2
	df['class3'] = class3
	print(f'pclass variable encoding complete. Columns "class1", "class2" and "class3" added to DataFrame')
	print(f'IMPORANT: ensure you drop the original pclass column.\n')


def encode_embarked(df):
	# A function used to create three columns, encoding the Embarked column
	cherbourg, queenstown, southampton = [], [], []
	for passenger in df['Embarked']:
		if passenger == 'C':
			cherbourg.append(1)
			queenstown.append(0)
			southampton.append(0)
		elif passenger == 'Q':
			cherbourg.append(0)
			queenstown.append(1)
			southampton.append(0)		
		elif passenger == 'S':
			cherbourg.append(0)
			queenstown.append(0)
			southampton.append(1)
		else:
			cherbourg.append(0)
			queenstown.append(0)
			southampton.append(0)
	df['cherbourg'] = cherbourg
	df['queenstown'] = queenstown
	df['southampton'] = southampton
	print(f'Embarked variable encoding complete. Columns "cherbourg", "queenstown" and "southampton" added to DataFrame')
	print(f'IMPORANT: ensure you drop the original Embarked column.\n')

def encode_sex(df):
	# A function used to replace the sex column in a df, with a binary value. 
	sex = []
	for passenger in df['Sex']:
		if passenger == 'male':
			sex.append(1)
		else:
			sex.append(-1)
	df['Sex']= sex
	print(f'Sex variable encoding complete in DataFrame.\n')


def encode_ticket(df):
	# A function used to remove the letters from the ticket column.
	df['Ticket'] = df['Ticket'].str.extract('(\d+)(?!.*\d)')
	df['Ticket'] = df['Ticket'].astype(float)
	print(f'Non-numeric characters successfully removed from ticket column.\n')


def normalise_col(df, col):
	# Function to normalise the data in each column.
	mean, std  = df[col].mean(), df[col].std()
	normalised_col = (df[col] - df[col].mean())/(df[col].std())
	df[col] = normalised_col
	print(f'Column normalised in DataFrame.')
	return mean, std

