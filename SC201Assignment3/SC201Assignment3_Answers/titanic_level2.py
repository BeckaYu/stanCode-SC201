"""
File: titanic_level2.py
Name: Rebecca
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math

import pandas
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

nan_cache = {}

def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None

	# Abandon features
	data.pop('PassengerId')
	data.pop('Name')
	data.pop('Ticket')
	data.pop('Cabin')

	# Covert data from str to int
	data['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
	data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)


	if mode == 'Train':
		# Remove rows with any missing data
		data = data.dropna()

		# Save the mean data of 'Age' and 'Embarked' to fill the missing test data
		nan_cache['Age'] = round(data.Age.mean(), 3)
		nan_cache['Embarked'] = round(data.Embarked.mean(), 3)
		nan_cache['Fare'] = round(data.Fare.mean(), 3)

		# Obtain the actual survival data
		labels = data.pop('Survived')

	else:	# mode == 'Test'
		data.Age.fillna(nan_cache['Age'], inplace=True)
		data.Embarked.fillna(nan_cache['Embarked'], inplace=True)
		data.Fare.fillna(nan_cache['Fare'], inplace=True)

	# Return different data in different modes
	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	############################
	unique_values = data[feature].unique()

	if feature == 'Sex' or feature == 'Embarked':
		# Create a new column for each unique value
		for value in unique_values:
			value = int(value)
			# Initialize the column with zeros
			data[f'{feature}_{value}'] = 0
			# Set the column to 1 where the feature equals the unique value
			data.loc[data[feature] == value, f'{feature}_{value}'] = 1

	if feature == 'Pclass':
		# Create a new column for each unique value
		for value in unique_values:
			value = int(value)
			# Pclass == n-> Pclass_n-1 == 1
			converted_value = value-1
			# Initialize the column with zeros
			data[f'{feature}_{converted_value}'] = 0
			# Set the column to 1 where the feature equals the unique value
			data.loc[data[feature]-1 == converted_value, f'{feature}_{converted_value}'] = 1

	# Remove original data
	data.pop(feature)
	############################
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	############################
	# Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')
	features = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
	extracted_features = data[features]
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(extracted_features)

	############################
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83848315
	TODO: real accuracy on degree3 -> 0.87640449
	"""
	# Data pre-processing
	training_data, labels = data_preprocess(TRAIN_FILE, mode='Train')

	# One hot encoding
	training_data = one_hot_encoding(training_data, 'Sex')
	training_data = one_hot_encoding(training_data, 'Pclass')
	training_data = one_hot_encoding(training_data, 'Embarked')

	# Standardization
	features = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_0', 'Sex_1', 'Pclass_0',
				'Pclass_1', 'Pclass_2', 'Embarked_0', 'Embarked_1']
	x_train = training_data[features]
	standardizer = preprocessing.StandardScaler()
	x_train= standardizer.fit_transform(x_train)

	# Create polynomial features
	poly_phi = preprocessing.PolynomialFeatures(degree=3)
	x_train = poly_phi.fit_transform(x_train)

	# Training
	h = linear_model.LogisticRegression(max_iter=10000)
	classifier = h.fit(x_train, labels)
	acc = round(classifier.score(x_train, labels), 8)
	print('Acc:', acc)


if __name__ == '__main__':
	main()
