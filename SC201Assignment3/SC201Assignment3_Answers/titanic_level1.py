"""
File: titanic_level1.py
Name: Rebecca
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyperparameter tuning to find the best model.
"""

from collections import defaultdict
from util import *
import math
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	############################
	is_header = True
	data = defaultdict(list)

	with open (filename, 'r') as f:
		for line in f:
			if mode == 'Train':
				# Ignore header
				if is_header:
					header = line.strip().split(',')
					# pop the header that will not be used, have to do it reversely to avoid index error
					header.pop(10)
					header.pop(8)
					header.pop(3)
					header.pop(0)
					is_header = False

				else:
					information = line.strip().split(',')
					information.pop(11)
					information.pop(9)
					information.pop(4)
					information.pop(3)
					information.pop(0)
					if '' not in information:		# Ignore incomplete data
						for i in range(len(header)):
							# Survived
							if i == 0:
								data[header[i]].append(int(information[i]))
							# Pclass
							if i == 1:
								data[header[i]].append(int(information[i]))
							# Sex
							if i == 2:
								if information[i] == 'male':
									data[header[i]].append(int(1))
								elif information[i] == 'female':
									data[header[i]].append(int(0))
							# Age
							if i == 3:
								data[header[i]].append(float(information[i]))
							# SibSp
							if i == 4:
								data[header[i]].append(int(information[i]))
							# Parch
							if i == 5:
								data[header[i]].append(int(information[i]))
							# Fare
							if i == 6:
								data[header[i]].append(float(information[i]))
							# Embarked
							if i == 7:
								if information[i] == 'S':
									data[header[i]].append(int(0))
								elif information[i] == 'C':
									data[header[i]].append(int(1))
								elif information[i] == 'Q':
									data[header[i]].append(int(2))
			else:
				# Ignore header
				if is_header:
					header = line.strip().split(',')
					# pop the header that will not be used, have to do it reversely to avoid index error
					header.pop(9)
					header.pop(7)
					header.pop(2)
					header.pop(0)
					print(header)
					is_header = False
				else:
					information = line.strip().split(',')
					information.pop(10)
					information.pop(8)
					information.pop(3)
					information.pop(2)
					information.pop(0)

					for i in range(len(header)):
						# Pclass
						if i == 0:
							data[header[i]].append(int(information[i]))
						# Sex
						if i == 1:
							if information[i] == 'male':
								data[header[i]].append(int(1))
							elif information[i] == 'female':
								data[header[i]].append(int(0))
						# Age
						if i == 2:
							mean_age = round((sum(training_data['Age']) / len(training_data['Age'])), 3)
							if information[i] == '':
								data[header[i]].append(mean_age)
							else:
								data[header[i]].append(float(information[i]))
						# SibSp
						if i == 3:
							data[header[i]].append(int(information[i]))
						# Parch
						if i == 4:
							data[header[i]].append(int(information[i]))
						# Fare
						if i == 5:
							mean_fare = round((sum(training_data['Fare']) / len(training_data['Fare'])), 3)
							if information[i] == '':
								data[header[i]].append(mean_fare)
							else:
								data[header[i]].append(float(information[i]))
						# Embarked
						if i == 6:
							if information[i] == 'S':
								data[header[i]].append(int(0))
							elif information[i] == 'C':
								data[header[i]].append(int(1))
							elif information[i] == 'Q':
								data[header[i]].append(int(2))
		return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	############################
	if feature == 'Pclass':
		data['Pclass_0'] = []
		data['Pclass_1'] = []
		data['Pclass_2'] = []

		for pclass_data in data['Pclass']:
			data['Pclass_0'].append(1) if pclass_data == 1 else data['Pclass_0'].append(0)
			data['Pclass_1'].append(1) if pclass_data == 2 else data['Pclass_1'].append(0)
			data['Pclass_2'].append(1) if pclass_data == 3 else data['Pclass_2'].append(0)


	elif feature == 'Sex' or 'Embarked':
		# Identify unique values in the feature
		unique_values = set(data[feature])

		for unique_value in unique_values:
			# Create new feature name for one-hot encoding
			new_feature_name = f'{feature}_{unique_value}'

			# Initialize new dict, defaulted to 0
			data[new_feature_name] = [0] * len(data[feature])

			# Create new dict
			for i, feature_value in enumerate(data[feature]):
				if feature_value == unique_value:
					data[new_feature_name][i] = 1

	# Remove the original feature column
	del data[feature]
	return data
	############################


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	############################
	for feature in data:
		min_data = min(data[feature])
		max_data = max(data[feature])
		for i, feature_value in enumerate(data[feature]):
			data[feature][i] = (feature_value-min_data) / (max_data-min_data)
	############################
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0

	# Step 2 : Start training
	for epoch in range(num_epochs):
		for i in range(len(labels)):		# Total numbers of passengers
			# Step 3 : Feature Extract
			individual_data = {}

			if degree == 1:
				for key in keys:
					# Create a new dict to store each passenger's data
					individual_data[key] = inputs[key][i]

			elif degree == 2:
				# Include original features
				for key in keys:
					individual_data[key] = inputs[key][i]
				# Interaction terms, note: also avoid redundant pairs
				for j in range(len(keys)):
					for k in range(j, len(keys)):
						individual_data[keys[j] + keys[k]] = inputs[keys[j]][i] * inputs[keys[k]][i]

			# Calculate the score of each feature
			score = dotProduct(individual_data, weights)

			# Apply the sigmoid function to the score for logistic regression probability
			h = 1 / (1 + math.exp(-score))

			# # Step 4 : Update weights: perform gradient descent update
			increment(weights, -alpha*(h-labels[i]), individual_data)

	return weights

