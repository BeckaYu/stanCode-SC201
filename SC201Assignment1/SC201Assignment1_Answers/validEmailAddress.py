"""
File: validEmailAddress.py
Name: Rebecca
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email
address classifier. You will use a given
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: 0.6538461538461539
"""
import numpy as np

# WEIGHT is a vector representing the importance of each feature
WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	"""
	Calculates the accuracy rate of email validation
	based on defined features and their weights.
	:return: None
	"""
	# Reads email addresses into a list from the data file
	maybe_email_list = read_in_data()

	# Converts the weight vector list to a NumPy array and transposes it
	weight_vector = np.array(WEIGHT)
	weight_vector = weight_vector.T

	# Stores validation results to calculate the accuracy rate
	is_valid = []
	for maybe_email in maybe_email_list:
		# Generates the feature vector for each email address
		feature_vector = feature_extractor(maybe_email)

		# Computes the score by multiplying the weight vector with the feature vector
		score = weight_vector.dot(feature_vector)

		# Classifies the email as valid if the score is positive
		if score > 0:
			is_valid.append(True)
		else:
			is_valid.append(False)

	print(find_accuracy_rate(is_valid))


def find_accuracy_rate(validation_result):
	"""
	Calculates the accuracy rate based on validation results.
	:param validation_result: list, results of email validation
	:return: float, the calculated accuracy rate
	"""
	num_correct = 0

	# Counts the number of correct predictions for invalid emails
	for i in range(0, len(validation_result)//2):	# The first-half of data are incorrect
		if validation_result[i] is False:
			num_correct += 1

	# Counts the number of correct predictions for valid emails
	for i in range(len(validation_result)//2, len(validation_result)):	# The second-half of data are correct
		if validation_result[i] is True:
			num_correct += 1

	accuracy_rate = num_correct / len(validation_result)
	return accuracy_rate


def feature_extractor(maybe_email):
	"""
	Creates a feature vector for an email string based on predefined criteria
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	# Initializes the feature vector with zeros based on the number of weights
	feature_vector = [[0] for _ in range(len(WEIGHT))]

	# Evaluates each feature for the email address
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i][0] = 1 if '@' in maybe_email else 0
		elif i == 1:
			feature_vector[i][0] = 1 if '.' not in maybe_email.split('@')[0] else 0
		elif i == 2:
			if feature_vector[0][0]:
				words = maybe_email.split('@')[0]
				has_string = False
				if words:
					for word in words:
						if word.isalpha():
							has_string = True
							break
				feature_vector[i][0] = 1 if has_string else 0
		elif i == 3:
			if feature_vector[0][0]:
				feature_vector[i][0] = 1 if maybe_email.split('@')[1] else 0
		elif i == 4:
			if feature_vector[0][0]:
				feature_vector[i][0] = 1 if '.' in maybe_email.split('@')[-1] else 0
		elif i == 5:
			feature_vector[i][0] = 1 if ' ' in maybe_email else 0
		elif i == 6:
			feature_vector[i][0] = 1 if maybe_email[-4:] == '.com' else 0
		elif i == 7:
			feature_vector[i][0] = 1 if maybe_email[-4:] == '.edu' else 0
		elif i == 8:
			feature_vector[i][0] = 1 if maybe_email[-3:] == '.tw' else 0
		elif i == 9:
			feature_vector[i][0] = 1 if len(maybe_email) > 10 else 0

	# Converts the feature vector to a NumPy array for easier manipulation
	feature_vector = np.array(feature_vector)
	return feature_vector


def read_in_data():
	"""
	Reads a text file and extracts a list of strings that are potential email addresses
	:return: list, containing strings that might be valid email addresses
	"""
	maybe_email_list = []
	with open(DATA_FILE, 'r') as f:
		for line in f:
			maybe_email_list.append(line.strip())
	return maybe_email_list


if __name__ == '__main__':
	main()
