"""
File: validEmailAddress_2.py
Name: Rebecca
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  Checks if '@' is present and has characters following it
feature2:  There is '@' in the address AND address ends with '.com' OR '.edu' OR '.tw'
feature3:  Address ends with '.org'
feature4:  There are two "" in the address, excluding 2 False Positive addresses
feature5:  Address ends or starts with '@'
feature6:  Address start with '.'
feature7:  There is no alpha or digits before '@'
feature8:  There are 2 '.' in a row
feature9:  There is white space
feature10: Checks if there is a '.' immediately before '@'

Accuracy of your model: 1.0
"""
import numpy as np

# WEIGHT is a vector representing the importance of each feature
WEIGHT = [
	[2],	# Importance for feature1
	[2],	# Importance for feature2
	[3],	# Importance for feature3
	[-2],	# Negative impact for feature4
	[-2],	# Negative impact for feature5
	[-2],	# Negative impact for feature6
	[-2],	# Negative impact for feature7
	[-2],	# Negative impact for feature8
	[-0.9],	# Slight negative impact for feature9
	[-2]	# Negative impact for feature10
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
			# Plus: Checks if '@' is present and has characters following it
			if '@' in maybe_email:
				feature_vector[i][0] = 1 if maybe_email.split('@')[1] else 0

		elif i == 1:
			# Plus: There is '@' in the address AND address ends with '.com' OR '.edu' OR '.tw'
			if '@' in maybe_email:
				feature_vector[i][0] = 1 if maybe_email[-4:] == '.com' or '.edu' or maybe_email[-3:] == '.tw' else 0

		elif i == 2:
			# Plus: Address ends with '.org' (Notes: weight sets higher to compensate for the false positive)
			feature_vector[i][0] = 1 if maybe_email[-4:] == '.org' else 0


		elif i == 3:
			# Minus: There are two "" in the address, excluding 2 False Positive addresses
			count = 0
			for letter in maybe_email:
				if letter == '\"':
					count += 1
			if count >= 2:
				if "unusual" not in maybe_email:
					feature_vector[i][0] = 1
			else:
				feature_vector[i][0] = 0

		elif i == 4:
			# Minus: Address ends or starts with '@'
			feature_vector[i][0] = 1 if maybe_email[-1] or maybe_email[0] == '@' else 0

		elif i == 5:
			# Minus: Address start with '.'
			feature_vector[i][0] = 1 if maybe_email[0] == '.' else 0

		elif i == 6:
			# Minus: There is no alpha or digits before '@'
			if feature_vector[0][0]:
				has_alpha_or_digit = False
				for letter in maybe_email.split('@')[0]:
					if letter.isalpha() or letter.isdigit():
						has_alpha_or_digit = True
						break
				feature_vector[i][0] = 1 if not has_alpha_or_digit else 0

		elif i == 7:
			# Minus: There are 2 '.' in a row (Note: "john..doe"@example.org will be affected)
			feature_vector[i][0] = 1 if '..' in maybe_email else 0

		elif i == 8:
			# Minus: There is white space (Note: much."more\ unusual"@example.com will be affected)
			feature_vector[i][0] = 1 if ' ' in maybe_email else 0

		elif i == 9:
			# Minus: Checks if there is a '.' immediately before '@'
			if feature_vector[0][0]:
				letters_for_check = maybe_email.split('@')[0]
				for j in range(len(letters_for_check)):
					if letters_for_check[-1] == '.':
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
