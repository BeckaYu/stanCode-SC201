"""
File: interactive.py
Name: Rebecca
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
from util import interactivePrompt
import submission

# File containing the trained weight vector
FINAL_WEIGHTS = 'weights'

def main():
	# Load trained weights into a dictionary
	with open(FINAL_WEIGHTS, 'r') as f:
		weights = {line.split()[0]: float(line.split()[1]) for line in f}

	# Feature extraction function
	feature_extractor = submission.extractWordFeatures

	# Start interactive prompt for review sentiment prediction
	interactivePrompt(feature_extractor, weights)


if __name__ == '__main__':
	main()
