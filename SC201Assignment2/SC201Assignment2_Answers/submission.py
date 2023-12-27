#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction
def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    feature_vector = defaultdict(int)
    for token in x.split():
        feature_vector[token] += 1
    return feature_vector


############################################################
# Milestone 4: Sentiment Classification
def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # the weight vector

    # Preprocess all features and labels for training
    all_features = [(featureExtractor(review), label) for review, label in trainExamples]

    for epoch in range(numEpochs):
        # Update weights using gradient descent for each example
        for features, label in all_features:
            # Calculate the dot product of weights and features; defaults to 0 if weights are not initialized
            score = dotProduct(weights, features)

            # Apply the sigmoid function to the score for logistic regression probability
            h = 1 / (1 + math.exp(-score))

            # Perform gradient descent update; adjust label for logistic regression: -1 becomes 0, 1 remains 1
            increment(weights, -alpha * (h- (1 if label == 1 else 0)), features)

        # Define a predictor and evaluate errors
        def predictor(x):
            features_final = featureExtractor(x)
            return 1 if dotProduct(features_final, weights) >= 0 else -1

        training_error = evaluatePredictor(trainExamples, predictor)
        validation_error = evaluatePredictor(validationExamples, predictor)
        print(f"Training error: ({epoch} epoch): {training_error}")
        print(f"Validation error: ({epoch} epoch): {validation_error}")

    return weights


############################################################
# Milestone 5a: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrence.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        phi = {word: random.randint(1, len(weights)) for word, weight in weights.items()}
        y = 1 if dotProduct(weights, phi) >=0 else -1
        return phi, y
    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features
def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """
    def extract(x: str) -> Dict[str, int]:
        # Initialize a default dictionary for feature vectors
        feature_vector_ch = defaultdict(int)

        # Concatenate words from x without spaces
        words = "".join(x.split())

        # Generate n-gram counts from the concatenated string
        for i in range(len(words)-n+1):
            feature_vector_ch[words[i:i+n]] += 1
        return feature_vector_ch
    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

