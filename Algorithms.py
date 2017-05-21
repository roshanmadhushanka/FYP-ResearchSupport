from nltk import TweetTokenizer
from numpy import dot
from numpy.linalg import norm
from pyjarowinkler import distance
import distance

def hammingDistance(text_a, text_b):
    """Return the Hamming distance between equal-length sequences"""

    if len(text_a) != len(text_b):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(text_a, text_b))


def cosineSimilarity(text_a, text_b):
    """
    Calulate cosine similarity between two texts
    :param text_a: Text a
    :param text_b: Text b
    :return: Cosine similarity value
    """

    # Tokenize sentences
    tknzr = TweetTokenizer()
    word_list_a = tknzr.tokenize(text_a)
    word_list_b = tknzr.tokenize(text_b)

    keys = list(set(word_list_a + word_list_b))
    vector_size = len(keys)
    vector_a = [0] * vector_size
    vector_b = [0] * vector_size

    for i in range(vector_size):
        vector_a[i] = word_list_a.count(keys[i])
        vector_b[i] = word_list_b.count(keys[i])

    return dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b))


def jaroWinklerDistance(text_a, text_b):
    """
    Calculate Jaro Winkler Distance
    :param text_a: Text a
    :param text_b: Text b
    :return: Jaro Winkler distance value
    """
    return distance.get_jaro_distance(text_a, text_b, winkler=True, scaling=0.1)


def levenshteinDistance(text_a, text_b):
    """
    Calculate Levenshtein distance
    :param text_a: Text a
    :param text_b: Text b
    :return: Levnshtein distance value
    """
    return distance.levenshtein(text_a, text_b)


