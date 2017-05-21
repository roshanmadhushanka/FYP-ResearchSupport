from nltk import TweetTokenizer
from numpy import dot
from numpy.linalg import norm


def cosineSimilarity(text_a, text_b):
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




# Load text from files
file_a = open('version4.html', 'r')
file_b = open('version5.html', 'r')
text_a = file_a.read()
text_b = file_b.read()
file_a.close()
file_b.close()

val = cosineSimilarity(text_a, text_b)
print(val)


