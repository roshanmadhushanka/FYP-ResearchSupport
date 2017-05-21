from bs4 import BeautifulSoup
import math
from nltk import TweetTokenizer


def KLDivergence(P, Q):
    # D_KL (PQ) = SUM( P(i) * log (P(i) / Q(i)))
    """
    Measure the KL Divergence of two sentences
    :param P: Sentence 1 [str]
    :param Q: Sentence 2 [str]
    :return: KL Divergence value
    """
    if not isinstance(P, str) or not isinstance(Q, str):
        # Invalid
        return

    # Tokenize sentences
    tknzr = TweetTokenizer()
    words_P = tknzr.tokenize(P)
    words_Q = tknzr.tokenize(Q)

    # Calculate probabilities
    keys_P = list(set(words_P))
    prob_P = {}
    size_P = len(words_P)
    for key in keys_P:
        count = words_P.count(key)
        prob_P[key] = float(count) / size_P

    keys_Q = list(set(words_Q))
    prob_Q = {}
    size_Q = len(words_Q)
    for key in keys_Q:
        count = words_Q.count(key)
        prob_Q[key] = float(count) / size_Q

    # Calculate KL divergence value
    kl_divergence = 0.0
    for key in keys_Q:
        if key in keys_P:
            kl_divergence += prob_P[key] * math.log(prob_P[key]/prob_Q[key])
    return kl_divergence


def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
    if len(s1) > len(s2):
        s2 = s2.ljust(len(s1))
    else:
        s1 = s1.ljust(len(s2))

    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


v1 = open('version1.html', mode='r', encoding='utf-8')
soup_1 = BeautifulSoup(v1.read(), 'lxml')
p_list_1 = soup_1.findAll('p')
a_list_1 = soup_1.findAll('a')

v2 = open('version3.html', mode='r', encoding='utf-8')
soup_2 = BeautifulSoup(v2.read(), 'lxml')
p_list_2 = soup_2.findAll('p')
a_list_2 = soup_2.findAll('a')

y = "මීතොටමුල්ල කුණු කන්ද කඩා වැටීමේ ඛේදවාචකයට වගකිව යුතු සියලු තරාතිරම්වල දේශපාලනඥයන් සහ නිලධාරීන්ට"

mini = 1000
t = ""
for p2 in p_list_2:
    x = p2.text.strip()

    val = hamming_distance(x, y)
    if val < mini:
        mini = val
        t = x

print(t)
print(mini)


