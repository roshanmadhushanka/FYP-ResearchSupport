from bs4 import BeautifulSoup
from urllib.request import urlopen
from bs4.element import Tag
from nltk.tokenize import TweetTokenizer
import math
import nltk


def lessFrequent(sentence, nwords):
    if not isinstance(sentence, str) or not isinstance(nwords, int):
        # Invalid
        return

    # Tokenize sentences
    tknzr = TweetTokenizer()
    words = tknzr.tokenize(sentence)
    total_words = len(words)
    keys = list(set(words))
    prob = {}
    for word in keys:
        prob[word] = float(words.count(word)) / total_words

    prob = sorted(prob.items(), key=lambda x: x[1])
    prob = [x[0] for x in prob]

    if len(prob) > nwords:
        return prob[:nwords]
    else:
        return prob


def middleFrequent(sentence, nwords):
    if not isinstance(sentence, str) or not isinstance(nwords, int):
        # Invalid
        return

    # Tokenize sentences
    tknzr = TweetTokenizer()
    words = tknzr.tokenize(sentence)
    total_words = len(words)
    keys = list(set(words))
    prob = {}
    for word in keys:
        prob[word] = float(words.count(word)) / total_words

    prob = sorted(prob.items(), key=lambda x: x[1])
    prob = [x[0] for x in prob]

    middle = len(keys) / 2
    offset = nwords / 2

    if offset > middle:
        return prob
    else:
        n1 = int(middle-offset+0.5)
        n2 = int(middle+offset+0.5)
        return prob[n1:n2]


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


def traverse(soup, path=[], maps={}):
    if soup.name is not None:
        maps[tuple(path)] = soup.text
        if isinstance(soup, Tag):
            count = 0
            for child in soup.children:
                count += 1
                temp = list(path)
                temp.append(str(child.name) + str(count))
                traverse(child, temp, maps)

# # Divaina
# soup = BeautifulSoup(open("C:\\Users\\Roshan\\PycharmProjects\\FYP-Server\\Divaina.html", encoding="utf8").read(), "lxml")
# tag_dictionary1 = {}
# traverse(soup, maps=tag_dictionary1)
#
# # Lankadeepa
# soup = BeautifulSoup(open("C:\\Users\\Roshan\\PycharmProjects\\FYP-Server\\Lankadeepa.html", encoding="utf8").read(), "lxml")
# tag_dictionary2 = {}
# traverse(soup, maps=tag_dictionary2)

text1 = 'Machine learning is the subfield of computer science that, according to Arthur Samuel in 1959, gives "computers' \
        ' the ability to learn without being explicitly programmed."[1] Evolved from the study of pattern recognition ' \
        'and computational learning theory in artificial intelligence,[2] machine learning explores the study and ' \
        'construction of algorithms that can learn from and make predictions on data[3] – such algorithms overcome ' \
        'following strictly static program instructions by making data-driven predictions or decisions,[4]:2 through ' \
        'building a model from sample inputs. Machine learning is employed in a range of computing tasks where designing ' \
        'and programming explicit algorithms with good performance is difficult or unfeasible; example applications ' \
        'include email filtering, detection of network intruders or malicious insiders working towards a data breach,[5]' \
        ' optical character recognition (OCR),[6] learning to rank and computer vision. Machine learning is closely ' \
        'related to (and often overlaps with) computational statistics, which also focuses on prediction-making through ' \
        'the use of computers. It has strong ties to mathematical optimization, which delivers methods, theory and ' \
        'application domains to the field. Machine learning is sometimes conflated with data mining,[7] where the latter ' \
        'subfield focuses more on exploratory data analysis and is known as unsupervised learning.[4]:vii[8] Machine ' \
        'learning can also be unsupervised[9] and be used to learn and establish baseline behavioral profiles for ' \
        'various entities[10] and then used to find meaningful anomalies. Within the field of data analytics, machine ' \
        'learning is a method used to devise complex models and algorithms that lend themselves to prediction; in ' \
        'commercial use, this is known as predictive analytics. These analytical models allow researchers, data ' \
        'scientists, engineers, and analysts to "produce reliable, repeatable decisions and results" and uncover ' \
        '"hidden insights" through learning from historical relationships and trends in the data.[11] As of 2016, ' \
        'machine learning is a buzzword, and according to the Gartner hype cycle of 2016, at its peak of inflated ' \
        'expectations.[12] Because finding patterns is hard, often not enough training data is available, and also ' \
        'because of the high expectations it often fails to deliver.[13][14]'

text2 = 'The process of machine learning is similar to that of data mining. Both systems search through data to look ' \
        'for patterns. However, instead of extracting data for human comprehension -- as is the case in data mining ' \
        'applications -- machine learning uses that data to detect patterns in data and adjust program actions ' \
        'accordingly.  Machine learning algorithms are often categorized as being supervised or unsupervized. ' \
        'Supervised algorithms can apply what has been learned in the past to new data. Unsupervised algorithms can draw' \
        ' inferences from datasets. Facebook\'s News Feed uses machine learning to personalize each member\'s feed. If a' \
        ' member frequently stops scrolling in order to read or "like" a particular friend\'s posts, the News Feed will ' \
        'start to show more of that friend\'s activity earlier in the feed. Behind the scenes, the software is simply ' \
        'using statistical analysis and predictive analytics to identify patterns in the user\'s data and use to ' \
        'patterns to populate the News Feed. Should the member no longer stop to read, like or comment on the friend\'s ' \
        'posts, that new data will be included in the data set and the News Feed will adjust accordingly.'

text3 = 'Love is a variety of different feelings, states, and attitudes that ranges from interpersonal affection ("I ' \
        'love my mother") to pleasure ("I loved that meal"). It can refer to an emotion of a strong attraction and ' \
        'personal attachment.[1] Love can also be a virtue representing human kindness, compassion, and affection—"the ' \
        'unselfish loyal and benevolent concern for the good of another".[2] It may also describe compassionate and ' \
        'affectionate actions towards other humans, one\'s self or animals.[3] Ancient Greek philosophers identified ' \
        'four forms of love: kinship or familiarity (in Greek, storge), friendship (philia), sexual or romantic desire ' \
        '(eros), and self-emptying or divine love (agape). Modern authors have distinguished further varieties of love: ' \
        'limerence, amour de soi, and courtly love. Non-Western traditions have also distinguished variants or symbioses' \
        ' of these states.[4][5] Love has additional religious or spiritual meaning—notably in Abrahamic religions. ' \
        'This diversity of uses and meanings combined with the complexity of the feelings involved makes love unusually ' \
        'difficult to consistently define, compared to other emotional states. Love in its various forms acts as a major' \
        ' facilitator of interpersonal relationships and, owing to its central psychological importance, is one of the ' \
        'most common themes in the creative arts.[6] Love may be understood as a function to keep human beings together ' \
        'against menaces and to facilitate the continuation of the species.[7]'

# print(KLDivergence(text2, text1))
print(nltk.pos_tag(nltk.word_tokenize(text1)))