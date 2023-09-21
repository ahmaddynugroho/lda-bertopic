import pickle
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
import pandas as pd

def get_coherence(topics=None, model=None, texts=None, dictionary=None):
    c = CoherenceModel(model=model, topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    cs = c.get_coherence()
    return cs

def get_diversity(topics):
    # https://github.com/MIND-Lab/OCTIS/blob/02fff36346b72818990b9d4d636a7bfb13aa048b/octis/evaluation_metrics/diversity_metrics.py#L12
    topk = 10
    unique_words = set()
    for topic in topics:
        unique_words = unique_words.union(set(topic[:topk]))
    td = len(unique_words) / (topk * len(topics))
    return td

def get_topics_lda(lda, dictionary):
    k = lda.num_topics
    topics = []
    for i in range(k):
        terms = lda.get_topic_terms(i)
        ki = [dictionary[t[0]] for t in terms]
        topics.append(ki)
    return topics

def get_topics_bertopic(m, topics, dictionary):
    topic_words = []
    for topic in range(len(set(topics)) - m._outliers):
        words = list(zip(*m.get_topic(topic)))[0]
        words = [word for word in words if word in dictionary.token2id]
        topic_words.append(words)
    topic_words = [words for words in topic_words if len(words) > 0]
    return topic_words

def e_variant(): # TODO: change dataset and vairant as needed
    dataset = ['H', 'A', 'S']
    # dataset = ['H']
    # variant = ['T', 'C', 'L', 'W', 'N', 'CL', 'CW', 'CN', 'LW', 'LN', 'WN', 'CLW', 'LWN', 'WNC', 'NCL', 'CLWN']
    variant = ['WN', 'CLWN']
    return [f'{e}{v}' for e in dataset for v in variant]