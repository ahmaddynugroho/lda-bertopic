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

def get_topics_bertopic(bertopic, all=False):
    topics = bertopic.get_topics().copy()
    result = [v for k, v in topics.items() if k != -1]
    if len(result) == 0:
        result = [v for k, v in topics.items()]
    result = [[w[0] for w in t] for t in result]
    result = [t for t in result if len(t) > 1]
    return result

def e_variant(): # TODO: change dataset and vairant as needed
    dataset = ['H', 'A', 'S']
    # dataset = ['H']
    # variant = ['T', 'C', 'L', 'W', 'N', 'CL', 'CW', 'CN', 'LW', 'LN', 'WN', 'CLW', 'LWN', 'WNC', 'NCL', 'CLWN']
    variant = ['WN', 'CLWN']
    return [f'{e}{v}' for e in dataset for v in variant]