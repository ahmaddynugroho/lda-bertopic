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

def get_topics_bertopic(bertopic):
    topics = bertopic.get_topics().copy()
    topics.pop(-1)
    result = topics.values()
    result = [[w[0] for w in t] for t in result]
    return result

def e_variant(): # TODO: change dataset and vairant as needed
    # dataset = ['H', 'A', 'S']
    dataset = ['H']
    # e_variant = ['T', 'C', 'CL', 'CLW', 'CW', 'L', 'LW', 'W']
    variant = ['CLW', 'LWN']
    return [f'{e}{v}' for e in dataset for v in variant]

def load_ldas():
    r = {} # result
    for v in e_variant():
        with open(f'./datasets/small/models_lda/{v}.pickle', 'rb') as f:
            r[v] = pickle.load(f)
    return r

def load_bertopics():
    path_prefix = './datasets/small/models_bertopic/'
    r = {}
    for v in e_variant():
        r[v] = {}
        r[v]['model'] = BERTopic.load(f'{path_prefix}{v}')
        r[v]['time'] = pd.read_csv(f'{path_prefix}{v}/time.csv')['time'][0]
    return r