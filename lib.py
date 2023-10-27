import ast
from time import time

import gensim.corpora as corpora
import pandas as pd
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def get_coherence(topics=None, model=None, texts=None, dictionary=None):
    c = CoherenceModel(model=model,
                       topics=topics,
                       texts=texts,
                       dictionary=dictionary,
                       coherence='c_v')
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


# %% get docs as list
def get_docs(pathf, ast_parse=False):
    docs = pd.read_csv(pathf)
    docs.columns = ["doc"]
    docs = docs["doc"].to_list()
    # docs = docs[:100]
    if ast_parse:
        docs = [ast.literal_eval(d) for d in docs]
    return docs


# %% train bertopic
def train_b(docs, **bargs):
    sentence_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device='cpu')
    model = BERTopic(embedding_model=sentence_model, nr_topics='auto', **bargs)
    topics, prob = model.fit_transform(docs)
    return (model, topics)


# %% get bertopic training args
def get_bargs(v):
    bargs = {}
    if "wg" in v:
        stop_words = StopWordRemoverFactory().get_stop_words()
        vectorizer_model = CountVectorizer(ngram_range=(1, 3),
                                           stop_words=stop_words)
        bargs["vectorizer_model"] = vectorizer_model
    return bargs


# %% get bertopic coherence
def get_coherence_b(m, topics, docs):
    vectorizer = m.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    topic_words = get_topics_bertopic(m, topics, dictionary)
    k = len(topic_words)
    c = get_coherence(topics=topic_words, texts=tokens, dictionary=dictionary)
    return (c, k, topic_words)


# %% used to get training time and basically any lambda fn
def timed(f):
    start = time()
    result = f()
    return result, time() - start


# %% get top 5 most used topics in bertopic
def get_top_7_b(model):
    r = []
    topics = model.get_topic_freq().head(7)["Topic"]
    for t in topics:
        tw = model.get_topic(t)
        r.append([w for w, _ in tw])
    return r


# %% get top 5 most used topics in lda
def get_top_7_l(m, corpus):
    td = {"Count": {}}
    for bow in corpus:
        _tds = sorted(m[bow], key=lambda x: x[1], reverse=True)
        ti = _tds[0][0]
        td["Count"][ti] = td["Count"][ti] + 1 if ti in td["Count"] else 1
    td = pd.DataFrame.from_dict(td).sort_values(by="Count",
                                                ascending=False).head(7)
    _tw = []
    for ti in td.index:
        ti
        _tw.append([w for w, p in m.show_topic(ti)])
    return _tw
