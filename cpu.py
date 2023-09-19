import pandas as pd
from time import time
import os
from bertopic import BERTopic
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm.auto import tqdm
from utils import get_topics_bertopic, get_coherence, get_diversity, get_topics_lda
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import ast
from gensim.models.ldamulticore import LdaMulticore

stop_words = StopWordRemoverFactory().get_stop_words()


def get_docs(pathf, ast_parse=False):
    docs = pd.read_csv(pathf)
    docs = docs['article'].to_list()
    docs = docs[:300]
    if ast_parse:
        docs = [ast.literal_eval(d) for d in docs]
    return docs


def train_b(docs, **bargs):
    pipe = make_pipeline(
        TfidfVectorizer(),
        TruncatedSVD(100)
    )
    model = BERTopic(
        embedding_model=pipe,
        nr_topics='auto',
        **bargs
    )
    model.fit_transform(docs)
    return model


def get_bargs(v):
    bargs = {}
    if 'p' in v:
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=stop_words
        )
        bargs['vectorizer_model'] = vectorizer_model
    return bargs


def get_coherence_b(model, docs):
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    topics = get_topics_bertopic(model, all=True)
    c = get_coherence(
        topics=topics,
        texts=tokens,
        dictionary=dictionary
    )
    return c


def timed(f):
    start = time()
    result = f()
    return result, time() - start


def get_topic_freq_l(model, corpus):
    topic_distribution = model.get_document_topics(corpus)
    topic_freq = {}
    for tf in topic_distribution:
        for t, _ in tf:
            if t not in topic_freq:
                topic_freq[t] = 0
            topic_freq[t] = topic_freq[t] + 1
    return topic_freq


def get_top_topics_b(model):
    topics = model.get_topics()


vs = 'ar,ap,sr,sp'.split(',')

path = './datasets/ds/bertopic'
ms = {}
cs = {}
ds = {}
fs = {}  # frequency topics
t_t = {}
t_c = {}
t_d = {}
for v in (tv := tqdm(vs)):
    tv.set_description('bertopic')
    _path = f'{path}/{v}.csv'
    docs = get_docs(_path)
    ms[v], t_t[v] = timed(lambda: train_b(docs, **get_bargs(v)))
    cs[v], t_c[v] = timed(lambda: get_coherence_b(ms[v], docs))
    ds[v], t_d[v] = timed(lambda: get_diversity(
        get_topics_bertopic(ms[v], all=True)))

lpath = './datasets/ds/lda'
lms = {}
lcs = {}
lds = {}
lt_t = {}
lt_c = {}
lt_d = {}
for v in (tv := tqdm(vs)):
    tv.set_description('lda')
    _path = f'{lpath}/{v}.csv'
    docs = get_docs(_path, ast_parse=True)
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(d) for d in docs]
    num_topics = len(ms[v].get_topics())
    lms[v], lt_t[v] = timed(lambda: LdaMulticore(
        corpus,
        num_topics,
        id2word
    ))
    list(lms[v].get_document_topics(corpus))
    get_topic_freq_l(lms[v], corpus)
    break
    lcs[v], lt_c[v] = timed(lambda: get_coherence(
        model=lms[v],
        texts=docs,
        dictionary=id2word
    ))
    lds[v], lt_d[v] = timed(get_diversity(get_topics_lda(lms[v], id2word)))

# len(ms['ar'].get_topics())
ms['ap'].get_topic_freq()['Topic'][2]
ms['ap'].get_topic_freq()['Count']
ms['ap'].get_topic_freq(len(ms['ap'].get_topics()) - 1)
ms['ar'].get_topics()
ms['ar'].get_topic_freq().reset_index(drop=True)
ms['ap'].get_topics()

ms
cs
ds
t_t
t_c
t_d

lms
lcs
lds
lt_t
lt_c
lt_d
