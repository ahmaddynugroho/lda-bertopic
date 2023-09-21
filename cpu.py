# %% imports
import ast
import json
from statistics import mean
from time import time

import gensim.corpora as corpora
import pandas as pd
import xarray as xr
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm

from utils import get_coherence, get_diversity, get_topics_bertopic, get_topics_lda

stop_words = StopWordRemoverFactory().get_stop_words()


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
    pipe = make_pipeline(TfidfVectorizer(), TruncatedSVD(100))
    model = BERTopic(embedding_model=pipe, nr_topics="auto", **bargs)
    topics, prob = model.fit_transform(docs)
    return (model, topics)


# %% get bertopic training args
def get_bargs(v):
    bargs = {}
    if "wg" in v:
        vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words)
        bargs["vectorizer_model"] = vectorizer_model
    return bargs


# %% get bertopic coherence
def get_coherence_b(model, topics, docs):
    vectorizer = m.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    topic_words = []
    for topic in range(len(set(topics)) - m._outliers):
        words = list(zip(*m.get_topic(topic)))[0]
        words = [word for word in words if word in dictionary.token2id]
        topic_words.append(words)
    topic_words = [words for words in topic_words if len(words) > 0]
    c = get_coherence(topics=topic_words, texts=tokens, dictionary=dictionary)
    return c


# %% used to get training time and basically any lambda fn
def timed(f):
    start = time()
    result = f()
    return result, time() - start


# %% get top 5 most used topics in bertopic
def get_top_5_b(model):
    r = []
    topics = model.get_topic_freq().head(5)["Topic"]
    for t in topics:
        tw = model.get_topic(t)
        r.append([w for w, _ in tw])
    return r


# %% get top 5 most used topics in lda
def get_top_5_l(model, corpus, docs, id2word):
    sorted_topics = m.top_topics(corpus, docs, id2word, coherence="c_v", topn=10)
    return [[w for wp, w in t] for t, p in sorted_topics[:5]]


# %% variants
vs = "ar,awg,alwg,sr,swg,slwg".split(",")

# %% bertopic training
path = "./datasets/dsn"
ds_runs = {}
ds = {}
for v in (tv := tqdm(vs, position=0)):
    bv = f"b_{v}"
    tv.set_description(f"bertopic {bv}")
    _path = f"{path}/{bv}.csv"
    docs = get_docs(_path)
    r = {"c": [], "d": [], "tt": [], "tc": [], "td": [], "t": [], "s": [], "tw": []}
    for i in tqdm(range(5), desc="runs", position=1, leave=False):
        mt, tt = timed(lambda: train_b(docs, **get_bargs(bv)))
        m, topics = mt
        c, tc = timed(lambda: get_coherence_b(m, topics, docs))
        d, td = timed(lambda: get_diversity(get_topics_bertopic(m, all=True)))
        t = tt + tc + td
        s = c * d
        tw = get_top_5_b(m)
        r["c"].append(c)
        r["d"].append(d)
        r["t"].append(t)
        r["s"].append(s)
        r["tt"].append(tt)
        r["tc"].append(tc)
        r["td"].append(td)
        r["tw"].append(tw)
        if all(s >= x for x in r["s"]):
            twt = json.dumps(get_top_5_b(m))
    ds_runs[bv] = {k: json.dumps(v) for (k, v) in r.items()}
    r = {k: mean(v) for (k, v) in r.items() if k != "tw"}
    r["twt"] = twt
    ds[bv] = r


# %% lda training
for v in (tv := tqdm(vs, position=0)):
    lv = f"l_{v}"
    tv.set_description(f"lda {v}")
    _path = f"{path}/{lv}.csv"
    docs = get_docs(_path, ast_parse=True)
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(d) for d in docs]
    num_topics = len(json.loads(ds[f"b_{v}"]["twt"]))
    r = {
        "c": [],
        "d": [],
        "tt": [],
        "tc": [],
        "td": [],
        "t": [],
        "s": [],
        "tw": [],
    }
    for i in tqdm(range(5), desc="runs", position=1, leave=False):
        m, tt = timed(lambda: LdaMulticore(corpus, num_topics, id2word))
        c, tc = timed(lambda: get_coherence(model=m, texts=docs, dictionary=id2word))
        d, td = timed(lambda: get_diversity(get_topics_lda(m, id2word)))
        t = tt + tc + td
        s = c * d
        tw = get_top_5_l(m, corpus, docs, id2word)
        r["c"].append(c)
        r["d"].append(d)
        r["t"].append(t)
        r["s"].append(s)
        r["tt"].append(tt)
        r["tc"].append(tc)
        r["td"].append(td)
        r["tw"].append(tw)
        if all(s >= x for x in r["s"]):
            twt = json.dumps(get_top_5_l(m, corpus, docs, id2word))
    ds_runs[lv] = {k: json.dumps(v) for (k, v) in r.items()}
    r = {k: mean(v) for (k, v) in r.items() if k != "tw"}
    r["twt"] = twt
    ds[lv] = r


# %% save ds
ds_df = pd.DataFrame.from_dict(ds, orient="index")
ds_df.to_csv("./results/haruna.csv")
ds_runs_df = pd.DataFrame.from_dict(ds_runs, orient="index")
ds_runs_df.to_csv("./results/haruna_runs.csv")
