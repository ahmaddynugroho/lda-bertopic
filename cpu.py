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
    docs = docs[:100]
    if ast_parse:
        docs = [ast.literal_eval(d) for d in docs]
    return docs


# %% train bertopic
def train_b(docs, **bargs):
    pipe = make_pipeline(TfidfVectorizer(), TruncatedSVD(100))
    model = BERTopic(embedding_model=pipe, nr_topics="auto", **bargs)
    model.fit_transform(docs)
    return model


# %% get bertopic training args
def get_bargs(v):
    bargs = {}
    if "wg" in v:
        vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words)
        bargs["vectorizer_model"] = vectorizer_model
    return bargs


# %% get bertopic coherence
def get_coherence_b(model, docs):
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    topics = get_topics_bertopic(model, all=True)
    c = get_coherence(topics=topics, texts=tokens, dictionary=dictionary)
    return c


# %% used to get training time and basically any lambda fn
def timed(f):
    start = time()
    result = f()
    return result, time() - start


# %% variants
vs = "ar,awg,alwg,sr,swg,slwg".split(",")

# %% bertopic training
path = "./datasets/dsn"
ds = {}
for v in (tv := tqdm(vs)):
    bv = f"b_{v}"
    tv.set_description(f"bertopic {bv}")
    _path = f"{path}/{bv}.csv"
    docs = get_docs(_path)
    r = {
        "c": [],
        "d": [],
        "tt": [],
        "tc": [],
        "td": [],
        "t": [],
        "s": [],
    }
    for i in range(3):
        m, tt = timed(lambda: train_b(docs, **get_bargs(bv)))
        c, tc = timed(lambda: get_coherence_b(m, docs))
        d, td = timed(lambda: get_diversity(get_topics_bertopic(m, all=True)))
        t = tt + tc + td
        s = c * d
        r["c"].append(c)
        r["d"].append(d)
        r["t"].append(t)
        r["s"].append(s)
        r["tt"].append(tt)
        r["tc"].append(tc)
        r["td"].append(td)
        if all(s >= x for x in r["s"]):
            bt = json.dumps(get_topics_bertopic(m, all=True))
    r = {k: mean(v) for (k, v) in r.items()}
    r["tw"] = bt
    ds[bv] = r


# %% bertopic training
for v in (tv := tqdm(vs)):
    lv = f"l_{v}"
    tv.set_description(f"lda {v}")
    _path = f"{path}/{lv}.csv"
    docs = get_docs(_path, ast_parse=True)
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(d) for d in docs]
    num_topics = len(json.loads(ds[f"b_{v}"]["tw"]))
    r = {
        "c": [],
        "d": [],
        "tt": [],
        "tc": [],
        "td": [],
        "t": [],
        "s": [],
    }
    for i in range(3):
        m, tt = timed(lambda: LdaMulticore(corpus, num_topics, id2word))
        c, tc = timed(lambda: get_coherence(model=m, texts=docs, dictionary=id2word))
        d, td = timed(lambda: get_diversity(get_topics_lda(m, id2word)))
        t = tt + tc + td
        s = c * d
        r["c"].append(c)
        r["d"].append(d)
        r["t"].append(t)
        r["s"].append(s)
        r["tt"].append(tt)
        r["tc"].append(tc)
        r["td"].append(td)
        if all(s >= x for x in r["s"]):
            lt = json.dumps(get_topics_lda(m, id2word))
    r = {k: mean(v) for (k, v) in r.items()}
    r["tw"] = lt
    ds[lv] = r


# %% save ds
ds_df = pd.DataFrame.from_dict(ds, orient="index")
ds_df.to_csv("./results/haruna.csv")
