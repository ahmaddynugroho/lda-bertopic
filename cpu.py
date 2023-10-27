# %% imports
import json
from statistics import mean

import gensim.corpora as corpora
import pandas as pd
from gensim.models.ldamulticore import LdaMulticore
from tqdm.auto import tqdm

from lib import (
    get_bargs,
    get_coherence,
    get_coherence_b,
    get_diversity,
    get_docs,
    get_top_7_b,
    get_top_7_l,
    get_topics_lda,
    timed,
    train_b,
)

# %% variants
vs = "ar,awg,alwg,sr,swg,slwg".split(",")

# %% bertopic training
path = "./datasets/ds"
ds_runs = {}
ds = {}
for v in (tv := tqdm(vs, position=0)):
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
        "t5": [],
        "k": [],
    }
    for i in tqdm(range(5), desc="runs", position=1, leave=False):
        mt, tt = timed(lambda: train_b(docs, **get_bargs(bv)))
        m, topics = mt
        cktw, tc = timed(lambda: get_coherence_b(m, topics, docs))
        c, k, tw = cktw
        d, td = timed(lambda: get_diversity(tw))
        t = tt + tc + td
        s = c * d
        t5 = get_top_7_b(m)
        r["c"].append(c)
        r["d"].append(d)
        r["t"].append(t)
        r["s"].append(s)
        r["k"].append(k)
        r["tt"].append(tt)
        r["tc"].append(tc)
        r["td"].append(td)
        r["t5"].append(t5)
        if all(s >= x for x in r["s"][:-1]):
            kt = k
            t5t = json.dumps(t5)
    ds_runs[bv] = {k: json.dumps(v) for (k, v) in r.items()}
    r = {k: mean(v) for (k, v) in r.items() if k not in ["t5", "k"]}
    r["t5t"] = t5t
    r["kt"] = kt
    ds[bv] = r

# %% lda training
for v in (tv := tqdm(vs, position=0)):
    lv = f"l_{v}"
    tv.set_description(f"lda {v}")
    _path = f"{path}/{lv}.csv"
    docs = get_docs(_path, ast_parse=True)
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(d) for d in docs]
    r = {
        "c": [],
        "d": [],
        "tt": [],
        "tc": [],
        "td": [],
        "t": [],
        "s": [],
        "t5": [],
        "k": [],
    }
    for i in tqdm(range(5), desc="runs", position=1, leave=False):
        num_topics = json.loads(ds_runs[f"b_{v}"]["k"])[i]
        m, tt = timed(lambda: LdaMulticore(corpus, num_topics, id2word))
        c, tc = timed(
            lambda: get_coherence(model=m, texts=docs, dictionary=id2word))
        d, td = timed(lambda: get_diversity(get_topics_lda(m, id2word)))
        t = tt + tc + td
        s = c * d
        t5 = get_top_7_l(m, corpus)
        r["c"].append(c)
        r["d"].append(d)
        r["t"].append(t)
        r["s"].append(s)
        r["k"].append(num_topics)
        r["tt"].append(tt)
        r["tc"].append(tc)
        r["td"].append(td)
        r["t5"].append(t5)
        if all(s >= x for x in r["s"][:-1]):
            kt = num_topics
            t5t = json.dumps(t5)
    ds_runs[lv] = {k: json.dumps(v) for (k, v) in r.items()}
    r = {k: mean(v) for (k, v) in r.items() if k not in ["t5", "k"]}
    r["t5t"] = t5t
    r["kt"] = kt
    ds[lv] = r

# %% save ds
ds_df = pd.DataFrame.from_dict(ds, orient="index")
ds_df.to_csv("./results/rio.csv")
ds_runs_df = pd.DataFrame.from_dict(ds_runs, orient="index")
ds_runs_df.to_csv("./results/rio_runs.csv")
