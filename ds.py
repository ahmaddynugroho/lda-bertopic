import stanza
import pandas as pd
from tqdm.auto import tqdm
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models.phrases import Phrases

stopwords = StopWordRemoverFactory().get_stop_words()

nlp = stanza.Pipeline(
    'id',
    processor='tokenize,pos,lemma',
    use_gpu=False
)

df = pd.read_csv('./datasets/artc_fltr.csv')
docs = df['article']

# docs = docs[:10]
# docs = docs[109:110]

# bertopic
b_sr = {'sentences': []}
b_slwg = {'sentences': []}
for doc in tqdm(docs):
    doc_nlp = nlp(doc)
    for sent in doc_nlp.sentences:
        b_sr['sentences'].append(sent.text)
        _words = []
        for word in sent.words:
            if word.upos == 'PUNCT':
                continue
            if word.lemma:
                _words.append(word.lemma)
            else:
                _words.append(word.text.lower())
        if not _words:
            continue
        _words = ' '.join(_words)
        b_slwg['sentences'].append(_words)
b_sr = pd.DataFrame.from_dict(b_sr)
b_slwg = pd.DataFrame.from_dict(b_slwg)
b_swg = pd.DataFrame.from_dict(b_sr)

path = './datasets/dsn'
b_sr.to_csv(f'{path}/b_sr.csv', index=False)
b_slwg.to_csv(f'{path}/b_slwg.csv', index=False)
b_swg.to_csv(f'{path}/b_swg.csv', index=False)

# lda
l_sr = {'sentences': []}
l_slwg = {'sentences': []}
l_swg = {'sentences': []}
for doc in tqdm(docs):
    doc_nlp = nlp(doc)
    for sent in doc_nlp.sentences:
        _words_r = []
        _words_l = []
        _words = []
        for word in sent.words:
            _words_r.append(word.text)
            if word.upos == 'PUNCT':
                continue
            _words.append(word.text.lower())
            if word.lemma:
                _words_l.append(word.lemma)
            else:
                _words_l.append(word.text.lower())
        if not _words:
            continue
        if not _words_l:
            continue
        if not _words_r:
            continue
        l_sr['sentences'].append(_words_r)
        l_slwg['sentences'].append(_words_l)
        l_swg['sentences'].append(_words)
l_slwg['sentences'] = [[w for w in s if w not in stopwords]
                       for s in l_slwg['sentences']]
l_swg['sentences'] = [[w for w in s if w not in stopwords]
                      for s in l_swg['sentences']]
bigram = Phrases(l_slwg['sentences']).freeze()
trigram = Phrases(bigram[l_slwg['sentences']]).freeze()
l_slwg['sentences'] = list(trigram[bigram[l_slwg['sentences']]])
bigram = Phrases(l_swg['sentences']).freeze()
trigram = Phrases(bigram[l_swg['sentences']]).freeze()
l_swg['sentences'] = list(trigram[bigram[l_swg['sentences']]])

l_sr = pd.DataFrame.from_dict(l_sr)
l_slwg = pd.DataFrame.from_dict(l_slwg)
l_swg = pd.DataFrame.from_dict(l_swg)

l_sr.to_csv(f'{path}/l_sr.csv', index=False)
l_slwg.to_csv(f'{path}/l_slwg.csv', index=False)
l_swg.to_csv(f'{path}/l_swg.csv', index=False)

b_sr
b_slwg
b_swg
l_sr
l_slwg
l_swg
