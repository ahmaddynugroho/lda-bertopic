import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

process_stem = StemmerFactory().create_stemmer().stem
process_stopword = StopWordRemoverFactory().create_stop_word_remover().remove

translation = str.maketrans('', '', string.punctuation)
def process_punctuation(s):
    return s.translate(translation)

def build_dataset(texts):
    # corpus, dict, texts, embedding
    r = {} # result
    r['texts'] = texts
    temp = [
        process_stopword(
            process_stem(
                process_punctuation(s)))
        for s in texts ]
