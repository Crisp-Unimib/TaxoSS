from nltk.corpus import wordnet as wn
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm

def in_wordnet(word):
    if wn.synsets(word) != []:
        return True
    else:
        return False


def calculate_IC(path_to_corpus, path_to_ic):

    corpus = pd.read_csv(path_to_corpus).dropna().iloc[:, 0].to_list()

    res = []
    for line in tqdm(corpus):
        for token in line.split():
            token = token.lower()
            res.append(token)

    cnt = dict(Counter(res))

    df = pd.DataFrame.from_dict(cnt, orient='index', columns=['freq']).reset_index()
    df.columns = ['word', 'freq']
    df['IC_10'] = df['freq'].apply(lambda x: -np.log10(x / len(res)))

    df.to_csv(path_to_ic, index=False)