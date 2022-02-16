from TaxoSS.functions import semantic_similarity
from nltk.corpus import wordnet as wn
import random
from random import seed
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

def imap_bar(func, args, n_processes = (multiprocessing.cpu_count()-1)):
    
    p = Pool(n_processes,maxtasksperchild=5000)
    res_list = []
    with tqdm(total = len(args),mininterval=60) as pbar:
        for res in tqdm(p.map(func, args)):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

# only unigrams
all_nouns = [word for synset in wn.all_synsets('n') for word in synset.lemma_names() if '_' not in word]
all_nouns = list(set(all_nouns))

seed(1995)
random_nouns = random.sample(all_nouns, 20000)
word1=random_nouns[0:10000]
word2=random_nouns[10000:20000]

hss = [semantic_similarity(w1, w2, 'hss') for w1, w2 in tqdm(zip(word1, word2))]

# saving as tsv file
pd.DataFrame({'word1':word1, 'word2':word2, 'hss':hss}).to_csv('benchmark/example.tsv', sep="\t")