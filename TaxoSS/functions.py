from re import A
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import pkgutil
from io import BytesIO

data = pkgutil.get_data('TaxoVec', "data/card_cache.csv")
df_card_cache = pd.read_csv(BytesIO(data))
ks = [wn.synset(x.split("(")[1].split(")")[0][1:-1]) for x in df_card_cache['keys'].tolist()]
vs = df_card_cache['card'].tolist()
card_cache = {k:v for k,v in zip(ks,vs)}

data_star = pkgutil.get_data('TaxoVec', "data/card_cache_star.csv")
df_card_cache_star = pd.read_csv(BytesIO(data_star))
ks_star = [wn.synset(x.split("(")[1].split(")")[0][1:-1]) for x in df_card_cache_star['keys'].tolist()]
vs_star = df_card_cache_star['card'].tolist()
card_cache_star = {k:v for k,v in zip(ks_star,vs_star)}


def get_cardinality(wordsynset):
    return card_cache[wordsynset]

def get_cardinality_star(wordsynset):
    return card_cache_star[wordsynset]


def information_content(wordsynset):
    """
    Gets the Information Content of the given synset
    """
    card =  get_cardinality(wordsynset)
    return -np.log10(card / 96308)


class HSS(object):

    def __init__(self, word1, word2):

        self.word1 = word1
        self.word2 = word2
        
        self.a = wn.synsets(word1)
        self.b = wn.synsets(word2)
        combs = list(itertools.product(self.a, self.b))
        all_LCA = [comb[0].lowest_common_hypernyms(comb[1]) for comb in combs]
        all_LCA = [LCA[0] for LCA in all_LCA if LCA !=[]]
        self.LCA_freq = dict(Counter(all_LCA))
        self.unique_LCA = list(set(all_LCA))
        self.unique_LCA = [x for x in self.unique_LCA if '.n.' in str(x)]

        self.LCA_IC = {LCA:information_content(LCA) for LCA in self.unique_LCA}
        self.LCA_cardinality = {LCA:get_cardinality(LCA) for LCA in self.unique_LCA}
        self.LCA_cardinality_star = {LCA:get_cardinality_star(LCA) for LCA in self.unique_LCA}
    
    def probability(self, LCA):
        """probability_v2"""
    
        aa =  self.LCA_freq[LCA] / (self.LCA_cardinality_star[LCA])**2
        bb = self.LCA_cardinality[LCA] / 96308
        c0 = []
        for l in self.unique_LCA:
            temp = (self.LCA_freq[l] / (self.LCA_cardinality_star[l])**2) * (self.LCA_cardinality[l] / 96308)
            c0.append(temp)
        c = sum(c0)

        return (aa * bb) / c
    
    def similarity(self):
        """HSS_v2"""

        if (len(self.a)==0) & (len(self.b)==0):
            return(self.word1 + ' and ' + self.word2 + ' out of vocabulary')
        elif len(self.a)==0:
            return(self.word1 + ' out of vocabulary')
        elif len(self.b)==0:
            return(self.word2 + ' out of vocabulary')
        else:
            temp = [self.probability(LCA)* self.LCA_IC[LCA] for LCA in self.unique_LCA]
            return sum(temp)


class IC_similarities(object):
    
    def __init__(self, word1, word2, ic_file):
        
        self.word1 = word1
        self.word2 = word2
        
        self.a = wn.synsets(word1)
        self.b = wn.synsets(word2)
        combs = list(itertools.product(self.a, self.b))
        all_LCA = [comb[0].lowest_common_hypernyms(comb[1]) for comb in combs]
        all_LCA = [LCA[0] for LCA in all_LCA if LCA !=[]]
        all_LCA = list(set(all_LCA))
        all_LCA_names = [x.name().split('.')[0] for x in all_LCA]
        
        self.all_LCA_names = all_LCA_names
        self.ic_file = ic_file
        self.ic1 = ic_file[ic_file.word== word1].IC_10.values
        self.ic2 = ic_file[ic_file.word== word2].IC_10.values
        self.word1 = word1
        self.word2 = word2
        try:
            self.resnik_base = max(ic_file[ic_file.word.isin(all_LCA_names)]['IC_10'].to_list())
        except ValueError:
            self.resnik_base = 0
        
    def resnik(self):
        """
        Guided by the intuition that the similarity between a pair of concepts
        may be judged by “the amount of shared information” Resnik defined
        the similarity between two concepts as the IC of their Lowest Common 
        Subsumer (LCS) noted as LCS(c1,c2)
        """
        if (len(self.a)==0) & (len(self.b)==0):
            return(self.word1 + ' and ' + self.word2 + ' out of vocabulary')
        elif len(self.a)==0:
            return(self.word1 + ' out of vocabulary')
        elif len(self.b)==0:
            return(self.word2 + ' out of vocabulary')
        else:
            return self.resnik_base
    
    def jiang_conrath(self):
        """
        This approach subtracts the IC of the LCS from the sum of the ICs of 
        the individual concepts. It is worth noting that this is a 
        dissimilarity measure because the more different the terms are, the 
        higher the difference between their ICs and the IC of their LCS will be.
        """
        res = (self.ic1 + self.ic2) - (2 * self.resnik_base)

        if (len(self.a)==0) & (len(self.b)==0):
            return(self.word1 + ' and ' + self.word2 + ' out of vocabulary')
        elif len(self.a)==0:
            return(self.word1 + ' out of vocabulary')
        elif len(self.b)==0:
            return(self.word2 + ' out of vocabulary')
        elif len(res)==0:
            return 0
        else:
            return res[0]
    
    def lin(self):
        """
        This similarity measure uses the same elements of jiang_conrath but in 
        a different way
        """
        res = (2 * self.resnik_base) / (self.ic1 + self.ic2)

        if (len(self.a)==0) & (len(self.b)==0):
            return(self.word1 + ' and ' + self.word2 + ' out of vocabulary')
        elif len(self.a)==0:
            return(self.word1 + ' out of vocabulary')
        elif len(self.b)==0:
            return(self.word2 + ' out of vocabulary')
        elif len(res)==0:
            return 0
        else:    
            return res[0]
    
    def pirro(self):
        """
        This similarity measure [46] is conceptually similar to the previous 
        ones, but is based on the feature-based the- ory of similarity described
        by Tversky. According to Tversky, the similarity of a concept c1
        to a concept c2 is a function of the features common to c1 and c2,
        those in c1 but not in c2 and those in c2 but not in c1
        """
        res = (3 * self.resnik_base) - self.ic1 - self.ic2

        if self.word1 == self.word2:
            return 1
        elif (len(self.a)==0) & (len(self.b)==0):
            return(self.word1 + ' and ' + self.word2 + ' out of vocabulary')
        elif len(self.a)==0:
            return(self.word1 + ' out of vocabulary')
        elif len(self.b)==0:
            return(self.word2 + ' out of vocabulary')
        elif len(res)==0:
            return 0
        else:
            return res[0]
    
    def meng(self):
        """
        This measure is based on the method of Lin
        """
        if (len(self.a)==0) & (len(self.b)==0):
            return(self.word1 + ' and ' + self.word2 + ' out of vocabulary')
        elif len(self.a)==0:
            return(self.word1 + ' out of vocabulary')
        elif len(self.b)==0:
            return(self.word2 + ' out of vocabulary')
        else:
            return np.exp(self.lin()) - 1
    
    def similarity_benchmark(self):
        print(f'resnik:        {round(self.resnik_base, 2)}')
        print(f'jiang_conrath: {round(self.jiang_conrath(), 2)}')
        print(f'lin:           {round(self.lin(), 2)}')
        print(f'pirro:         {round(self.pirro(), 2)}')
        print(f'meng:          {round(self.meng(), 2)}')
        

def get_wn_paths(word1, word2, kind):
    a = wn.synsets(word1)
    b = wn.synsets(word2)
    all_sim = []
    for s1 in a:
        for s2 in b:
            try:
                if kind == 'path_sim':
                    similarity = s1.path_similarity(s2)
                elif kind == 'lcs':
                    similarity = s1.lch_similarity(s2)
                elif kind == 'wup':
                    similarity = s1.wup_similarity(s2)

                similarity = s1.path_similarity(s2)
                if similarity:
                    all_sim.append(similarity)
            except :
                return 0
            # else:
            #     all_sim.append(0)
    if (len(a)==0) & (len(b)==0):
            return(word1 + ' and ' + word2 + ' out of vocabulary')
    elif len(a)==0:
        return(word1 + ' out of vocabulary')
    elif len(b)==0:
        return(word2 + ' out of vocabulary')
    else:
        if len(all_sim) > 0:
            if kind == 'path_sim':
                shortest = round(min(all_sim), 4)
            if kind == 'lcs':
                shortest = round(max(all_sim), 4)
            if kind == 'wup':
                shortest = round(max(all_sim), 4)
        else:
            shortest = 0
        return shortest


def seco(word):
    syns = wn.synsets(word)
    for syn in syns:
        hypos = syn.hyponyms()
        ICs = []
        if len(hypos) != 0:
            IC = 1 - (np.log10(len(hypos) + 1) / np.log10(96308))
            ICs.append(IC)
        return max(ICs)

def resnik_seco(word1, word2):
    a = wn.synsets(word1)
    b = wn.synsets(word2)
    combs = list(itertools.product(a, b))
    all_LCA = [comb[0].lowest_common_hypernyms(comb[1]) for comb in combs]
    all_LCA = [LCA[0] for LCA in all_LCA if LCA !=[]]
    all_LCA = list(set(all_LCA))
    all_LCA_names = [x.name().split('.')[0] for x in all_LCA]

    seco_vals = []
    for name in all_LCA_names:
        try:
            seco_vals.append(seco(name))
            #print(seco(name))
        except ValueError:
            pass
    
    if (len(a)==0) & (len(b)==0):
            return(word1 + ' and ' + word2 + ' out of vocabulary')
    elif len(a)==0:
        return(word1 + ' out of vocabulary')
    elif len(b)==0:
        return(word2 + ' out of vocabulary')
    else:
        return max(seco_vals)


def semantic_similarity(w1: str, w2: str, kind: str = 'hss', ic: str = 'data/information_content_default.csv') -> float:
    """
    Returns the similarity for the given words using the given metric

    args
    w1 : The first word we want to get its similarity with w2
    w1 : The second word we want to get its similarity with w1
    kind: Metric used to calculate the similarity (default: resnik)
    ic: Path to the ic file (default: 'data/information_content_default.csv')

    returns:
    A float denoting the similarity between the given words
    """
    data_ic = pkgutil.get_data('TaxoVec', ic)
    ic_file = pd.read_csv(BytesIO(data_ic))

    if kind == 'path_sim':
        return get_wn_paths(w1, w2, kind='path_sim')
    elif kind == 'lcs':
        return get_wn_paths(w1, w2, kind='lcs')
    elif kind == 'wup':
        return get_wn_paths(w1, w2, kind='wup')
    elif kind == 'resnik':
        return IC_similarities(w1, w2, ic_file).resnik()
    elif kind == 'jcn':
        return IC_similarities(w1, w2, ic_file).jiang_conrath()
    elif kind == 'lin':
        return IC_similarities(w1, w2, ic_file).lin()
    elif kind == 'seco':
        return resnik_seco(w1, w2)
    elif kind == 'hss':
        return HSS(w1, w2).similarity()
