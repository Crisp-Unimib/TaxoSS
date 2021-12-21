from collections import Counter
from nltk.corpus import wordnet as wn
import pandas as pd



def cardinality(wordsynset):
    """Gets the cardinality of the given synset
    """
    #global card_cache

    if wordsynset in card_cache.keys():
    
        return card_cache[wordsynset]
    
    num = 0
    listak = [wordsynset]
    while len(listak) > 0:
        for item in listak:
            
            num += len(item.hyponyms())

            listak.remove(item)
            for hypo in item.hyponyms():
                listak.append(hypo)

    # Adding 1 to count for the root
    #card_cache[wordsynset] = num + 1
    return num


def cardinality_star(wordsynset):
    """Gets the cardinality of the given synset without counting the last level
    """
    num = 0
    listak = [wordsynset]
    while len(listak) > 0:
        for item in listak:
            rich_nodes = sum([1 for x in item.hyponyms() if cardinality(x) != 1])
            
            num += rich_nodes
            listak.remove(item)
            for hypo in item.hyponyms():
                listak.append(hypo)

    # Adding 1 to count for the root
    return num + 1


def make_cardinality_star_file():
    """an iterator that goes throw synsets and calculates their cardinality
    """
    wordsynset = wn.synset('entity.n.01')
    global card_cache_star
    card_cache_star = {}
    
    def make_cardinality_star_0(wordsynset):
        """Adds the wordsyn cardinality if it's not already there
        """
        
        if wordsynset not in card_cache_star.keys():

            num = 0
            listak = [wordsynset]
            while len(listak) > 0:
                for item in listak:
                    rich_nodes = sum([1 for x in item.hyponyms() if cardinality(x) != 1])
                    num += rich_nodes
                    listak.remove(item)
                    for hypo in item.hyponyms():
                        listak.append(hypo)

            card_cache_star[wordsynset] = num + 1
    
    
    if wordsynset not in card_cache_star.keys():
        listak = [wordsynset]
        
        while len(listak) > 0:
            for item in listak:
                make_cardinality_star_0(item)
                listak.remove(item)
                for hypo in item.hyponyms():
                    listak.append(hypo)
    

def make_cardinality_file():
    """an iterator that goes throw synsets and calculates their cardinality
    """
    
    wordsynset = wn.synset('entity.n.01')
    global card_cache
    card_cache = {}
    
    def make_cardinality_0(wordsynset):
        """Adds the wordsyn cardinality if it's not already there
        """
        
        if wordsynset not in card_cache.keys():

            num = 0
            listak = [wordsynset]
            while len(listak) > 0:
                for item in listak:
                    num += len(item.hyponyms())
                    listak.remove(item)
                    for hypo in item.hyponyms():
                        listak.append(hypo)

            card_cache[wordsynset] = num + 1
    
    
    if wordsynset not in card_cache.keys():
        listak = [wordsynset]
        
        while len(listak) > 0:
            for item in listak:
                make_cardinality_0(item)
                listak.remove(item)
                for hypo in item.hyponyms():
                    listak.append(hypo)


make_cardinality_file()
make_cardinality_star_file()

pd.DataFrame({'keys':list(card_cache.keys()), 'card':list(card_cache.values())}).to_csv('data/card_cache.csv', index=False)
pd.DataFrame({'keys':list(card_cache_star.keys()), 'card':list(card_cache_star.values())}).to_csv('data/card_cache_star.csv', index=False)