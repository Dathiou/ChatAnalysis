'''
Created on Sep 23, 2016

@author: DAMIEN.THIOULOUSE
'''

class SentimentScore(object):
    _lookup = None

    def __init__(self, sent_file):
        if SentimentScore._lookup is None:
            SentimentScore._lookup = self._build_lookup(sent_file)

    def _build_lookup(self, sent_file):
        scores = {} # initialize an empty dictionary
        for line in sent_file:
            term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
            scores[term] = int(score)  # Convert the score to an integer.

        return scores

    def get_score(self, word):
        return self._lookup.get(word, 0)