from typing import Set

# string utility functions

def normalize_string(s):
    """
    Process string name; strip string, lowercase, and replace some characters
    :param s: string
    :return:
    """
    return s.strip().lower().replace('-', ' ').replace('_', ' ')


def tokenize_string(s, tok, stop):
    """
    Process name string and return tokenized words minus stop words
    :param s: string
    :param tok: tokenizer
    :param stop: set of stop words
    :return:
    """
    toks = tuple([t for t in tok.tokenize(normalize_string(s))])
    keep_toks = tuple([t for t in toks if t not in stop])
    return keep_toks if keep_toks else toks


def get_token_ngrams(s, tok, n):
    """
    Process name string and return token ngrams
    :param s: string
    :param tok: tokenizer
    :param n: length of ngram (2 = bigram, 3 = trigram etc)
    :return:
    """
    toks = [t for t in tok.tokenize(normalize_string(s))]
    toks_padded = ['\0'] * (n - 1) + toks + ['\0'] * (n - 1)
    return zip(*[toks_padded[i:] for i in range(n)])


def get_character_ngrams(s, n):
    """
    Generate character ngrams of length l from string
    :param s: input string
    :param l: length of ngrams
    :return:
    """
    s_padded = '\0' * (n - 1) + normalize_string(s) + '\0' * (n - 1)
    return zip(*[s_padded[i:] for i in range(n)])


def jaccard(a: Set, b: Set):
    """
    return jaccard index between input sets
    :param a:
    :param b:
    :return:
    """
    return len(a.intersection(b)) / len(a.union(b))