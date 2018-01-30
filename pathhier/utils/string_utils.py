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
    return tuple([t for t in tok.tokenize(normalize_string(s)) if t not in stop])


def get_character_ngrams(s, n):
    """
    Generate character ngrams of length l from string
    :param s: input string
    :param l: length of ngrams
    :return:
    """
    s_padded = '\0' * (n - 1) + s + '\0' * (n - 1)
    return zip(*[s_padded[i:] for i in range(n)])