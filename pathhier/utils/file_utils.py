
def read_dat_lines(filename, comment=None):
    """
    Read all non-blank lines from `filename` dat file.
    Skip any lines that begin the comment character.
    :param filename:
    :param comment: ignore lines starting with this text or character
    :return:
    """
    with open(filename, 'rb') as f:
        for l in f:
            l_decoded = l.decode('unicode_escape').strip()
            if comment and l_decoded.startswith(comment):
                continue
            yield l_decoded
