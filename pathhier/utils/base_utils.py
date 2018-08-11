import jsonlines
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from overrides import overrides

from collections import defaultdict


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
) -> requests.Session:
    """
    Create a requests retry session
    :param retries:
    :param backoff_factor:
    :param status_forcelist:
    :param session:
    :return:
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def flatten(l):
    """
    Flatten list of lists
    :param l:
    :return:
    """
    return [item for sublist in l for item in sublist]


def read_jsonlines(fpath):
    """
    Read all entries from jsonlines file
    :param fpath:
    :return:
    """
    data = []
    with jsonlines.open(fpath, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data


