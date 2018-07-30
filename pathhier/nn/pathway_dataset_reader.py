from typing import Dict
import logging

from overrides import overrides
import json
import tqdm
import spacy

from nltk.tokenize import RegexpTokenizer

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import Field, TextField
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from pathhier.nn.boolean_field import BooleanField


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pw_aligner")
class PathwayDatasetReader(DatasetReader):
    """
    Reads instances from a jsonlines file where each line is in the following format:
    {"match": X, "pathway: (id, pathway_string), "pw_cls": (id, pw_class_string)}
     X in [0, 1]
    and converts it into a ``Batch`` suitable for alignment.
    Parameters
    ----------
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexer: Dict[str, TokenIndexer] = None) -> None:
        super(PathwayDatasetReader, self).__init__(False)
        self._token_indexer = token_indexer or \
                                   {'w2v_tokens': SingleIdTokenIndexer(namespace="tokens")}
        self._tokenizer = tokenizer or WordTokenizer()
        self.tokenizer = RegexpTokenizer(r'[A-Za-z\d]+')
        self.nlp = spacy.load('en_core_web_sm')

    @overrides
    def read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        # open data file and read lines
        with open(file_path, 'r') as train_file:
            logger.info("Reading pathway alignment data from jsonl dataset at: %s", file_path)
            for line in tqdm.tqdm(train_file):
                training_pair = json.loads(line)
                kb_cls = training_pair['kb_cls']
                pw_cls = training_pair['pw_cls']
                label = bool(training_pair['label'])

                # convert entry to instance and append to instances
                instances.append(self.text_to_instance(kb_cls, pw_cls, label))

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Batch(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         kb_cls: list,
                         pw_cls: list,
                         label: bool) -> Instance:
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        # tokenize string
        kb_cls_tokens = self._tokenizer.tokenize(kb_cls)
        pw_cls_tokens = self._tokenizer.tokenize(pw_cls)

        # add entity name fields
        fields['kb_cls'] = TextField(kb_cls_tokens, self._token_indexer)
        fields['pw_cls'] = TextField(pw_cls_tokens, self._token_indexer)

        # add boolean label (0 = no match, 1 = match)
        fields['label'] = BooleanField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'PathwayDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexer = TokenIndexer.dict_from_params(params.pop('token_indexer', {}))
        params.assert_empty(cls.__name__)
        return PathwayDatasetReader(tokenizer=tokenizer,
                                    token_indexer=token_indexer)
