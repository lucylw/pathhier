from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor


@Predictor.register('pw_aligner')
class PathwayPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(PathwayPredictor, self).__init__(model, dataset_reader)

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like
        ``{"tags": [...], "class_probabilities": [[...], ..., [...]]}``
        """
        kb_cls = json["kb_cls"]
        pw_cls = json["pw_cls"]
        label = json["label"]
        return self._dataset_reader.text_to_instance(kb_cls, pw_cls, label)
