from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from pathhier.nn.boolean_f1 import BooleanF1
import pathhier.constants as constants


@Model.register("pw_aligner")
class PWAlignNN(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pathway_encoder: Seq2VecEncoder,
                 pw_encoder: Seq2VecEncoder,
                 decision_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 threshold: float = constants.NN_DECISION_THRESHOLD) -> None:
        super(PWAlignNN, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.pathway_encoder = pathway_encoder
        self.pw_encoder = pw_encoder
        self.decision_feedforward = decision_feedforward
        self.sigmoid = torch.nn.Sigmoid()
        self.accuracy = BooleanF1()
        self.loss = torch.nn.BCELoss()
        self.threshold = threshold

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                kb_cls: Dict[str, torch.LongTensor],
                pw_cls: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Embed pathway and PW class, classify as match/non-match
        """

        # embed and encode pathway
        embedded_pathway = self.text_field_embedder(kb_cls)
        pathway_mask = get_text_field_mask(kb_cls)
        encoded_pathway = self.pathway_encoder(embedded_pathway, pathway_mask)

        embedded_pw_cls = self.text_field_embedder(pw_cls)
        pw_cls_mask = get_text_field_mask(pw_cls)
        encoded_pw_cls = self.pw_encoder(embedded_pw_cls, pw_cls_mask)

        # concatenate outputs
        aggregate_input = torch.cat([
            encoded_pathway,
            encoded_pw_cls
        ], dim=-1)

        # run aggregate through a decision layer and sigmoid function
        decision_output = self.decision_feedforward(aggregate_input)

        sigmoid_output = self.sigmoid(decision_output)

        # build output dictionary
        output_dict = dict()
        output_dict["score"] = sigmoid_output
        predicted_label = (sigmoid_output >= self.threshold)
        output_dict["predicted_label"] = predicted_label

        if label is not None:
            # compute loss and accuracy
            loss = self.loss(sigmoid_output.float(), label.float())
            self.accuracy(predicted_label, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, accuracy, f1 = self.accuracy.get_metric(reset)
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'PWAlignNN':
        text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("text_field_embedder"))
        pathway_encoder = Seq2VecEncoder.from_params(params.pop("pathway_encoder"))
        pw_encoder = Seq2VecEncoder.from_params(params.pop("pw_encoder"))
        decision_feedforward = FeedForward.from_params(params.pop("decision_feedforward"))

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   pathway_encoder=pathway_encoder,
                   pw_encoder=pw_encoder,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)