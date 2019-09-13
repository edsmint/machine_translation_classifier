from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
import torch.nn.functional as F
import numpy

@Model.register("mt_classifier")
class MTClassifier(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output
        projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    """

    def __init__(self, vocab: Vocabulary,
                 en_text_field_embedder: TextFieldEmbedder, # Embeddings for English
                 de_text_field_embedder: TextFieldEmbedder, # Embeddings for German
                 en_pos_text_field_embedder: TextFieldEmbedder,
                 en_encoder: Seq2VecEncoder,
                 de_encoder: Seq2VecEncoder,
                 pos_encoder: Seq2VecEncoder,
                 feedforward: Optional[FeedForward] = None) -> None:
        super().__init__(vocab)
        self.en_text_field_embedder = en_text_field_embedder
        self.de_text_field_embedder = de_text_field_embedder
        self.en_pos_text_field_embedder = en_pos_text_field_embedder

        self.en_encoder = en_encoder
        self.de_encoder = de_encoder
        self.pos_encoder = pos_encoder

        self.num_classes = 2

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.feedforward = torch.nn.Linear(in_features=(self.en_encoder.get_output_dim() + self.pos_encoder.get_output_dim() + 8),
                                          out_features=2)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }

    @overrides
    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor],
                candidate: Dict[str, torch.LongTensor],
                candidate_pos: Dict[str, torch.LongTensor],
                features: List[int] = None,
                label: List[int] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        source : ``Dict[str, torch.LongTensor]``, required
            The translation's source sentence.
        candidate : ``Dict[str, torch.LongTensor]``, required
            The translation candidate.
        label : List[int], optional (default = None)
            Whether the candidate is human- or machine-translated, if known.
        """

        # de_embedded_text = self.de_text_field_embedder(source)
        en_embedded_text = self.en_text_field_embedder(candidate)
        pos_embedded_text = self.en_pos_text_field_embedder(candidate_pos)

        # de_mask = util.get_text_field_mask(source)
        en_mask = util.get_text_field_mask(candidate)
        pos_mask = util.get_text_field_mask(candidate_pos)

        # de_encoded_text = self.de_encoder(de_embedded_text, de_mask)
        en_encoded_text = self.en_encoder(en_embedded_text, en_mask)
        pos_encoded_text = self.pos_encoder(pos_embedded_text, pos_mask)

        encoded_combined = torch.cat([pos_encoded_text, en_encoded_text, features], -1)

        logits = self.feedforward(encoded_combined)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits, "source": source, "candidate": candidate}
        # print(logits)
        # print(logits.size())
        if label is not None:
            # print(label.size())
            output["loss"] = self.loss_function(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)

        return output


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Taken from https://github.com/allenai/allennlp-as-a-library-example/blob/master/my_library/models/academic_paper_classifier.py
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        return metrics_to_return