from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

import json

@Predictor.register("mt_classifier_pred")
class MTClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in a source and candidate sentences and
    classifies it as a machine- or human-translated sentence.
    """
    def __init__(self, model: Model,
                       dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, source: str, candidate: str) -> JsonDict:
        return self.predict_json({ "source": source, "candidate": candidate })

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        label = outputs['label']
        source = outputs['source']
        candidate = outputs['candidate']
        return source + "\t" + candidate + "\t" + label + "\n"

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "...", "candidate": "..."}
        """
        source = json_dict["source"]
        candidate = json_dict["candidate"]

        # dataset reader handles splitting
        return self._dataset_reader.text_to_instance(source, candidate)

