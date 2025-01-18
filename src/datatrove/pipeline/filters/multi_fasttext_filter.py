from collections import defaultdict
from typing import Tuple

import numpy as np
from fasttext.FastText import _FastText

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import SPLIT_TEXT_DOCUMENTS, split_into_parts


class MultiFastTextClassifierFilter(BaseFilter):
    """
    Only keeps documents that have
    - AT LEAST ONE of the labels in `keep_labels` with a score above the configured threshold, or
    - NONE of the labels in `remove_labels` with a score above the configured threshold.

    You can only supply one of these, to avoid conflicts. Use multiple filters if you need to. If you supply
    neither, the block will simply annotate each document with the labels (set `save_labels_in_metadata=True`)

    Example:
        for `keep_labels=[("math", 0.9)]` will only keep samples with a score on __label__math of at least 0.9
        for `remove_labels=[("math", 0.9)]` will remove samples with a score on __label__math of at least 0.9

    Info to train your own classifier: https://fasttext.cc/docs/en/supervised-tutorial.html

    Args:
        model_url: url to download the model from or local path
        keep_labels: tuple of (label name without "__label__", min score) (or list of such tuples)
        remove_labels: tuple of (label name without "__label__", min score) (or list of such tuples)
        save_labels_in_metadata: whether to save all the label scores in the document metadata
        newline_replacement: str to replace \n with before predicting scores
        filter_mode: predict and filter on DOCUMENT, PARAGRAPH or SENTENCE level
        exclusion_writer:
    """

    name = "👥 MultiFastText"
    _requires_dependencies = [("fasttext", "fasttext-wheel"), "fasteners"]

    def __init__(
        self,
        model_urls: dict[str, str],
        labels: dict[Tuple[str, float] | list[Tuple[str, float]]],
        save_labels_in_metadata: bool = True,
        exclusion_writer: DiskWriter | None = None,
        newline_replacement="",
        filter_mode: str = SPLIT_TEXT_DOCUMENTS,
    ):
        super().__init__(exclusion_writer)
        self.model_urls = model_urls
        self.labels = labels
        self.filter_mode = filter_mode
        self.newline_replacement = newline_replacement
        for name in self.model_urls:
          if isinstance(self.labels[name][0], str):
              self.labels[name] = [self.labels[name]]
        self.save_labels_in_metadata = save_labels_in_metadata
        self._models = None

    @property
    def models(self):

        def build_model(model_url, labels):
            model_file = cached_asset_path_or_download(
                model_url, namespace="filters", subfolder="fasttext", desc="fast-text model"
            )
            model = _FastText(model_file)
            # check label values
            available_labels = [x.removeprefix("__label__") for x in model.labels]
            for label, _ in labels:
                if label not in available_labels:
                    raise ValueError(
                        f"Label '{label}' passed as labels is not available in this "
                        f"FastText model. Available labels: {available_labels}"
                    )
            return model
        if self._models is None:
            self._models = dict()
            for name, model_url in self.model_urls.items():
                self._models[name] = build_model(model_url, self.labels[name])
        return self._models

    def filter(self, doc: Document) -> bool:
        def split_labels(labels):
            keep_labels = []
            remove_labels = []
            for label, min_score in labels:
                if min_score > 0:
                    keep_labels.append((label, min_score))
                else:
                    remove_labels.append((label, min_score))
            if keep_labels and remove_labels:
                raise ValueError("You can only supply one of `keep_labels` or `remove_labels`.")
            return keep_labels, remove_labels
        def check_label_scores(unit_scores, keep_labels, remove_labels):
            if keep_labels:
                return any(
                    unit_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in keep_labels
                )
            else:
                return not remove_labels or not any(
                    unit_scores.get(f"__label__{label}", -9e9) >= min_score for label, min_score in remove_labels
                )

        label_splits = {name: split_labels(self.labels[name]) for name in self.models}
        units = split_into_parts(doc.text, mode=self.filter_mode)
        kept_spans = []
        label_scores = {name: defaultdict(list) for name in self.models}
        for unit in units:
            flags = []
            for name in self.models:
                labels, scores = self.models[name].predict(unit.strip().replace("\n", self.newline_replacement), k=-1)
                if self.save_labels_in_metadata:
                    for label, score in zip(labels, scores):
                        label_scores[name][label].append(score)
                flags.append(check_label_scores(dict(zip(labels, scores)), label_splits[name][0], label_splits[name][1]))
            if any(flags):
                kept_spans.append(unit)
                self.stat_update("kept_span")
            else:
                self.stat_update("removed_span")
        doc.text = "".join(kept_spans)
        if self.save_labels_in_metadata:
            doc.metadata.update({name: {label: np.mean(scores).item() for label, scores in label_scores[name].items()} for name in self.models})
        return not not doc.text.strip()
