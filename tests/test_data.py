import torch

from arxiv_classifier.data import arxiv_dataset


def test_arxiv_data_loading_correctly():
    """Crude data validity test."""
    (train_texts, train_labels), (test_texts, test_labels) = arxiv_dataset()

    # Check lengths match
    assert len(train_texts) == len(train_labels)
    assert len(test_texts) == len(test_labels)

    # Check types
    assert isinstance(train_labels, torch.Tensor)
    assert train_labels.dtype == torch.long

    # Check texts are strings
    assert isinstance(train_texts[0], str)
    assert len(train_texts[0]) > 0
