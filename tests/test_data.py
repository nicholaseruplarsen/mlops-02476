import torch

from arxiv_classifier.data import arxiv_dataset, load_label_encoder


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


def test_all_splits_have_valid_labels():
    """Verify all splits have labels within the valid range defined by label_encoder."""
    encoder = load_label_encoder()
    num_categories = len(encoder["label_to_id"])

    for split in ["train", "val", "test", "calibration"]:
        labels = torch.load(f"data/processed/{split}_labels.pt", weights_only=True)
        assert labels.min() >= 0, f"{split} has negative labels"
        assert labels.max() < num_categories, f"{split} has labels >= {num_categories}"
