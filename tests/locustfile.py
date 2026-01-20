"""Load testing for the arXiv classifier API using Locust."""

from locust import HttpUser, task, between


# Sample papers for testing
SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
    },
    {
        "title": "Gradient-based learning applied to document recognition",
        "abstract": "Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns.",
    },
    {
        "title": "A Survey on Knowledge Graphs: Representation, Acquisition, and Applications",
        "abstract": "Human knowledge provides a formal understanding of the world. Knowledge graphs that represent structural relations between entities have become an increasingly popular research direction towards cognition and human-level intelligence.",
    },
]


class ArxivAPIUser(HttpUser):
    """Simulated user for load testing the arXiv classifier API."""

    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5s between tasks

    @task(1)
    def health_check(self):
        """Test the health endpoint."""
        self.client.get("/health")

    @task(10)
    def predict(self):
        """Test the predict endpoint with sample papers."""
        import random

        paper = random.choice(SAMPLE_PAPERS)
        self.client.post("/predict", json=paper)
