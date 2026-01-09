# arXiv Paper Classifier

![arXiv ML](arxiv-ML-landscape.webp)


This project develops an end-to-end ML pipeline for classifying scientific research papers into subject categories based on their textual content. The goal is to build a system that can ingest paper metadata and predict relevant research domains, with emphasis on robust MLOps practices including reproducibility, containerization, continuous integration, and cloud deployment.

## Data

We will use this arXiv dataset from Kaggle, which contains metadata for approximately 2.4 million papers including titles, abstracts, author lists, and hierarchical category labels.

https://www.kaggle.com/datasets/Cornell-University/arxiv


For initial development and faster iteration, we subsample to a manageable subset focusing on categories with clear boundaries. The arXiv category taxonomy provides ground truth labels for supervised training without requiring manual annotation.

## Model

We will use pretrained transformer models for multi-label text classification, with the input being concatenated title and abstract and the output a probability distribution over target categories. Scientific text contains domain-specific terminology that benefits from pretrained representations, so we avoid training from scratch.

Our initial approach scales with available compute. The baseline is sentence-transformers to embed abstracts into fixed vectors, followed by a lightweight classifier (logistic regression or small MLP). This requires no GPU for training since embeddings can be precomputed. 

If we need better performance, we can fine-tune DistilBERT (~66M parameters) with a frozen base, updating only the classification head. With access to an RTX 4070 (12GB VRAM), full fine-tuning of DistilBERT or SciBERT on a 50-100k paper subset is realistic and can be done in under an hour per epoch.

We start with the embedding approach to build out the pipeline, then swap in fine-tuning once the MLOps infrastructure is in place.

## Focus

The primary focus of this project is the MLOps infrastructure surrounding the model rather than achieving state-of-the-art classification performance. Key components include: version-controlled data pipelines using DVC backed by cloud storage, containerized training and inference environments with Docker, experiment tracking and hyperparameter logging with Weights and Biases, configuration management using Hydra, CI/CD pipelines via GitHub Actions for automated testing and linting, cloud training on GCP Vertex AI, deployment of inference as a FastAPI service on Cloud Run, and monitoring for input data drift in production.

## Deliverable

The deliverable is a functioning classification API with documented model performance, reproducible training pipeline, and comprehensive MLOps tooling as specified in the course requirements.
