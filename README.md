# arXiv Paper Classifier

![arXiv ML](arxiv-ml-landscape.webp)

["Good artists copy, great artists steal - Picasso"](https://www.reddit.com/r/dataisbeautiful/comments/18b1qrd/oc_a_data_map_of_machine_learning_papers_on_arxiv/)

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

## Project checklist

**Chosen project**:

3. [Classification of scientific papers](https://github.com/eyhl/group5-pyg-dtu-mlops)

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed very point on the
checklist for the exam. The parenthesis at the end indicates what module the bullet point is related to.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [ ] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [ ] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [ ] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [ ] Build the docker files locally and make sure they work as intended (M10)
* [ ] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [ ] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] Write unit tests related to the data part of your code (M16)
* [ ] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub
