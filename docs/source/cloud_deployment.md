# Cloud Deployment

This guide covers deploying the arXiv Paper Classifier API to Google Cloud Run.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and configured
- Docker installed locally
- A GCP project with billing enabled
- A trained model (`best_model.pt`) and label encoder (`label_encoder.json`)

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client        │────▶│   Cloud Run     │────▶│   GCS Bucket    │
│   (HTTP POST)   │     │   (API)         │     │   (Model/Data)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

The API runs as a serverless container on Cloud Run, loading model weights and label encoder from a mounted GCS bucket.

## Step 1: Enable Required Services

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage.googleapis.com
```

## Step 2: Create GCS Bucket and Upload Artifacts

```bash
# Create bucket (choose a unique name)
export BUCKET_NAME=arxiv-classifier-<your-project-id>
gcloud storage buckets create gs://$BUCKET_NAME --location=europe-west1

# Upload model and label encoder
gcloud storage cp models/best_model.pt gs://$BUCKET_NAME/models/
gcloud storage cp data/processed/label_encoder.json gs://$BUCKET_NAME/data/processed/
```

Expected bucket structure:
```
gs://<BUCKET_NAME>/
├── models/
│   └── best_model.pt
└── data/
    └── processed/
        └── label_encoder.json
```

## Step 3: Create Artifact Registry Repository

```bash
export REGION=europe-west1
export PROJECT_ID=$(gcloud config get-value project)

gcloud artifacts repositories create mlops-repo \
  --repository-format=docker \
  --location=$REGION \
  --description="MLOps Docker images"
```

## Step 4: Build and Push Docker Image

Build the image locally first to verify it works, then push to Artifact Registry.

```bash
# Configure Docker for Artifact Registry (one-time setup)
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build image locally
docker build -f dockerfiles/api.dockerfile . -t arxiv-api:latest

# Test locally before pushing (optional but recommended)
docker run -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  arxiv-api:latest

# Tag for Artifact Registry
docker tag arxiv-api:latest \
  $REGION-docker.pkg.dev/$PROJECT_ID/mlops-repo/arxiv-api:latest

# Push to registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/mlops-repo/arxiv-api:latest
```

### Testing Locally

Before pushing, verify the API works:

```bash
# In another terminal, test the health endpoint
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Paper", "abstract": "This is a test abstract."}'
```

## Step 5: Deploy to Cloud Run

```bash
gcloud run deploy arxiv-api \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/mlops-repo/arxiv-api:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --add-volume=name=gcs-data,type=cloud-storage,bucket=$BUCKET_NAME \
  --add-volume-mount=volume=gcs-data,mount-path=/gcs \
  --timeout=300 \
  --cpu-boost \
  --execution-environment=gen2
```

### Deployment Options

| Flag | Description |
|------|-------------|
| `--memory 4Gi` | Memory allocation (transformer models need ~2-4GB) |
| `--cpu 2` | CPU allocation |
| `--timeout 300` | Request timeout in seconds |
| `--cpu-boost` | Extra CPU during startup (helps with model loading) |
| `--execution-environment=gen2` | 2nd gen environment (required for GCS volume mounts) |
| `--min-instances 0` | Scale to zero when idle (cost saving, default) |
| `--max-instances 10` | Maximum concurrent instances |
| `--concurrency 80` | Requests per instance |

## Step 6: Test the Deployment

After deployment, Cloud Run provides a URL (e.g., `https://arxiv-api-xxxxx-ew.a.run.app`).

### Health Check

```bash
export SERVICE_URL=$(gcloud run services describe arxiv-api --region=$REGION --format='value(status.url)')

curl $SERVICE_URL/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "scibert",
  "device": "cpu",
  "num_classes": 176
}
```

### Classify a Paper

```bash
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
  }'
```

Expected response:
```json
{
  "top_category": "cs.CL",
  "predictions": [
    {"category": "cs.CL", "confidence": 0.7823},
    {"category": "cs.LG", "confidence": 0.1456},
    {"category": "cs.AI", "confidence": 0.0412}
  ]
}
```

## Environment Variables

The API supports configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | `scibert` | Model architecture (`scibert` or `sentence_transformer`) |
| `MODEL_PATH` | Auto-detected | Path to model weights |
| `LABEL_ENCODER_PATH` | Auto-detected | Path to label encoder JSON |
| `DEVICE` | `auto` | Compute device (`auto`, `cpu`, `cuda`) |
| `TOP_K` | `3` | Number of predictions to return |

To set environment variables in Cloud Run:

```bash
gcloud run services update arxiv-api \
  --region $REGION \
  --set-env-vars MODEL_TYPE=sentence_transformer,TOP_K=5
```

## Local Development

### Run without Docker

For rapid iteration during development:

```bash
uv run uvicorn arxiv_classifier.api:app --reload --port 8080
```

Access the API at `http://localhost:8080` and interactive docs at `http://localhost:8080/docs`.

### Run with Docker

See [Step 4](#step-4-build-and-push-docker-image) for building and running the Docker image locally.

## Performance and Load Testing

The API has been load tested using [Locust](https://locust.io/). Results on a local machine (CPU, single uvicorn worker):

| Concurrent Users | Requests (30s) | Avg Latency | 95th %ile | Throughput |
|------------------|----------------|-------------|-----------|------------|
| 10 | 750 | 54ms | 84ms | ~27 req/s |
| 50 | 151 | 216ms | 510ms | ~5 req/s |

**Key findings:**
- Single-worker uvicorn with CPU-bound SciBERT inference is the bottleneck
- With 10 concurrent users: 0% failure rate, sub-100ms p95 latency
- With 50 concurrent users: requests queue up, latency degrades

**Scaling options:**
1. **Cloud Run auto-scaling**: Handles load by spawning multiple container instances automatically
2. **Multiple uvicorn workers**: `--workers 4` in the CMD
3. **GPU inference**: Significantly faster model inference (not available on Cloud Run)

### Running Load Tests

```bash
# Install locust (included in dev dependencies)
uv sync --group dev

# Start the API locally
uv run uvicorn arxiv_classifier.api:app --port 8080

# Run load test (in another terminal)
uv run locust -f tests/locustfile.py --host http://localhost:8080

# Or headless mode
uv run locust -f tests/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:8080
```

## Monitoring and Logs

### View Logs

```bash
gcloud run services logs read arxiv-api --region $REGION --limit 50
```

### Stream Logs

```bash
gcloud run services logs tail arxiv-api --region $REGION
```

### Cloud Console

Visit the [Cloud Run Console](https://console.cloud.google.com/run) to view:

- Request metrics (latency, requests/sec)
- Error rates
- Instance count
- Resource utilization

## Troubleshooting

### Model Not Loading

Check logs for errors:
```bash
gcloud run services logs read arxiv-api --region $REGION | grep -i error
```

Common issues:

1. **GCS mount not working**: Ensure the bucket exists and Cloud Run service account has `storage.objectViewer` role
2. **Out of memory**: Increase `--memory` allocation
3. **Timeout on cold start**: Transformer models take 10-30s to load; increase `--timeout`

### Permission Denied on GCS

Grant the Cloud Run service account access to your bucket:

```bash
export SERVICE_ACCOUNT=$(gcloud run services describe arxiv-api \
  --region $REGION --format='value(spec.template.spec.serviceAccountName)')

gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.objectViewer"
```

## Cost Optimization

1. **Scale to zero**: Use `--min-instances 0` (default) to avoid charges when idle
2. **Right-size resources**: Start with 2GB memory, increase only if needed
3. **Use CPU-only**: GPU instances are expensive; CPU is sufficient for inference
4. **Set concurrency**: Higher concurrency = fewer instances = lower cost

## CI/CD with Cloud Build (Optional)

For automated builds in CI/CD pipelines, you can use Cloud Build instead of building locally. The project includes a `cloudbuild.yaml`:

```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'dockerfiles/api.dockerfile'
      - '-t'
      - 'europe-west1-docker.pkg.dev/$PROJECT_ID/mlops-repo/arxiv-api:latest'
      - '.'

  # Push the container image to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'europe-west1-docker.pkg.dev/$PROJECT_ID/mlops-repo/arxiv-api:latest'

images:
  - 'europe-west1-docker.pkg.dev/$PROJECT_ID/mlops-repo/arxiv-api:latest'

timeout: '1200s'
```

A `.gcloudignore` file excludes large files from the Cloud Build context:
```
.git/
.venv/
*.pt
*.pth
data/
/models/    # Only top-level, not src/arxiv_classifier/models/
```

Note: `uv.lock` is **not** excluded - it's needed for reproducible builds.

### Manual Cloud Build

```bash
gcloud builds submit --config=cloudbuild.yaml .
```

### Automated Builds on Push

Set up a trigger to build automatically when pushing to main:

```bash
gcloud builds triggers create github \
  --repo-name=<your-repo> \
  --repo-owner=<your-github-username> \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```
