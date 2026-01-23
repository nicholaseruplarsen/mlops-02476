# Prometheus Metrics

## Input Text Size (`arxiv_api_input_text_size_chars`)
```
Count: 445 requests
Sum:   78,534 characters
Avg:   176.5 characters per request
```

## Request Metrics (`arxiv_api_requests_total`)
```
Total Predict Requests: 445
Total Latency Sum:      2955.6s
Average Latency:        6.64s
```

## Inference Metrics (`arxiv_api_inference_seconds`)
```
Total Inferences:       445
Total Inference Time:   2076.8s
Average Inference:      4.67s
Success Rate:           100%
```

## Latency Distribution (predict)
| Bucket | Count | % |
|--------|-------|---|
| ≤ 1.0s | 8 | 1.8% |
| ≤ 2.5s | 46 | 10.3% |
| ≤ 5.0s | 197 | 44.3% |
| ≤ 10.0s | 365 | 82.0% |
| > 10.0s | 80 | 18.0% |

## System Metrics
```
CPU Usage:    49.9%
Memory:       1.19 GB
```
