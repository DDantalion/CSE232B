# Setup

This repository extends vLLM prefix caching with multiple cache eviction policies, implements a scheduler for dynamically switching between those policies, and provides benchmark scripts for comparing cache hit rate and serving throughput across conversational workloads.

## Cloud Machine Recommendation
**We recommend using a cloud machine with high-performance GPUs for running these experiments.** We use **[Hyperstack H100](https://console.hyperstack.cloud/deploy-virtual-machine)** for optimal performance. Other cloud options include:
- AWS EC2 (Deep Learning Base OSS Nvidia Driver GPU AMI - Ubuntu 24.04)
- Google Cloud Platform with A100/H100 GPUs
- Lambda Labs
- Paperspace

Our experiments were run on one H100 GPU with:

```text
Driver Version: 570.195.03
CUDA Version: 12.8
```

## Quick Setup on Cloud Machine
Run the setup script from `server.sh` to automatically install dependencies and configure the environment:

```bash
# Run the server.sh setup script
wget https://raw.githubusercontent.com/DDantalion/CSE232B/main/server.sh
sudo su
bash server.sh
```

## Local Environment Setup

After cloning the repository, create the benchmark environment and install the
local vLLM package:

```bash
cd CSE232B
bash vllm_cache_bench/setup_env.sh
```

The setup script creates the `vllm-cuda121` conda environment, installs the
precompiled vLLM wheel in editable mode, downloads the ShareGPT dataset, checks
CUDA availability, and then starts a scheduler benchmark run.

For subsequent runs:

```bash
conda activate vllm-cuda121
cd vllm_cache_bench
```

## Model

The benchmark model is configured in `vllm_cache_bench/constants_nips.py`:

```python
MODEL = os.getenv("VLLM_BENCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
```

To use a different model without editing code:

```bash
export VLLM_BENCH_MODEL="Qwen/Qwen2.5-14B-Instruct"
```

The scheduler infers the model size from this `MODEL` string, so no separate
model-size argument is required.

## Eviction Policies

The implementation supports these eviction policies:

- `ml`: predictor-based LPC policy.
- `lru`: least recently used.
- `rrip`: RRIP with an aging loop.
- `fifo`: first in, first out.
- `scheduler`: warmup-based dynamic selector over `ml`, `lru`, `rrip`, and
  `fifo`.

The scheduler starts with real `ml` for the warmup period. During warmup it
measures the real ML hit rate and maintains metadata-only shadow tables for
`lru`, `rrip`, and `fifo`. After warmup, it freezes to the best policy. If the
selected policy is not `ml`, new cache hints no longer enter the ML predictor
queue, avoiding continued embedding/prediction overhead.

Default scheduler settings:

- Warmup: `200` seconds.
- Small-model ML threshold: `0.10`.
- Large-model ML threshold: `0.05`.
- Models up to and including `14B` are treated as small models.
- Initial policy: `ml`.
- Minimum event warning threshold: `0`.

`--min-events` is a guard for sample size during warmup. It counts prefix-block
access observations, not requests. The scheduler still finalizes after warmup;
if the count is below the threshold, it prints a warning because the selected
policy may be less stable.

## Running Benchmarks

Run the baseline policy sweep from `run_nips.py`:

```bash
cd vllm_cache_bench
python run_nips.py
```

By default this runs `ml`, `lru`, `rrip`, and `fifo` clients on:

```text
sharegpt, lmsys, chatbot
```

Run only the scheduler version:

```bash
cd vllm_cache_bench
python run_scheduler.py
```

By default `run_scheduler.py` also runs:

```text
sharegpt, lmsys, chatbot
```

Useful scheduler options:

```bash
python run_scheduler.py --warmup 200
python run_scheduler.py --datasets sharegpt
python run_scheduler.py --datasets sharegpt,lmsys,chatbot --sizes 8000 --scales 1
python run_scheduler.py --small-threshold 0.10 --large-threshold 0.05
```

Dataset-specific request settings, checkpoint paths, dataset paths, and
`time_limit` values are inherited from `run_nips.py`. For example, ShareGPT uses:

```python
c["request_rate"] = 0.01
c["max_active_conversations"] = 200
c["checkpoint"] = f"{HOME}/vllm/benchmarks/checkpoints_sharegpt_20/sharegpt_epoch19_metric_0_5427.pt"
c["dataset_file"] = f"{HOME}/vllm_cache_bench/ShareGPT_V3_unfiltered_cleaned_split.json"
c["time_limit"] = 1200
```

## Dataset Access

ShareGPT is downloaded by `setup_env.sh`.

The LMSYS and Chatbot Arena datasets are hosted on Hugging Face. If access fails
with a gated dataset error, authenticate first:

```bash
huggingface-cli login
```

or export a token:

```bash
export HF_TOKEN="..."
```

## Results

Raw client JSON files are written under:

```text
vllm_cache_bench/results/<model>/<benchmark>-<tag>/client_logs/
```

Examples:

```text
results/Qwen2.5-7B-Instruct/sharegpt-size++/client_logs/8000_ml.json
results/Qwen2.5-7B-Instruct/sharegpt-scheduler++/client_logs/8000_scheduler.json
```

Experiment summaries are written as:

```text
results/<model>/exp_<benchmark>.json
```

## Collecting Metrics

Collect the four main metrics for every benchmark/policy:

```bash
cd vllm_cache_bench
python collect_policy_metrics.py --results-dir results
```

The collected metrics are:

- `hit_ratio`
- `request_throughput`
- `output_throughput`
- `total_token_throughput`

Filter one benchmark:

```bash
python collect_policy_metrics.py --results-dir results/Qwen2.5-7B-Instruct --benchmark sharegpt
```

Write a CSV:

```bash
python collect_policy_metrics.py \
  --results-dir results/Qwen2.5-7B-Instruct \
  --output policy_metrics.csv
```

If multiple runs exist for the same benchmark/policy, the default mode picks
the latest record with the most complete metrics. To average runs:

```bash
python collect_policy_metrics.py --results-dir results --mode mean
```
