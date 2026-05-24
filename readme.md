# Setup

This repository extends vLLM prefix caching with multiple cache eviction policies, implements a scheduler for dynamically switching between those policies, and provides benchmark scripts for comparing cache hit rate and serving throughput across conversational workloads. LPC can improve cache hit rate, but on small models its predictor and embedding overhead can reduce actual serving throughput; the scheduler is designed to keep LPC only when its hit-rate gain is large enough to justify that overhead.

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
- `scheduler`: warmup-based dynamic selector over LPC (`ml`), `lru`, `rrip`,
  and `fifo`.

## Scheduler Workflow

In the code, the LPC policy is named `ml`.

1. The scheduler initially sets `current_policy = ml`.
2. During the first `scheduler_warmup = 100` seconds:
   - The real cache uses LPC.
   - The predictor is allowed to compute `prob_has_next`.
   - The scheduler maintains three metadata-only shadow tables to simulate
     `lru`, `rrip`, and `fifo`.
   - The LPC hit rate is measured using the real cache hit rate.
3. After the warmup period ends:
   - The scheduler compares the real LPC hit rate with the three shadow hit
     rates.
   - It selects one final policy and freezes that choice.
   - If the LPC hit rate is greater than the best shadow hit rate plus the
     threshold, it chooses LPC. The threshold offsets LPC's predictor overhead.
   - Otherwise, it chooses the best-performing traditional eviction policy
     among `lru`, `rrip`, and `fifo`.
4. After final selection:
   - If the final policy is not LPC, new requests are no longer enqueued to the
     predictor.
   - If the final policy is LPC, the predictor continues to run.

## Policy Switching Cost

The scheduler is designed to avoid expensive cache-state rebuilds during policy
selection. It does not maintain four real KV caches. During warmup, the real
cache uses LPC, while `lru`, `rrip`, and `fifo` are simulated with lightweight
metadata-only shadow tables keyed by prefix block hash.

When warmup ends, the scheduler performs a single policy switch and freezes the
choice. The existing KV cache contents are not flushed or rebuilt. Existing
cached blocks keep their current metadata, and new or updated blocks are tagged
with the selected policy as they re-enter the evictor. This makes the switch
gradual rather than a full cache reset.

As a result, switching cost is mainly:

- Maintaining three metadata-only shadow tables during warmup.
- Computing the final hit-rate comparison once.
- Updating policy metadata for blocks as they are added or refreshed after the
  switch.

If the selected final policy is not LPC, the system stops enqueuing new cache
hints to the predictor, so the ongoing embedding/prediction overhead is removed
after warmup. The tradeoff is that the cache may contain blocks admitted under
the warmup LPC policy for some time, so the post-switch behavior converges
gradually as requests continue.

Default scheduler settings:

- Warmup: `100` seconds.
- Small-model ML threshold: `0.10`.
- Large-model ML threshold: `0.05`.
- Models up to and including `14B` are treated as small models.
- Initial policy: `ml`.
- Minimum event warning threshold: `0`.
- Shadow-table observe stride: `4`.

`--min-events` is a guard for sample size during warmup. It counts sampled
prefix-block access observations, not requests. The scheduler still finalizes
after warmup; if the count is below the threshold, it prints a warning because
the selected policy may be less stable.

`--observe-stride` controls shadow-table sampling during warmup. The default
`4` means the scheduler updates the warmup hit-rate statistics once every four
prefix-block access observations. Increasing it reduces warmup overhead, while
decreasing it makes the shadow hit-rate estimate more exact.

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
python run_scheduler.py --warmup 100
python run_scheduler.py --observe-stride 8
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

## Example Results

The following results were collected on `Qwen/Qwen2.5-7B-Instruct` with cache
size `8000` on one H100 GPU:

| benchmark | policy | hit_ratio | request_throughput | output_throughput | total_token_throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| sharegpt | fifo | 0.333792 | 2.836372 | 847.605603 | 1204.963509 |
| sharegpt | lru | 0.334573 | 2.840481 | 848.422010 | 1205.845568 |
| sharegpt | ml/LPC | 0.406039 | 2.793798 | 837.231959 | 1189.284579 |
| sharegpt | rrip | 0.320504 | 2.837320 | 848.041088 | 1205.432825 |
| chatbot | fifo | 0.441553 | 10.535377 | 1640.248933 | 2030.162041 |
| chatbot | lru | 0.446812 | 10.533364 | 1639.518445 | 2029.388466 |
| chatbot | ml/LPC | 0.446987 | 10.520410 | 1637.553964 | 2026.891773 |
| chatbot | rrip | 0.448729 | 10.537930 | 1640.603080 | 2030.594933 |
| lmsys | fifo | 0.388348 | 4.400751 | 776.207398 | 1053.311897 |
| lmsys | lru | 0.418865 | 4.424570 | 771.011086 | 1049.713360 |
| lmsys | ml/LPC | 0.463312 | 4.419360 | 769.977454 | 1048.207036 |
| lmsys | rrip | 0.431042 | 4.426383 | 769.400127 | 1048.162597 |

These runs show the core motivation for the scheduler: LPC often improves
`hit_ratio`, but for this 7B model it does not necessarily improve throughput
because predictor overhead can dominate the saved cache misses.

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
