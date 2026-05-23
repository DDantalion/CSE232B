import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


POLICIES = ("ml", "lru", "rrip", "fifo")
BENCHMARKS = ("sharegpt", "lmsys", "chatbot", "tay")
METRICS = (
    "hit_ratio",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_records(path: str) -> Iterable[tuple[str, Dict[str, Any]]]:
    data = load_json(path)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield path, item
    elif isinstance(data, dict):
        yield path, data


def iter_result_files(results_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(results_dir):
        for name in files:
            if name.endswith(".json") and not name.endswith(".pytorch.json"):
                yield os.path.join(root, name)


def infer_policy(path: str, record: Dict[str, Any]) -> Optional[str]:
    algorithm = record.get("algorithm")
    if isinstance(algorithm, str):
        for policy in POLICIES:
            if algorithm == policy or algorithm.startswith(f"{policy}-"):
                return policy

    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for part in stem.split("_"):
        if part in POLICIES:
            return part
    return None


def infer_benchmark(path: str, record: Dict[str, Any]) -> str:
    dataset_name = record.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name:
        return dataset_name

    result_file = record.get("result_file")
    candidates = []
    if isinstance(result_file, str):
        candidates.append(result_file.lower())

    norm_path = path.replace("\\", "/").lower()
    candidates.append(norm_path)
    candidates.append(os.path.splitext(os.path.basename(path))[0].lower())

    for candidate in candidates:
        base = os.path.basename(candidate)
        if base.startswith("exp_"):
            name = os.path.splitext(base)[0]
            return name.removeprefix("exp_")
        for benchmark in BENCHMARKS:
            if benchmark in candidate:
                return benchmark

    parent = os.path.basename(os.path.dirname(os.path.dirname(norm_path)))
    if parent:
        return parent.split("-")[0]
    return "unknown"


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    metrics = {metric: to_float(record.get(metric)) for metric in METRICS}

    # run_nips.py summary files store a time series called hit_ratios.
    # Use the final value as the run's hit ratio when the raw client JSON is
    # not available.
    if metrics["hit_ratio"] is None:
        hit_ratios = record.get("hit_ratios")
        if isinstance(hit_ratios, list) and hit_ratios:
            metrics["hit_ratio"] = to_float(hit_ratios[-1])

    return metrics


def has_any_metric(metrics: Dict[str, Optional[float]]) -> bool:
    return any(value is not None for value in metrics.values())


def summarize(records: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    if mode == "latest":
        return max(records, key=lambda r: r["mtime"])

    summary = {
        "benchmark": records[0]["benchmark"],
        "policy": records[0]["policy"],
        "source": f"{len(records)} runs",
    }
    for metric in METRICS:
        values = [
            record[metric] for record in records
            if record.get(metric) is not None
        ]
        summary[metric] = mean(values) if values else None
    return summary


def collect(results_dir: str, mode: str) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for path in iter_result_files(results_dir):
        try:
            records = iter_json_records(path)
            for source, record in records:
                policy = infer_policy(source, record)
                if policy is None:
                    continue
                benchmark = infer_benchmark(source, record)

                metrics = extract_metrics(record)
                if not has_any_metric(metrics):
                    continue

                grouped[(benchmark, policy)].append({
                    "benchmark": benchmark,
                    "policy": policy,
                    "source": source,
                    "mtime": os.path.getmtime(source),
                    **metrics,
                })
        except (OSError, json.JSONDecodeError):
            continue

    rows = []
    for benchmark, policy in sorted(grouped.keys()):
        rows.append(summarize(grouped[(benchmark, policy)], mode))
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ("benchmark", "policy", *METRICS, "source")
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            value = row.get(header)
            text = "" if value is None else str(value)
            widths[header] = max(widths[header], len(text))

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = "" if value is None else str(value)
            values.append(text.ljust(widths[header]))
        print("  ".join(values))


def write_csv(rows: List[Dict[str, Any]], output: str) -> None:
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=("benchmark", "policy", *METRICS, "source"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-policy cache benchmark metrics.")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing benchmark JSON files.")
    parser.add_argument(
        "--mode",
        choices=("latest", "mean"),
        default="latest",
        help="How to summarize multiple runs per policy.")
    parser.add_argument(
        "--benchmark",
        default="",
        help="Optional benchmark filter, e.g. sharegpt, lmsys, chatbot.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV output path.")
    args = parser.parse_args()

    rows = collect(args.results_dir, args.mode)
    if args.benchmark:
        rows = [row for row in rows if row.get("benchmark") == args.benchmark]
    print_table(rows)

    if args.output:
        write_csv(rows, args.output)


if __name__ == "__main__":
    main()
