import pytest
import torch

from vision_bench.benchmark import benchmark_model, summarize_latencies
from vision_bench.config import BenchmarkConfig


def test_latency_summary_uses_batch_throughput() -> None:
    summary = summarize_latencies([10.0, 20.0, 30.0], batch_size=4)
    assert summary["latency_mean_ms"] == pytest.approx(20.0)
    assert summary["latency_median_ms"] == pytest.approx(20.0)
    assert summary["latency_p90_ms"] == pytest.approx(28.0)
    assert summary["throughput_images_per_second"] == pytest.approx(200.0)


def test_cpu_benchmark_completes_fp32_and_skips_fp16() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    config = BenchmarkConfig(
        precisions=("fp32", "fp16"),
        batch_sizes=(1,),
        warmup_iterations=1,
        timed_iterations=2,
        repeats=1,
    )
    records = benchmark_model(model, "tiny", config, input_size=4, device=torch.device("cpu"))
    assert [record["status"] for record in records] == ["complete", "skipped"]
    assert records[0]["samples"] == 2
    assert records[0]["throughput_images_per_second"] > 0
