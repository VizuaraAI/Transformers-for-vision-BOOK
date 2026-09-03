import pytest
import torch

from vision_bench.checkpointing import (
    append_jsonl,
    capture_rng_state,
    load_checkpoint,
    load_jsonl,
    restore_rng_state,
    save_checkpoint,
    write_jsonl,
)


def test_jsonl_helpers_replace_and_append(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_jsonl(path, [{"epoch": 1}])
    append_jsonl(path, {"epoch": 2})
    assert load_jsonl(path) == [{"epoch": 1}, {"epoch": 2}]
    write_jsonl(path, [{"epoch": 1}])
    assert load_jsonl(path) == [{"epoch": 1}]


def test_restore_rng_state_accepts_non_byte_tensor_metadata() -> None:
    torch.manual_seed(123)
    state = capture_rng_state()
    expected = torch.rand(4)
    state["torch"] = state["torch"].to(dtype=torch.int64)

    restore_rng_state(state)

    assert torch.equal(torch.rand(4), expected)


def test_load_checkpoint_normalizes_rng_and_generator_tensors(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    rng_state = capture_rng_state()
    rng_state["torch"] = rng_state["torch"].to(dtype=torch.int64)
    rng_state["cuda"] = [torch.tensor([1, 2], dtype=torch.int64)]
    rng_state["mps"] = torch.tensor([3, 4], dtype=torch.int64)
    generator_state = torch.Generator().get_state().to(dtype=torch.int64)
    save_checkpoint(
        path,
        {"rng_state": rng_state, "data_generator_state": generator_state},
    )

    checkpoint = load_checkpoint(path)

    assert checkpoint["rng_state"]["torch"].device.type == "cpu"
    assert checkpoint["rng_state"]["torch"].dtype == torch.uint8
    assert checkpoint["rng_state"]["cuda"][0].dtype == torch.uint8
    assert checkpoint["rng_state"]["mps"].dtype == torch.uint8
    assert checkpoint["data_generator_state"].device.type == "cpu"
    assert checkpoint["data_generator_state"].dtype == torch.uint8


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_load_checkpoint_keeps_rng_on_cpu_after_accelerator_mapping(tmp_path) -> None:
    path = tmp_path / "accelerator-checkpoint.pt"
    save_checkpoint(
        path,
        {
            "model_state": {"weight": torch.ones(2)},
            "rng_state": capture_rng_state(),
            "data_generator_state": torch.Generator().get_state(),
        },
    )

    checkpoint = load_checkpoint(path, map_location=torch.device("mps"))

    assert checkpoint["model_state"]["weight"].device.type == "mps"
    assert checkpoint["rng_state"]["torch"].device.type == "cpu"
    assert checkpoint["rng_state"]["torch"].dtype == torch.uint8
    assert checkpoint["data_generator_state"].device.type == "cpu"
