import json
from dataclasses import replace

import pytest

from vision_bench.config import DataConfig
from vision_bench.data import (
    create_manifest,
    load_manifest,
    stratified_indices,
    stratified_limit,
    write_manifest,
)


def _targets(items_per_class: int) -> list[int]:
    return [class_id for class_id in range(100) for _ in range(items_per_class)]


def test_stratified_indices_are_balanced_disjoint_and_deterministic() -> None:
    targets = _targets(5)
    first_train, first_val = stratified_indices(targets, 3, 2, seed=2027)
    second_train, second_val = stratified_indices(targets, 3, 2, seed=2027)
    assert (first_train, first_val) == (second_train, second_val)
    assert len(first_train) == 300
    assert len(first_val) == 200
    assert set(first_train).isdisjoint(first_val)
    assert {targets[index] for index in first_train} == set(range(100))


def test_stratified_test_limit_represents_every_class() -> None:
    targets = _targets(10)
    selected = stratified_limit(targets, limit=250, seed=2027)
    counts = {class_id: 0 for class_id in range(100)}
    for index in selected:
        counts[targets[index]] += 1
    assert len(selected) == 250
    assert set(counts.values()) == {2, 3}


def test_manifest_round_trip_and_checksum_detection(tmp_path) -> None:
    config = DataConfig(train_per_class=3, val_per_class=2, test_limit=200)
    manifest = create_manifest(_targets(5), _targets(3), config)
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert load_manifest(path) == manifest

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_indices"][0] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_manifest(path)


def test_data_config_rejects_invalid_test_limit() -> None:
    with pytest.raises(ValueError, match="test_limit"):
        replace(DataConfig(), test_limit=99).validate()
