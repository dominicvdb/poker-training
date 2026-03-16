"""Tests for src/data_loader.py."""

import pytest
from datasets import DatasetDict


@pytest.fixture(scope="session")
def dataset():
    from src.data_loader import load_pokerbench
    return load_pokerbench()


def test_returns_dataset_dict(dataset):
    assert isinstance(dataset, DatasetDict)


def test_train_and_test_splits_exist(dataset):
    assert "train" in dataset
    assert "test" in dataset


def test_train_row_count(dataset):
    assert len(dataset["train"]) == 563_200


def test_test_row_count(dataset):
    assert len(dataset["test"]) == 11_000


def test_column_names(dataset):
    for split in ("train", "test"):
        assert dataset[split].column_names == ["instruction", "output"]


def test_no_null_instructions(dataset):
    sample = dataset["train"].select(range(1000))
    assert all(row["instruction"] is not None and row["instruction"] != "" for row in sample)


def test_no_null_outputs(dataset):
    sample = dataset["train"].select(range(1000))
    assert all(row["output"] is not None and row["output"] != "" for row in sample)


def test_example_row_structure(dataset):
    row = dataset["train"][0]
    assert "instruction" in row
    assert "output" in row
    assert isinstance(row["instruction"], str)
    assert isinstance(row["output"], str)
