"""Download and cache the PokerBench dataset from HuggingFace."""

from pathlib import Path

from datasets import load_dataset, DatasetDict

DATASET_ID = "RZ412/PokerBench"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def load_pokerbench(cache_dir: Path = CACHE_DIR) -> DatasetDict:
    """Return the PokerBench train and test splits.

    Downloads on first call and reads from cache_dir on subsequent calls.
    Each row has 'instruction' (scenario text) and 'output' (optimal action).
    """
    dataset = load_dataset(DATASET_ID, cache_dir=str(cache_dir))
    return dataset
