"""Tests for train_single.py progress parsing and param normalization."""

import json
import re
import sys
from pathlib import Path

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.train_single import (
    _camel_to_snake,
    _normalize_params,
    _TQDM_RE,
    _TQDM_RE_MINIMAL,
    _EPOCH_RE,
    TrainingProgress,
)


# --- camelCase / snake_case tests ---

def test_camel_to_snake_simple():
    assert _camel_to_snake("loraName") == "lora_name"

def test_camel_to_snake_multi_word():
    assert _camel_to_snake("networkDim") == "network_dim"

def test_camel_to_snake_already_snake():
    assert _camel_to_snake("epochs") == "epochs"

def test_camel_to_snake_consecutive_caps():
    assert _camel_to_snake("vaeChunkSize") == "vae_chunk_size"

def test_normalize_params():
    result = _normalize_params({
        "loraName": "test",
        "networkDim": 32,
        "trainingImages": "/path/to/images",
        "epochs": 10,
    })
    assert result == {
        "lora_name": "test",
        "network_dim": 32,
        "training_images": "/path/to/images",
        "epochs": 10,
    }

def test_normalize_params_nested_keys():
    result = _normalize_params({
        "maxSteps": 1000,
        "outputDir": "/output",
        "batchSize": 4,
    })
    assert "max_steps" in result
    assert "output_dir" in result
    assert "batch_size" in result


# --- TQDM regex tests ---

def test_tqdm_regex_full():
    line = "steps:  45%|████▌     | 450/1000 [02:15<02:45, 3.35it/s, avr_loss=0.123]"
    m = _TQDM_RE.search(line)
    assert m is not None
    assert m.group(1) == "45"
    assert m.group(2) == "450"
    assert m.group(3) == "1000"
    assert m.group(4) == "0.123"

def test_tqdm_regex_minimal():
    line = "steps:  10%|█         | 100/1000 [00:30<04:30, 3.00it/s]"
    m = _TQDM_RE_MINIMAL.search(line)
    assert m is not None
    assert m.group(1) == "10"
    assert m.group(2) == "100"
    assert m.group(3) == "1000"

def test_tqdm_regex_no_match():
    line = "some random log line"
    m = _TQDM_RE.search(line)
    assert m is None
    m = _TQDM_RE_MINIMAL.search(line)
    assert m is None


# --- Epoch regex tests ---

def test_epoch_regex():
    line = "\nepoch 3/10\n"
    m = _EPOCH_RE.search(line)
    assert m is not None
    assert m.group(1) == "3"
    assert m.group(2) == "10"

def test_epoch_regex_no_match():
    line = "no epoch here"
    m = _EPOCH_RE.search(line)
    assert m is None


# --- TrainingProgress tests ---

def test_progress_initial():
    p = TrainingProgress()
    d = p.to_dict()
    assert d["status"] == "running"
    assert d["current_step"] == 0
    assert d["total_steps"] == 0
    assert d["avg_loss"] is None

def test_progress_update_tqdm():
    p = TrainingProgress()
    p.update_from_tqdm(45, 450, 1000, 0.123)
    d = p.to_dict()
    assert d["current_step"] == 450
    assert d["total_steps"] == 1000
    assert d["avg_loss"] == 0.123

def test_progress_update_tqdm_no_loss():
    p = TrainingProgress()
    p.update_from_tqdm(10, 100, 1000)
    d = p.to_dict()
    assert d["current_step"] == 100
    assert d["total_steps"] == 1000
    assert d["avg_loss"] is None

def test_progress_update_epoch():
    p = TrainingProgress()
    p.update_epoch(3, 10)
    d = p.to_dict()
    assert d["current_epoch"] == 3
    assert d["total_epochs"] == 10

def test_progress_mark_completed():
    p = TrainingProgress()
    p.mark_completed(0)
    d = p.to_dict()
    assert d["status"] == "completed"
    assert d["exit_code"] == 0

def test_progress_mark_failed():
    p = TrainingProgress()
    p.mark_failed(1, "OOM error")
    d = p.to_dict()
    assert d["status"] == "failed"
    assert d["exit_code"] == 1
    assert d["error"] == "OOM error"
