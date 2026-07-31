# PLAN: Optional `.tags` Files for Clean LoRA Metadata

## Problem

When Caption-Studio's "For Anima" mode generates captions, the `.txt` file contains:

```
1girl, long hair, blue eyes, forest background. A girl standing in a sunlit forest. The wind gently rustles through the trees.
```

This full caption (booru tags + natural language addition) is excellent for training - the model benefits from the rich descriptions. However, the same text gets embedded as metadata inside the final `.safetensors` LoRA file, making it extremely hard to read since it's a comma/space-separated wall of text with no structure.

## Solution

Caption-Studio now writes **two files** per image in For Anima mode:

| File | Content | Use |
|------|---------|-----|
| `image.txt` | Full caption (tags + NL addition) | Training prompts |
| `image.tags` | Booru tags only | Clean LoRA metadata |

Example for `forest-girl.jpg`:

```
forest-girl.txt:
  1girl, long hair, blue eyes, forest background. A girl standing in a sunlit forest. The wind gently rustles through the trees.

forest-girl.tags:
  1girl, long hair, blue eyes, forest background
```

## How to Use `.tags` Files for Metadata

### Option A: Use `.tags` as the caption source (simplest)

If you want clean metadata and are OK with training on tags-only captions, change the caption extension in your dataset TOML:

```toml
[general]
caption_extension = ".tags"
```

This can be done by modifying `scripts/dataset_toml.py` (change `".txt"` to `".tags"` in the `caption_extension` field) or by adding a CLI flag.

**Trade-off:** Training uses only booru tags (no NL enrichment), but metadata is clean.

### Option B: Keep `.txt` for training, swap to `.tags` for metadata only (recommended)

Train with `.txt` files (full captions) for best results, then regenerate metadata from `.tags` files post-training.

Two sub-approaches:

#### B1: Post-training metadata swap

After training completes, use `merge_captions_to_metadata.py` from sd-scripts to regenerate the metadata JSON from `.tags` files:

```bash
uv run python sd-scripts/finetune/merge_captions_to_metadata.py \
  datasets/my-char/img/ \
  datasets/my-char/out/.work/metadata.json \
  --caption_extension .tags
```

Then re-embed the cleaned metadata into the `.safetensors` file (requires a separate metadata injection tool).

#### B2: Two-pass dataset config (future work)

Extend the trainer to support separate caption sources for training vs. metadata:

```python
# In dataset_toml.py, add:
"caption_extension": ".txt",           # for training
"metadata_caption_extension": ".tags", # for embedded metadata
```

This would require changes to kohya-ss's training script (`anima_train_network.py`) to support a secondary caption source for metadata embedding.

### Option C: Add `--caption-extension` CLI flag (quick win)

Add a simple CLI flag to override the default `.txt` extension:

```bash
uv run python scripts/train.py \
  --type character \
  --dataset datasets/my-char/ \
  --caption-extension .tags \
  --name MyChar
```

Pass it through `cli_args.py` into `dataset_toml.py`'s `generate_dataset_toml()`:

```python
# In dataset_toml.py -> generate_dataset_toml():
def generate_dataset_toml(
    ...
    caption_extension: str = ".txt",  # new param
) -> str:
    ...
    config = {
        "general": {
            "caption_extension": caption_extension,
            ...
        },
    }
```

## Recommended Implementation

**Phase 1 (quick):** Add `--caption-extension` CLI flag (Option C). This lets users train with `.tags` files if they want clean metadata.

**Phase 2 (ideal):** Implement Option B2 - separate caption sources for training and metadata. This requires upstream changes to kohya-ss but gives the best of both worlds.

## Files Changed (Caption-Studio side - DONE)

| File | Change |
|------|--------|
| `src/lib/anima-prompt.ts` | `assembleFinalCaption()`: join with `. ` (period separator) |
| `src/lib/temp-files.ts` | Added `writeTags()` function |
| `src/app/api/caption/for-anima/route.ts` | Write both `.txt` and `.tags` files per image |

## Files to Change (Trainer side - TODO)

| File | Change |
|------|--------|
| `scripts/cli_args.py` | Add `--caption-extension` argument |
| `scripts/dataset_toml.py` | Accept `caption_extension` param in `generate_dataset_toml()` |
| `scripts/train.py` | Pass caption_extension through to dataset_toml |
| `README.md` | Document the new flag and `.tags` file concept |
