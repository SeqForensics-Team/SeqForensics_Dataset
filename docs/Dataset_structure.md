# Dataset Structure

This document describes the on-disk layout and metadata schema produced by
`code/generate_dataset.py` and consumed by `code/validate_dataset.py`.

## Folder layout

```text
dataset/
└── generated_videos/
    ├── <source_id>__<sequence>__001.mp4
    ├── <source_id>__<sequence>__002.mp4
    └── ...
metadata/
└── manifest.csv
```

Each generated `.mp4` corresponds to exactly one row in `metadata/manifest.csv`.

## Metadata fields (`manifest.csv`)

| Field | Type | Description |
|---|---|---|
| `video_name` | string | Filename of the generated clip (relative to `generated_videos/`) |
| `video_path` | string | Full/relative path used to locate the file on disk |
| `source_id` | string | Identifier of the original FaceForensics++ source video. Used to build source-leakage-free train/val/test splits — every clip sharing a `source_id` is guaranteed to land in the same split. |
| `source_video` | string | Original FaceForensics++ source filename |
| `edit_keys` / `sequence` | string | Pipe-separated, ordered manipulation codes applied to this clip, e.g. `FS\|FSh\|DF` |
| `num_frames` | int | Frame count of the generated clip |

## Manipulation vocabulary

The label vocabulary is defined once, in a single place, and both the generation
and validation code read from that same definition — this avoids the class of
bug where the generator and validator silently disagree on label order.

| Code | Manipulation | ID |
|:---:|---|:---:|
| `DF`  | Deepfakes | 0 |
| `F2F` | Face2Face | 1 |
| `FS`  | FaceSwap | 2 |
| `FSh` | FaceShifter | 3 |
| `NT`  | NeuralTextures | 4 |

- **Vocabulary size:** 5
- **Sequence length:** 3 manipulations per clip (fixed — no padding, no variable-length clips)
- **Distinct sequence classes:** `P(5, 3) = 60` ordered permutations

## Naming convention

```
<source_id>__<manip1>-<manip2>-<manip3>__<variant_index>.mp4
```

Example: `sid00187__FS-FSh-DF__001.mp4` is a clip derived from source video
`sid00187`, manipulated with FaceSwap, then FaceShifter, then Deepfakes.

## Splitting logic

Splits are computed over **unique `source_id` values**, not over individual
clips:

1. Collect the sorted, deduplicated set of `source_id`s.
2. Shuffle deterministically with a fixed seed.
3. Assign ~70% of source IDs to `train`, ~15% to `val`, ~15% to `test`.
4. Every clip generated from a given source ID inherits that source's split.

This guarantees no source video's frames appear in more than one split, which
prevents a model from learning to recognize the base video's identity instead
of the manipulation sequence applied to it.

## Reading a label

`edit_keys` (or `sequence`) is parsed by splitting on `|` and mapping each code
through the vocabulary table above, e.g.:

```
"FS|FSh|DF"  ->  [2, 3, 0]
```

Every row must contain **exactly 3** codes — rows with more or fewer are
treated as a data bug and rejected during validation rather than silently
padded or truncated.
