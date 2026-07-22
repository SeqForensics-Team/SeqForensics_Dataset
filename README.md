# SeqForensics

**A benchmark dataset and baseline for detecting the *order* of sequential deepfake manipulations.**

Most deepfake datasets label a clip with a single manipulation type. In practice, forensically
"interesting" videos are rarely touched once — they pass through a **chain** of edits (e.g. a face
swap, followed by a re-enactment pass, followed by a neural-texture cleanup) before they are
re-uploaded. SeqForensics is built to study exactly that: given a video, can a model recover **which**
manipulations were applied **and in what order**?

---

## What's in this repo

This repository supports **reproducibility of the SeqForensics dataset** — the generation pipeline,
the validation pipeline, and the metadata schema. It does **not** host the paper, slides, poster, or
project video, and it does not redistribute raw FaceForensics++ source video.

| | |
|---|---|
| **Base data source** | [FaceForensics++](https://github.com/ondyari/FaceForensics) |
| **Generated clips** | 6,000 |
| **Manipulation types** | 5 |
| **Ordered sequences per clip** | 3 manipulations applied in sequence |
| **Distinct sequence classes** | 60 (all ordered permutations of 3-out-of-5 manipulation types) |
| **Frames per source clip** | ~300–400 |
| **Split strategy** | Source-leakage-free (train / val / test) |

---

## Manipulation vocabulary

Every clip is built by chaining **3** of the following **5** manipulation types, drawn from the
standard FaceForensics++ manipulation set:

| Code | Manipulation |
|:---:|---|
| `DF`  | Deepfakes |
| `F2F` | Face2Face |
| `FS`  | FaceSwap |
| `FSh` | FaceShifter |
| `NT`  | NeuralTextures |

A clip's label is stored as an ordered, pipe-separated string, e.g. `FS\|FSh\|DF`, which maps to a
fixed integer sequence via a single vocabulary defined once in code (no re-derivation, no risk of the
generation and validation scripts disagreeing on label order).

Since order matters, the label space is **ordered permutations**, not combinations:

```
P(5, 3) = 5 × 4 × 3 = 60 distinct sequence classes
```

This is what makes SeqForensics harder than "what manipulation is this?" — a model has to reason
about **temporal/compositional structure**, not just detect an artifact.

---

## Dataset construction

1. **Source selection** — clean, unmanipulated source clips are pulled from FaceForensics++.
2. **Sequence assignment** — each source clip is assigned one or more of the 60 valid ordered
   3-manipulation sequences.
3. **Chained manipulation + stitching** — each manipulation in the sequence is applied on top of the
   previous stage's output, then the three stages are stitched into a single output clip.
4. **Manifest recording** — every generated clip is logged with its source ID, output path, applied
   sequence, and frame count.
5. **Validation pass** — every one of the 6,000 generated clips is checked for corruption, playback
   integrity, frame-count consistency, and label correctness (see [`validation/`](validation/)).
6. **Split assignment** — splits are built **per source ID**, not per clip. All sequence variants
   generated from the same source video are forced into the same split, so a model can't cheat by
   memorizing source-video identity instead of learning manipulation structure.

---

## Repository structure

```text
SeqForensics/
├── README.md
├── LICENSE
├── requirements.txt
├── code/
│   ├── generate_dataset.py     # builds the 6,000 sequential clips from FF++ sources
│   └── validate_dataset.py     # integrity + label validation over the full dataset
├── metadata/
│   └── sample_metadata.csv     # example rows (schema only — not the full manifest)
├── validation/
│   └── dataset_validation_summary.txt
└── docs/
    └── dataset_structure.md    # metadata schema, naming conventions, folder layout
```

> **Note:** `code/generate_dataset.py`, `code/validate_dataset.py`, and
> `metadata/sample_metadata.csv` are added separately alongside this scaffold — this drop contains
> the documentation, licensing, and validation-summary files.

---

## Metadata schema

Each row in the manifest describes one generated clip:

| Field | Description |
|---|---|
| `video_name` | Filename of the generated sequential clip |
| `source_video` | ID/filename of the original FaceForensics++ source video |
| `sequence` | Pipe-separated manipulation order, e.g. `FS\|FSh\|DF` |
| `num_frames` | Frame count of the generated clip |

Full field-by-field documentation lives in [`docs/dataset_structure.md`](docs/dataset_structure.md).

---

## Validation summary

Every one of the 6,000 generated clips was checked for file integrity, decodability, expected frame
count, and correct sequence-label formatting before being included in the final manifest. See
[`validation/dataset_validation_summary.txt`](validation/dataset_validation_summary.txt) for the
full report.

---

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the sequential dataset from your local FaceForensics++ copy
python code/generate_dataset.py \
    --ff-root /path/to/FaceForensics++ \
    --out-dir dataset/generated_videos \
    --metadata-out metadata/manifest.csv

# 3. Validate the generated dataset
python code/validate_dataset.py \
    --metadata metadata/manifest.csv \
    --video-root dataset/generated_videos \
    --report-out validation/dataset_validation_report.csv
```

---

## Baseline task

The manifest produced by this pipeline is directly consumable by an order-prediction baseline: a
shared-weight CNN frame encoder + temporal self-attention model with one classification head per
sequence position, trained with a source-leakage-free split. This dataset repo does not host that
training code — it exists to make the **data** reproducible.

---

## Data availability & licensing

- **This repository's code** (generation + validation scripts, docs) is released under the license
  in [`LICENSE`](LICENSE).
- **Raw FaceForensics++ source video is not redistributed here.** Access to FaceForensics++ requires
  agreeing to its own terms — see the [official repository](https://github.com/ondyari/FaceForensics)
  before generating or sharing derived clips.
- Generated sequential clips are derivative of FaceForensics++ source material; check FF++'s
  redistribution terms before publishing the full generated dataset.

## Citation

If you use this dataset construction pipeline, please cite this repository and the original
FaceForensics++ paper (Rössler et al., ICCV 2019).
