"""
SeqForensics Dataset Validation Script

Validates:
1. Metadata structure and completeness
2. Expected number of videos
3. Duplicate metadata entries
4. Source and sequence distribution
5. Physical existence of every video
6. Video readability
7. Frame decoding
8. Frame count consistency with metadata
9. Valid FPS and resolution
10. Empty/corrupted video detection
11. Sequence balance
12. Position-wise manipulation balance

Outputs:
- dataset_validation_report.csv
- dataset_validation_summary.txt
"""

import cv2
import pandas as pd
from pathlib import Path
from collections import Counter

# ============================================================
# PATH CONFIGURATION
# Same folder structure as dataset generation script
# ============================================================

ROOT = (
    Path(__file__).resolve().parent.parent
    / "dataset"
    / "archive"
    / "FaceForensics++_C23"
)

VIDEO_DIR = ROOT / "generated_videos_60k"
METADATA_CSV = ROOT / "labels_60k" / "_metadata_index.csv"

REPORT_CSV = ROOT / "labels_60k" / "dataset_validation_report.csv"
SUMMARY_TXT = ROOT / "labels_60k" / "dataset_validation_summary.txt"

EXPECTED_SOURCES = 100
EXPECTED_SEQUENCES = 60
EXPECTED_VIDEOS = EXPECTED_SOURCES * EXPECTED_SEQUENCES

MANIPULATIONS = [
    "FaceSwap",
    "FaceShifter",
    "DeepFake",
    "NeuralTextures",
    "Face2Face"
]


# ============================================================
# START VALIDATION
# ============================================================

print("=" * 70)
print("SEQFORENSICS DATASET VALIDATION")
print("=" * 70)

print("\nVideo directory:")
print(VIDEO_DIR)

print("\nMetadata file:")
print(METADATA_CSV)


# ============================================================
# CHECK REQUIRED FILES/FOLDERS
# ============================================================

if not VIDEO_DIR.exists():
    raise FileNotFoundError(
        f"Generated video directory not found:\n{VIDEO_DIR}"
    )

if not METADATA_CSV.exists():
    raise FileNotFoundError(
        f"Metadata CSV not found:\n{METADATA_CSV}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_CSV)

required_columns = [
    "video_name",
    "source_video",
    "sequence",
    "num_frames"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required metadata columns: {missing_columns}"
    )


# ============================================================
# 1. STRUCTURAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("1. STRUCTURAL VALIDATION")
print("=" * 70)

total_records = len(df)

unique_videos = df["video_name"].nunique()

unique_sources = df["source_video"].nunique()

unique_sequences = df["sequence"].nunique()

duplicate_rows = df.duplicated().sum()

duplicate_video_names = df["video_name"].duplicated().sum()

missing_values = df[
    required_columns
].isnull().sum().sum()


print(f"Metadata records:       {total_records}")
print(f"Unique videos:          {unique_videos}")
print(f"Unique source videos:   {unique_sources}")
print(f"Unique sequences:       {unique_sequences}")
print(f"Duplicate rows:         {duplicate_rows}")
print(f"Duplicate video names:  {duplicate_video_names}")
print(f"Missing metadata:       {missing_values}")


# ============================================================
# 2. SOURCE DISTRIBUTION VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("2. SOURCE DISTRIBUTION")
print("=" * 70)

source_counts = df.groupby(
    "source_video"
).size()

sources_with_60 = (
    source_counts == EXPECTED_SEQUENCES
).sum()

invalid_sources = source_counts[
    source_counts != EXPECTED_SEQUENCES
]


print(
    f"Sources containing exactly 60 sequences: "
    f"{sources_with_60}/{unique_sources}"
)

if len(invalid_sources) == 0:
    print("Source distribution: PASS")
else:
    print("Source distribution: FAIL")

    print("\nSources with incorrect sequence count:")

    print(invalid_sources)


# ============================================================
# 3. SEQUENCE DISTRIBUTION VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("3. SEQUENCE DISTRIBUTION")
print("=" * 70)

sequence_counts = df.groupby(
    "sequence"
).size()

expected_per_sequence = EXPECTED_SOURCES

invalid_sequences = sequence_counts[
    sequence_counts != expected_per_sequence
]

print(
    f"Unique sequences found: {unique_sequences}"
)

print(
    f"Expected samples per sequence: "
    f"{expected_per_sequence}"
)

if len(invalid_sequences) == 0:
    print("Sequence distribution: PASS")
else:
    print("Sequence distribution: FAIL")

    print("\nSequences with incorrect counts:")

    print(invalid_sequences)


# ============================================================
# 4. MANIPULATION BALANCE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("4. MANIPULATION BALANCE")
print("=" * 70)

overall_counter = Counter()

position_counters = [
    Counter(),
    Counter(),
    Counter()
]

sequence_parse_errors = []

for sequence in df["sequence"]:

    parts = [
        part.strip()
        for part in str(sequence).split("->")
    ]

    if len(parts) != 3:

        sequence_parse_errors.append(
            sequence
        )

        continue

    for position, manipulation in enumerate(parts):

        overall_counter[
            manipulation
        ] += 1

        position_counters[
            position
        ][manipulation] += 1


print("\nOverall manipulation occurrences:")

for manipulation in MANIPULATIONS:

    print(
        f"{manipulation:20s}: "
        f"{overall_counter[manipulation]}"
    )


print("\nPosition-wise occurrences:")

for manipulation in MANIPULATIONS:

    print(
        f"{manipulation:20s} "
        f"P1={position_counters[0][manipulation]:4d} "
        f"P2={position_counters[1][manipulation]:4d} "
        f"P3={position_counters[2][manipulation]:4d}"
    )


# ============================================================
# 5. PHYSICAL VIDEO VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5. PHYSICAL VIDEO VALIDATION")
print("=" * 70)

validation_results = []

total = len(df)

for index, row in df.iterrows():

    video_name = str(
        row["video_name"]
    )

    expected_frames = int(
        row["num_frames"]
    )

    video_path = (
        VIDEO_DIR
        / video_name
    )

    file_exists = (
        video_path.exists()
    )

    file_size_bytes = (
        video_path.stat().st_size
        if file_exists
        else 0
    )

    opens_successfully = False

    first_frame_decoded = False

    middle_frame_decoded = False

    last_frame_decoded = False

    actual_frames = 0

    fps = 0.0

    width = 0

    height = 0

    frame_count_match = False

    status = "FAIL"

    failure_reason = []


    if not file_exists:

        failure_reason.append(
            "FILE_MISSING"
        )

    elif file_size_bytes == 0:

        failure_reason.append(
            "EMPTY_FILE"
        )

    else:

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            failure_reason.append(
                "CANNOT_OPEN"
            )

        else:

            opens_successfully = True

            actual_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            fps = float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )


            # ----------------------------------------
            # Decode first frame
            # ----------------------------------------

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            ret, frame = cap.read()

            first_frame_decoded = (
                ret
                and frame is not None
            )


            # ----------------------------------------
            # Decode middle frame
            # ----------------------------------------

            if actual_frames > 0:

                middle_index = (
                    actual_frames // 2
                )

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    middle_index
                )

                ret, frame = cap.read()

                middle_frame_decoded = (
                    ret
                    and frame is not None
                )


            # ----------------------------------------
            # Decode last frame
            # ----------------------------------------

            if actual_frames > 0:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    actual_frames - 1
                )

                ret, frame = cap.read()

                last_frame_decoded = (
                    ret
                    and frame is not None
                )


            cap.release()


            # ----------------------------------------
            # Validate properties
            # ----------------------------------------

            frame_count_match = (
                actual_frames
                == expected_frames
            )


            if not first_frame_decoded:

                failure_reason.append(
                    "FIRST_FRAME_DECODE_FAILED"
                )


            if not middle_frame_decoded:

                failure_reason.append(
                    "MIDDLE_FRAME_DECODE_FAILED"
                )


            if not last_frame_decoded:

                failure_reason.append(
                    "LAST_FRAME_DECODE_FAILED"
                )


            if actual_frames <= 0:

                failure_reason.append(
                    "ZERO_FRAMES"
                )


            if fps <= 0:

                failure_reason.append(
                    "INVALID_FPS"
                )


            if width <= 0 or height <= 0:

                failure_reason.append(
                    "INVALID_RESOLUTION"
                )


            if not frame_count_match:

                failure_reason.append(
                    "FRAME_COUNT_MISMATCH"
                )


            if len(failure_reason) == 0:

                status = "PASS"


    validation_results.append({

        "video_name":
            video_name,

        "source_video":
            row["source_video"],

        "sequence":
            row["sequence"],

        "file_exists":
            file_exists,

        "file_size_bytes":
            file_size_bytes,

        "opens_successfully":
            opens_successfully,

        "first_frame_decoded":
            first_frame_decoded,

        "middle_frame_decoded":
            middle_frame_decoded,

        "last_frame_decoded":
            last_frame_decoded,

        "expected_frames":
            expected_frames,

        "actual_frames":
            actual_frames,

        "frame_count_match":
            frame_count_match,

        "fps":
            fps,

        "width":
            width,

        "height":
            height,

        "validation_status":
            status,

        "failure_reason":
            ";".join(
                failure_reason
            )

    })


    # Progress display

    if (
        (index + 1) % 100 == 0
        or index + 1 == total
    ):

        print(
            f"Validated "
            f"{index + 1}/"
            f"{total} videos"
        )


# ============================================================
# SAVE DETAILED REPORT
# ============================================================

results_df = pd.DataFrame(
    validation_results
)

results_df.to_csv(
    REPORT_CSV,
    index=False
)


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

passed = (
    results_df[
        "validation_status"
    ]
    == "PASS"
).sum()

failed = (
    results_df[
        "validation_status"
    ]
    == "FAIL"
).sum()

files_found = (
    results_df[
        "file_exists"
    ]
).sum()

opened = (
    results_df[
        "opens_successfully"
    ]
).sum()

frame_matches = (
    results_df[
        "frame_count_match"
    ]
).sum()

first_decoded = (
    results_df[
        "first_frame_decoded"
    ]
).sum()

middle_decoded = (
    results_df[
        "middle_frame_decoded"
    ]
).sum()

last_decoded = (
    results_df[
        "last_frame_decoded"
    ]
).sum()


# ============================================================
# FRAME STATISTICS
# ============================================================

frame_mean = df[
    "num_frames"
].mean()

frame_min = df[
    "num_frames"
].min()

frame_max = df[
    "num_frames"
].max()

frame_std = df[
    "num_frames"
].std()


# ============================================================
# OVERALL VALIDATION DECISION
# ============================================================

structural_pass = (

    total_records
    == EXPECTED_VIDEOS

    and unique_videos
    == EXPECTED_VIDEOS

    and unique_sources
    == EXPECTED_SOURCES

    and unique_sequences
    == EXPECTED_SEQUENCES

    and duplicate_rows
    == 0

    and duplicate_video_names
    == 0

    and missing_values
    == 0

    and len(invalid_sources)
    == 0

    and len(invalid_sequences)
    == 0

    and len(sequence_parse_errors)
    == 0
)


file_validation_pass = (

    passed
    == EXPECTED_VIDEOS
)


overall_pass = (

    structural_pass
    and file_validation_pass
)


# ============================================================
# PRINT FINAL REPORT
# ============================================================

summary = f"""
============================================================
SEQFORENSICS DATASET VALIDATION SUMMARY
============================================================

STRUCTURAL VALIDATION

Expected videos:              {EXPECTED_VIDEOS}
Metadata records:             {total_records}
Unique generated videos:      {unique_videos}
Unique source videos:         {unique_sources}
Unique sequences:             {unique_sequences}

Duplicate rows:               {duplicate_rows}
Duplicate filenames:          {duplicate_video_names}
Missing metadata values:      {missing_values}

Sources with 60 sequences:    {sources_with_60}/{unique_sources}

------------------------------------------------------------

PHYSICAL VIDEO VALIDATION

Expected videos:              {total}
Files found:                  {files_found}
Videos opened successfully:   {opened}

First frames decoded:         {first_decoded}
Middle frames decoded:        {middle_decoded}
Last frames decoded:          {last_decoded}

Frame counts matched:         {frame_matches}

Videos PASSED:                {passed}
Videos FAILED:                {failed}

------------------------------------------------------------

FRAME STATISTICS

Average frames:               {frame_mean:.2f}
Minimum frames:               {frame_min}
Maximum frames:               {frame_max}
Standard deviation:           {frame_std:.2f}

------------------------------------------------------------

Structural validation:
{"PASS" if structural_pass else "FAIL"}

File-level validation:
{"PASS" if file_validation_pass else "FAIL"}

FINAL DATASET VALIDATION:
{"PASS" if overall_pass else "FAIL"}

============================================================
"""

print(summary)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary
    )


print(
    "\nDetailed validation report saved to:"
)

print(
    REPORT_CSV
)

print(
    "\nValidation summary saved to:"
)

print(
    SUMMARY_TXT
)


# ============================================================
# PRINT FAILURES IF ANY
# ============================================================

if failed > 0:

    print(
        "\nFAILED VIDEOS:"
    )

    failed_df = results_df[

        results_df[
            "validation_status"
        ]
        == "FAIL"

    ]

    print(

        failed_df[

            [
                "video_name",
                "failure_reason"
            ]

        ].to_string(

            index=False

        )

    )

else:

    print(
        "\nAll generated videos passed "
        "structural and file-level validation."
    )
