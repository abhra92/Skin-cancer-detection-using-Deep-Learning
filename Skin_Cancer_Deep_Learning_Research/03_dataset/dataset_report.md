# Dataset audit report

- Audit time (UTC): `2026-10-06T09:32:21+00:00`
- Requested dataset root: `/home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000`
- HAM10000 metadata CSV: `NOT AVAILABLE`
- Metadata rows: `NOT AVAILABLE`
- Candidate image files: `0`
- Metadata rows linked to a readable image: `0`
- Status: **BLOCKED — NO EXPERIMENTAL TRAINING**

## Checks

| Check | Value | Status | Details |
|---|---:|---|---|
| data_root | /home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000 | MISSING |  |
| metadata_csv_count | 0 | MISSING |  |
| image_file_count | 0 | MISSING |  |
| dataset_status | HAM10000 assets were not found under the requested root | BLOCKED |  |
| dataset_status | not suitable for training until blockers are resolved | BLOCKED | Dataset root does not exist: /home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000 |

## Interpretation

No class frequencies or dataset-level statistics are inferred when the metadata/images are absent. The reason(s) training is blocked are:
- Dataset root does not exist: /home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000

A bibliography CSV is not treated as dataset metadata. Image classes, lesion counts, corruption rates, and duplicate rates are reported only when calculated from the actual image/metadata files.
