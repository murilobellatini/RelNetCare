# RelNetCare Dataset Directory

This README outlines the structure and content of the 'data' directory in the RelNetCare repository.

> Note: Use the `src.processing.DialogREDatasetResampler` class to recreate the datasets from the raw data.

| Directory | Description |
| :-------- | :---------- |
| `raw` | Contains the unprocessed DialogRE dataset (download from DialogRE repo [here](https://github.com/nlpdata/dialogre)). |
| `processed` | Contains processed versions of the DialogRE dataset, as described below. |

## Processed Datasets

| Dataset | Description |
| :------ | :---------- |
| `dialog-re-binary` | A binary version of DialogRE, recoded to express presence/absence of a relation. |
| `dialog-re-ternary` | A ternary version of DialogRE, recoded to express "no relation", "unanswerable", or "with relation". |
| `dialog-re-ternary-oversampled` | An oversampled version of the ternary DialogRE dataset. |
| `dialog-re-ternary-undersampled` | An undersampled version of the ternary DialogRE dataset. |
| `dialog-re-with-no-relation` | Includes instances where no relation exists, unlike the original dataset. |

Datasets are named as `dialog-re-<descriptor>`, where `<descriptor>` indicates a unique characteristic or processing step.
