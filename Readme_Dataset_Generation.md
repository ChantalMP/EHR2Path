## Steps for generating a patient-timepoint dataset

### Step 1: extract MIMIC-IV Patient Data (1-2 days)
- File: mimic_iv_extraction/extract_patient_data.py
- Input: MIMIC-IV database as csv files - specified in mimic_iv_extraction/paths.py
- Output: one folder per patient admission with all their data (saved to all_admissions/) + stay_to_ids_dict.json (stay id -> ed, adm and icu info)

```
python -m mimic_iv_extraction.extract_patient_data
```

### Step 2: extract Text Datasets for LLM Training

- File: patient_model/generate_dataset.py
- Input: resamples_json for train and val
- Output: saves train and val samples as yaml files ({patient_folder}/stay_{stay_id}.json)
- in generate_patient_data_helper_from_timepoints need to specify: RESTRICT_TO_N_HOURS and SUMMARY_LEVEL
- need to regenerate for different model types:
  - text_only: RESTRICT_TO_N_HOURS=24, SUMMARY_LEVEL=1 -> At this point text-only models can be trained - continue for summary models
  ```
  python -m patient_model.generate_dataset --mode text_model
  ```
  - summary model: RESTRICT_TO_N_HOURS=None, SUMMARY_LEVEL=1 -> full data to be embedded into summary embeddings later
  ```
  python -m patient_model.generate_dataset --mode summary_model
  ```
  - summary model LOS: only LOS indicator -> only needed for summary-only model variant
  ```
  python -m patient_model.generate_dataset --mode summary_model_LOS
  ```

### Step 3: prepare section dataset for training summary model
- File: patient_model/dataset.py
- first call create_section_to_sample_dict() -> for all resampled samples checks which sections are present
- then call sample_summary_dataset() -> creates dataset with sections as labels (mimiciv_path + f"section_samples_for_summary_{split}.json"), which is loaded in SummaryDataset
- summary data is based on same sub-sampled timepoints as text data, but currently we sample only 500.000 samples for training
  ```
  python -m patient_model.dataset 
  ```
-> with this output summarizer models can be trained

### Step 4: extract summary embeddings for Summary-based LLaMA training
- File: model_code/extract_summary_embs.py -> need to have trained summary model first
- Input: SectionIterableDataset on [split]_weighting_sample_timepoint_paths.json.gz -> iterates over all sections of one patient in order
- Output: saves embeddings for each section in a patient stay as json file ({mimiciv_path}all_summaries_last_int/{current_stay_id}_{hour_idx}.pt.gz')
- generated locally -> after generation need to rsync to sol mimic folder
  ```
  python -m model_code.extract_summary_embs --dataset_idx -1
  ```
-> with this output summary-based or mixed models can be trained


## Create datasets for Longitudinal Simulation Tasks
- File: patient_model/downstream_task_datasets.py
- run for all tasks to create the datasets
  ```
  python -m patient_model.downstream_task_datasets --task <TASK_NAME> --split <SPLIT>
  ```
- <TASK_NAME> can be set to the following values: "ED_ADM", "ED_ICD", "ED_TS", "Discharge_ICD", "HOSP_TS", "ICU_LOS", "ICU_Mort", "ICU_TS"
- \<SPLIT> can be set to "val" or "test"