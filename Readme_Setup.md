## Setup

Create new conda environment:
```
conda create --name ehr2path python=3.10
conda activate ehr2path
```

Install requirements:
```
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-deps "xformers<0.0.27" "trl<0.9.0"
```

Build the Cython extensions:
```
python setup.py build_ext --inplace
```

## Prepare data files and model weights

- Download precomputed_data and place all contents in your mimiciv_path folder (e.g. /data/mimiciv/).
- This data includes:
  - `train_event_counts.json`: counts of events in training data; used to calculate weights for training samples
  - `percentile_dict.json` and `min_max_dict.json`: used for normalization of features at validation and test time
  - `train_paths.json`, `val_paths.json`, `test_paths.json`: paths/sample_ids for training, validation, and test splits: each path has the format '/stay_<patient_id>_<adm_id/ed_stay_id>.json', ed_stay_id is only used for ED admissions without hospital adm_id
  - `train_weighting_sample_timepoint_paths_1Mio.json.gz`, `val_noweighting_sample_timepoint_paths.json.gz`, `test_noweighting_sample_timepoint_paths.json.gz`: paths for each sample (on timepoint level) in the training, validation, and test splits we used for pathway prediction, format: '/stay_<patient_id>_<adm_id/ed_stay_id>.json_<timepoint_idx>'

- Download pre-trained model weights and place them in the `outputs` folder.
- Available model checkpoints include:
  - finetune_<TASK>_direct/pathway: fine-tuned models for simulation tasks (e.g., ED_ADM, ICU_LOS, etc.)
  - summary_full_1M_8token_sumonly_clean: pre-trained summarization model - used for summerizing patient stays into embeddings
  - summary_input_8_last_int_only_summ_clean2: EHR2Path-summary - only uses summary input of entire stay
  - train_qwen_5000_v2_weight_24h_1M_clean: EHR2Path-text - only uses textual input of last 24 hours
  - summary_input_8_last_int_mixed_24h_clean2_DropAugment_curriculum: EHR2Path-text+summary - uses both summary and text input of last 24 hours