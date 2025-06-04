## Generating a fine-tuning dataset

- This can only be performed after data generation for Longitudinal Simulation Tasks has been completed (see Readme_Dataset_Generation.md)
- For the ed_admission and ed_icd tasks, need to generate summary embeddings first:
  - TODO how does weighting.py play into this? and extract_summ_embs_finetune.py 
  - TODO in which order do I need to run the scripts? When run generate_dataset_finetune?

First create weighted training sample paths using:
```
python -m model_code.weighting --task <TASK_NAME>
```
- `<TASK_NAME>` can be set to "ed_admission", "ed_icd", "mort", "los", "hosp_icd"

Then generate training and validation datasets using:
```
python -m patient_model.generate_dataset_finetune --task <TASK_NAME> --mode <MODE>
```
- `<TASK_NAME>` can be set to "ed_admission", "ed_icd", "mort", "los", "hosp_icd"
- `<MODE>` can be set to "path" or "outcome"

Lastly, generate fitting summary embeddings for ed_admission and ed_icd tasks:
```
python -m model_code.extract_summary_embs --task <TASK_NAME>
```
- `<TASK_NAME>` can be set to "ed_admission", "ed_icd"
- for all other tasks, we can reuse the already generated summary embeddings from the main dataset generation.


## Steps for fine-tuning for outcome prediction tasks

```
python -m model_code.train_with_summ_embs_finetune --config <FINETUNE_CONFIG>
```
- These are the available configs:
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_ED_ADM_direct.yaml`
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_ED_ADM_pathway.yaml`
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_ED_icd_pathway.yaml`
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_HOSP_icd_pathway.yaml`
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_los_direct.yaml`
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_los_path.yaml`
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_mort_direct.yaml`
  - `configs/finetune_config_qwen_summ_input_last_int_mixed_24h_mort_pathway.yaml`

## Evaluation of fine-tuned models

```
python -u -m patient_model.downstream_task_eval --model_path <CHECKPOINT_PATH> --task <TASK_NAME> --split "test" --use_summ
```
- <CHECKPOINT_PATH> should be set to a checkpoint such as outputs/<MODEL_NAME>/checkpoint_\<NUMBER>
- <TASK_NAME> can be set to the following values: "ED_ADM", "ED_ICD", "ED_TS", "Discharge_ICD", "HOSP_TS", "ICU_LOS", "ICU_Mort", "ICU_TS"
- the "_TS" (time series) tasks will at the same time calculate results for all development tasks for the respective clinical unit.
