## Commands for training different Pathway Model Variants:

### Train summerizer model:
```
python -m model_code.train_text_model --config "configs/train_summarizer_model.yaml"
```
This will save checkpoints at outputs/train_summarizer_model/

### Train pathway model - text only 24h:
```
python -m model_code.train_text_model --config "configs/train_pathway_24h_textonly.yaml"
```

### Train pathway model - summary only:
```
python -m model_code.train_with_summ_embs --config "configs/train_pathway_24h_summonly.yaml"
```
This will save checkpoints at outputs/train_pathway_24h_summonly/ 

### Train pathway model - text + summary:
- Before starting set checkpoint path to summonly model in the config file.
```
python -m model_code.train_with_summ_embs --config "configs/train_pathway_24h_text_summ.yaml"
```

## Commands for evaluating next-timestep prediction:

In all validation configs "val_split" is set to "val", can be changed to "test" for testing on the test set.

### Evaluate pathway model - text only 24h:
```
python -m model_code.train_text_model --config "configs/val_pathway_24h_textonly.yaml"
```

### Evaluate pathway model - summary only:
```
python -m model_code.train_with_summ_embs --config "configs/val_pathway_24h_summonly.yaml"
```

### Evaluate pathway model - text + summary:
```
python -m model_code.train_with_summ_embs --config "configs/val_pathway_24h_text_summ.yaml"
```

## Command for evaluating on Longitudinal Simulation Tasks:

```
python -u -m patient_model.downstream_task_eval --model_path <CHECKPOINT_PATH> --task <TASK_NAME> --split "test"
```
- <CHECKPOINT_PATH> should be set to a checkpoint such as outputs/<MODEL_NAME>/checkpoint_\<NUMBER>
- <TASK_NAME> can be set to the following values: "ED_ADM", "ED_ICD", "ED_TS", "Discharge_ICD", "HOSP_TS", "ICU_LOS", "ICU_Mort", "ICU_TS"
- the "_TS" (time series) tasks will at the same time calculate results for all development tasks for the respective clinical unit.
- for mixed models set `--use_summ` flag
- for summary-only models set `--summ_only` and `--use_summ` flags