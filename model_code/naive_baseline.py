import json

import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from patient_model.dataset import TextDataset
from model_code.evaluation import get_patient_model_metrics
from model_code.train_text_model import CustomDataCollatorForSeq2Seq

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU


def get_baseline_results(split):
    ''' Load baseline results '''
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2-0.5B-Instruct-bnb-4bit")
    dataset_eval = TextDataset(tokenizer=tokenizer, split=split, max_input_len=4000, max_output_len=1000, predict=True, weighted_sampling=False, dataset_name='24h_los', predict_all=False)

    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    eval_data_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['stay_id', 'hour_idx'])
    dataloader_eval = DataLoader(
        dataset_eval,
        batch_size=1,
        collate_fn=eval_data_collator,
        num_workers=0,
        pin_memory=True,
    )
    generated_texts = []
    original_texts = []
    stay_ids = []
    hour_idxs = []
    for i, batch in tqdm(enumerate(dataloader_eval)):
        labels = batch['labels']
        labels = [[token for token in seq if token != -100] for seq in labels]
        stay_id = batch['stay_id']
        hour_idx = batch['hour_idx']
        stay_ids.extend(stay_id)
        hour_idxs.extend(hour_idx)
        original_texts.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
        generated_texts.append(None)

    metrics_event = get_patient_model_metrics(generated_texts, original_texts, stay_ids, hour_idxs, baseline=True, mean_maj=False)
    # save baseline metrics
    with open(f"baseline_metrics_{split}_MAE_newmetrics_5000.json", "w") as f:
        json.dump(metrics_event, f)

if __name__ == '__main__':
    # get_baseline_results(split='val')
    get_baseline_results(split='test')