# noinspection PyUnresolvedReferences
from unsloth_helpers.patch_unsloth import *

import gzip
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import transformers
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing as mp

from mimic_iv_extraction.paths import mimiciv_path
from model_code.train_text_model import CustomDataCollatorForSeq2Seq
from patient_model.dataset import SummaryDataset, SectionIterableDataset, create_sample_to_section_dict, create_sample_to_section_dict_ed_adm
from unsloth import FastLanguageModel

import argparse

max_seq_length = 34000  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
SUMMARY_LEN = 8
CONV_TOKENS_AFTER_SUMMARY = 5

def load_summ_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    model.base_model.model.config.use_custom_attn = True

    new_special_tokens = {'additional_special_tokens': ['<SUMMARY>']}
    tokenizer.add_special_tokens(new_special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # load summary model
    file_path = "outputs/summary_full_1M_8token_sumonly_clean/checkpoint-108000/adapter_model.safetensors"
    lora_weights = load_file(file_path)
    # change all keys from .weight to .default.weight
    lora_weights = {k.replace(".weight", ".default.weight"): v for k, v in lora_weights.items()}
    model.load_state_dict(lora_weights, strict=False)

    return model, tokenizer


def extract_embs_ed_vitals(model, tokenizer, split):
    model.eval()
    dataset = SummaryDataset(tokenizer=tokenizer, split=split,
                             max_input_len=4000,
                             max_output_len=1000, model_name="Qwen2-0.5B-Instruct-bnb-4bit",
                             custom_attn_mask=True, predict=True, add_gen_prompt_for_predict=False, predict_all=True,
                             summary_len=SUMMARY_LEN)

    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    eval_data_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['stay_id', 'hour_idx'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=eval_data_collator)

    start = time.time()
    for idx, batch in tqdm(enumerate(dataloader)):
        inputs = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        stay_ids = batch['stay_id']
        hour_idxs = batch['hour_idx']

        with torch.inference_mode():
            outputs = model(inputs, attention_mask=attention_mask, return_dict=False, use_cache=False, output_hidden_states=True)

            hidden_states = outputs[1]
            last_hidden_states = hidden_states[-1]
            last_hidden_states.clamp_(-128, 127)

            # max_pool_hidden_states = torch.max(torch.stack(hidden_states), dim=0).values
            for idx, hidden_state in enumerate(last_hidden_states):
                input_len = attention_mask[idx].sum().item() - 2  # -2 because of <im_end> token
                extracted_summary_tokens = hidden_state[input_len - SUMMARY_LEN:input_len]
                # extracted_summary_tokens_max = max_pool_hidden_states[idx, input_len-SUMMARY_LEN:input_len]
                # extracted_summary_tokens_max = extracted_summary_tokens_max.detach()

                # save as int to save space
                with gzip.open(f'{mimiciv_path}ed_vital_summaries_last_int/{stay_ids[idx]}_{hour_idxs[idx]}.pt.gz', 'wb') as f:
                    torch.save(extracted_summary_tokens.type(torch.int8).cpu(), f)

    print(f"Time taken: {time.time() - start}")


executor = ThreadPoolExecutor(max_workers=1)


def save_embs(collected_embs):
    for key, value in collected_embs.items():
        stay_id, hour_idx = key
        with gzip.open(f'{mimiciv_path}all_summaries_last_int/{stay_id}_{hour_idx}.pt.gz', 'wb') as f:
            torch.save(collected_embs[key], f)


def extract_embs_all(model, tokenizer, split, dataset_idx):
    model.eval()
    dataset = SectionIterableDataset(tokenizer=tokenizer, split=split, weighted_sampling=True, max_input_len=5000, max_output_len=None,
                                     model_name="Qwen2-0.5B-Instruct-bnb-4bit", custom_attn_mask=True, dataset_name="all_steps", predict=True,
                                     add_gen_prompt_for_predict=False, dataset_idx=dataset_idx, summary_len=SUMMARY_LEN)

    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    eval_data_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['stay_id', 'hour_idx', 'block_idx'])

    batch_size = 8
    # prefetch_factor = 2000 // batch_size assures that the next 100 patients are already prepared while the last are processed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=eval_data_collator, pin_memory=True, prefetch_factor=20000//batch_size)
    collected_embs = defaultdict(dict)
    current_block_idx = None

    processed_parts = 0

    start = time.time()

    # section_lens = []

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        processed_parts += batch_size
        inputs = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        stay_ids = batch['stay_id']
        timepoint_idxs = batch['hour_idx']
        block_idxs = batch['block_idx']

        # section_lens.extend([elem.item() for elem in list(attention_mask.sum(dim=1))])

        with torch.inference_mode():
            outputs = model(inputs, attention_mask=attention_mask, return_dict=True, use_cache=True, output_hidden_states=False)
            last_hidden_states = outputs['hidden_states']
            input_lens = attention_mask.sum(dim=1) - 2
            last_hidden_states = torch.stack([
                hidden_state[input_len - SUMMARY_LEN:input_len]
                for hidden_state, input_len in zip(last_hidden_states, input_lens)
            ])

            last_hidden_states.clamp_(-128, 127)
            last_hidden_states = last_hidden_states.type(torch.int8).cpu()

            for idx, extracted_summary_tokens in enumerate(last_hidden_states):
                stay_id, icu_stay, section, part_idx = stay_ids[idx]

                if current_block_idx != block_idxs[idx]:
                    # save previous block
                    if current_block_idx is not None:
                        executor.submit(save_embs, collected_embs)
                    current_block_idx = block_idxs[idx]
                    collected_embs = defaultdict(dict)
                    collected_embs[(stay_id, timepoint_idxs[idx])][(stay_id, timepoint_idxs[idx], icu_stay, section, part_idx)] = extracted_summary_tokens

                else:
                    # write into current block dict, add key -value pair to dictionary at stay_id
                    collected_embs[(stay_id, timepoint_idxs[idx])][(stay_id, timepoint_idxs[idx], icu_stay, section, part_idx)] = extracted_summary_tokens

        del outputs, last_hidden_states, inputs, attention_mask, extracted_summary_tokens

    # save last block
    executor.submit(save_embs, collected_embs)

    print(f"Time taken: {time.time() - start}")
    print(f"Processed parts: {processed_parts}")


def extract_embs_sample(model, samples, tokenizer, batch_size=8, eval_data_collator=None):

    collected_embs = {}

    # Create batches using the eval_data_collator
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        batch_samples = [sample[1] for sample in batch_samples]

        # Prepare batched inputs using the data collator
        batch = eval_data_collator(batch_samples)
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        stay_ids = batch['stay_id']
        timepoint_idxs = batch['hour_idx']

        with torch.inference_mode():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True  # Required if accessing 'hidden_states'
            )
            last_hidden_states = outputs['hidden_states'][-1]  # Extract the last layer
            input_lens = attention_mask.sum(dim=1) - 2  # Adjust for special tokens
            last_hidden_states = torch.stack([
                hidden_state[input_len - SUMMARY_LEN:input_len]
                for hidden_state, input_len in zip(last_hidden_states, input_lens)
            ])

            last_hidden_states.clamp_(-128, 127)
            last_hidden_states = last_hidden_states.type(torch.int8)

            # Collect embeddings
            for idx, extracted_summary_tokens in enumerate(last_hidden_states):
                stay_id, icu_stay, section, part_idx = stay_ids[idx]

                collected_embs[(stay_id, timepoint_idxs[idx], icu_stay, section, part_idx)] = extracted_summary_tokens

    return collected_embs

if __name__ == '__main__':

    sections = {'ED_Stay': [('Emergency Department Stay',)],
                'HS_General__HA_PatientLocation__HA_CareTaker': [('Hospital Stay', 'General'), ('Hospital Admission', 'Patient Location'),
                                                                 ('Hospital Admission', 'Care Taker')],
                'HS_OutpatientMeasurements': [('Hospital Stay', 'Outpatient Measurements')],
                'HS_LabResults': [('Hospital Stay', 'Lab Results')],
                'HS_RadiologyNotes': [('Hospital Stay', 'Radiology Notes')],
                'HS_Prescriptions': [('Hospital Stay', 'Prescriptions')],
                'HS_Procedures': [('Hospital Stay', 'Procedures')],
                'HS_Microbiology': [('Hospital Stay', 'Microbiology Growth Results')],
                'ICU_Medication': [('ICU', 'Medication')],
                'ICU_Output': [('ICU', 'Output')],
                'ICU_Procedures': [('ICU', 'Procedures')],
                'ICU_CE_VitalSigns': [('ICU', 'Chart Events', 'RoutineVitalSigns')],
                'ICU_CE_AdmHistory': [('ICU', 'Chart Events', 'AdmHistory_FHPA')],
                'ICU_CE_MDProgressNote': [('ICU', 'Chart Events', 'MDProgressNote')],
                'ICU_CE_Respiratory': [('ICU', 'Chart Events', 'Respiratory')],
                'ICU_CE_Pulmonary': [('ICU', 'Chart Events', 'Pulmonary')],
                'ICU_CE_SkinAssessment': [('ICU', 'Chart Events', 'Skin-Assessment')],
                'ICU_CE_SkinImpairment': [('ICU', 'Chart Events', 'Skin-Impairment')],
                'ICU_CE_SkinIncisions': [('ICU', 'Chart Events', 'Skin-Incisions')],
                'ICU_CE_CardioPulses': [('ICU', 'Chart Events', 'Cardiovascular(Pulses)')],
                'ICU_CE_Neurological': [('ICU', 'Chart Events', 'Neurological')],
                'ICU_CE_Hemodynamics': [('ICU', 'Chart Events', 'Hemodynamics')],
                'ICU_CE_Alarms': [('ICU', 'Chart Events', 'Alarms')],
                'ICU_CE_PainSedation': [('ICU', 'Chart Events', 'Pain_Sedation')],
                'ICU_CE_GIGU': [('ICU', 'Chart Events', 'GI_GU')],
                'ICU_CE_Cardio': [('ICU', 'Chart Events', 'Cardiovascular')],
                'ICU_CE_CardioPacerData': [('ICU', 'Chart Events', 'Cardiovascular(PacerData)')],
                'ICU_CE_IABP': [('ICU', 'Chart Events', 'IABP')],
                'ICU_CE_Dialysis': [('ICU', 'Chart Events', 'Dialysis')],
                'ICU_CE_Toxicology': [('ICU', 'Chart Events', 'Toxicology')],
                'ICU_CE_NICOM': [('ICU', 'Chart Events', 'NICOM')],
                }

    parser = argparse.ArgumentParser(description="Extract embeddings for a given dataset index.")
    parser.add_argument('--dataset_idx', type=int, required=True,
                        help="Index of the dataset to process (0, 1, 2, or 3), -1 for all at once")
    args = parser.parse_args()
    dataset_idx = args.dataset_idx

    print("Creating sample to section dict")
    create_sample_to_section_dict(split="train", sections=sections, num_workers=100)
    create_sample_to_section_dict(split="val", sections=sections, num_workers=100)
    create_sample_to_section_dict(split="test", sections=sections, num_workers=100)

    print("Extracting dataset idx ", dataset_idx)
    model, tokenizer = load_summ_model()

    extract_embs_all(model, tokenizer, "train", dataset_idx=dataset_idx)
    extract_embs_all(model, tokenizer, "val", dataset_idx=dataset_idx)
    extract_embs_all(model, tokenizer, "test", dataset_idx=dataset_idx)