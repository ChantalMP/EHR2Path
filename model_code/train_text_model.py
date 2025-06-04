# noinspection PyUnresolvedReferences
from unsloth_helpers.patch_unsloth import *

import argparse
import gc
import time

import torch
import transformers
import wandb
import yaml
from nltk.translate.bleu_score import corpus_bleu
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

# noinspection PyUnresolvedReferences
from mimic_iv_extraction.extract_patient_data import Patient_Visit, Patient_ADM, Patient_ED, Patient_ICU
from model_code.evaluation import get_patient_model_metrics
from patient_model.dataset import TextDataset, SummaryDataset
from unsloth import FastLanguageModel, is_bfloat16_supported

# from retrieve_change_log import extract_changes_from_changelog

max_seq_length = 34000  # Choose any! We auto support RoPE Scaling internally! #was 2048
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

from model_code.unsloth_SFTTrainer_local import SFTTrainer_US
from transformers import TrainingArguments
from transformers.cache_utils import Cache


def _my_get_initial_cache_position(self, input_ids, model_kwargs):
    """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
    past_length = 0
    if model_kwargs.get("past_key_values") is not None:
        cache = model_kwargs["past_key_values"]
        if not isinstance(cache, Cache):
            past_length = cache[0][0].shape[2]
        elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
            past_length = cache.get_seq_length()

    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    else:
        cur_len = input_ids.shape[-1]
    if "position_ids" in model_kwargs:
        model_kwargs["cache_position"] = model_kwargs["position_ids"]
    else:
        model_kwargs["cache_position"] = torch.arange(past_length, cur_len, device=input_ids.device)
    return model_kwargs


def log_metrics(model, eval_dataloader, step, summary_len):
    model.eval()
    FastLanguageModel.for_inference(model)
    generated_texts = []
    original_texts = []
    stay_ids = []
    hour_idxs = []
    for idx, batch in tqdm(enumerate(eval_dataloader)):
        inputs = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        labels = [[token for token in seq if token != -100] for seq in labels]
        attention_mask = batch['attention_mask'].to(model.device)
        stay_id = batch['stay_id']
        hour_idx = batch['hour_idx']
        stay_ids.extend(stay_id)
        hour_idxs.extend(hour_idx)

        if eval_dataloader.dataset.custom_attn_mask:  # This code tests the model output if only the summary tokens are given as input
            SUMMARY_LENGTH = summary_len
            CONV_TOKENS_AFTER_SUMMARY = 5
            inputs_part, last_tokens, attention_mask = inputs[:, :-CONV_TOKENS_AFTER_SUMMARY], inputs[:, -CONV_TOKENS_AFTER_SUMMARY:], attention_mask[
                                                                                                                                       :,
                                                                                                                                       :-CONV_TOKENS_AFTER_SUMMARY]
            with torch.no_grad():
                outputs = model(inputs_part, attention_mask=attention_mask, return_dict=True, use_cache=True)

            summary_tokens = outputs.past_key_values
            # only keep summary tokens of past key values
            new_summary_tokens = []
            for layer_idx, layer in enumerate(summary_tokens):
                new_summary_tokens.append(
                    (summary_tokens[layer_idx][0][:, :, -SUMMARY_LENGTH:, :], summary_tokens[layer_idx][1][:, :, -SUMMARY_LENGTH:, :]))
            summary_tokens = tuple(new_summary_tokens)

            # convert to tuple
            pos_id_curr = torch.tensor([inputs_part.shape[1]]).unsqueeze(dim=0).cuda()
            # add rest of input tokens with only seeing to summary tokens
            with torch.no_grad():
                for i in range(CONV_TOKENS_AFTER_SUMMARY - 1):  # last token is directly given to generate
                    outputs = model(
                        input_ids=last_tokens[:, i:i + 1],
                        past_key_values=summary_tokens,
                        attention_mask=torch.tensor([1]).unsqueeze(dim=0).cuda(),
                        use_cache=True,
                        return_dict=True,
                        position_ids=pos_id_curr
                    )

                    pos_id_curr += 1
                    summary_tokens = outputs.past_key_values

            attention_mask = torch.cat([attention_mask, torch.tensor([1] * 5).unsqueeze(dim=0).cuda()], axis=1)[:,
                             -(SUMMARY_LENGTH + CONV_TOKENS_AFTER_SUMMARY):]
            inputs = torch.cat([inputs_part, last_tokens], axis=1)[:, -(SUMMARY_LENGTH + CONV_TOKENS_AFTER_SUMMARY):]
            model.base_model.model._get_initial_cache_position = _my_get_initial_cache_position.__get__(model.base_model.model)
            generated_ids = model.generate(input_ids=inputs, past_key_values=summary_tokens, attention_mask=attention_mask, max_new_tokens=4000,
                                           cache_implementation=None, position_ids=pos_id_curr)

        else:
            generated_ids = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=4000)
        generated_texts.extend(tokenizer.batch_decode(generated_ids[:, len(inputs[0]):], skip_special_tokens=True))
        original_texts.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))


    # log text in table in wandb
    table_cols = ["Step", "Predicted", "GT"]
    table_data = [[step, generated_texts[i], original_texts[i]] for i in range(min(len(generated_texts), 5))]  # [0, 1, 2, 3, 4, 8, 15]

    b1, b4 = corpus_bleu([[ref] for ref in generated_texts], original_texts, weights=[(1, 0, 0, 0), (0.25, 0.25, 0.25, 0.25)])

    start = time.time()
    patient_model_metrics = get_patient_model_metrics(generated_texts, original_texts, stay_ids, hour_idxs)
    print(f"Time taken for patient model metrics: {time.time() - start}")

    split = eval_dataloader.dataset.split
    mytable = wandb.Table(data=table_data, columns=table_cols)

    # Initialize a dictionary to hold all metrics
    metrics = {}

    # Add the table and BLEU scores
    metrics[f"{split}_output_log"] = mytable
    metrics[f"{split}_bleu1"] = b1
    metrics[f"{split}_bleu4"] = b4

    # Add the global step
    metrics["train/global_step"] = step

    # Add patient model metrics
    for outer_key, value_dict in patient_model_metrics.items():
        for inner_key, value in value_dict.items():
            metrics[f"{split}_{outer_key}_{inner_key}"] = value

    # Log all metrics together
    wandb.log(metrics)  # , step=step

    del mytable
    torch.cuda.empty_cache()
    FastLanguageModel.for_training(model)


# Custom callback for logging
class EvalCallback(transformers.TrainerCallback):
    def __init__(self, eval_dataloader_callback, frequency=1000, summary_len=8):
        self.eval_dataloader_callback = eval_dataloader_callback
        self.frequency = frequency
        self.summary_len = summary_len

    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % self.frequency == 0:
            log_metrics(kwargs['model'], self.eval_dataloader_callback, state.global_step, self.summary_len)


class CustomAttnDataCollatorForSeq2Seq:
    def __init__(self, base_collator, additional_keys=[]):
        self.base_collator = base_collator
        self.additional_keys = additional_keys

    def __call__(self, features):
        # Separate standard model inputs and additional fields
        standard_features = [{k: v for k, v in f.items() if k not in self.additional_keys} for f in features]
        additional_features = {key: [f[key] for f in features] for key in self.additional_keys}

        # Use the base collator for standard inputs
        collated_features = self.base_collator(standard_features)

        # pad attention mask to max length
        attention_masks = additional_features['attention_mask']
        pad_to_len = collated_features['input_ids'].shape[1]
        batch_size = len(attention_masks)
        batch_attention_mask = torch.zeros((batch_size, 1, pad_to_len, pad_to_len), dtype=attention_masks[0].dtype)  # 1 for head dimension

        padding_side = self.base_collator.tokenizer.padding_side if hasattr(self.base_collator,
                                                                            'tokenizer') else self.base_collator.base_collator.tokenizer.padding_side
        if padding_side == 'right':
            for i, mask in enumerate(attention_masks):
                seq_len = mask.size(1)
                # Place the existing mask in the top-left corner
                batch_attention_mask[i, 0, :seq_len, :seq_len] = mask[0]
        else:  # padding_side == 'left'
            for i, mask in enumerate(attention_masks):
                seq_len = mask.size(1)
                # Place the existing mask in the bottom-right corner
                batch_attention_mask[i, 0, -seq_len:, -seq_len:] = mask[0]

        batch_attention_mask = torch.where(batch_attention_mask == 0, torch.tensor(torch.finfo(torch.bfloat16).min), torch.tensor(0.0))

        collated_features.data['attention_mask'] = batch_attention_mask

        return collated_features


class CustomDataCollatorForSeq2Seq:
    def __init__(self, base_collator, additional_keys=[]):
        self.base_collator = base_collator
        self.additional_keys = additional_keys

    def __call__(self, features):
        # Separate standard model inputs and additional fields
        standard_features = [{k: v for k, v in f.items() if k not in self.additional_keys} for f in features]
        additional_features = {key: [f[key] for f in features] for key in self.additional_keys}

        # Use the base collator for standard inputs
        collated_features = self.base_collator(standard_features)

        # Add back the additional fields without modifying them
        collated_features.update(additional_features)

        return collated_features


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Load config for training")
    parser.add_argument('--config', type=str, default="configs/train_config_llama.yaml",
                        help="Path to the config file")
    return parser.parse_args()


class ClearMemoryCallback(transformers.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Clear GPU memory cache
        torch.cuda.empty_cache()
        # Perform garbage collection
        gc.collect()


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()

    # load config name from run parameters
    config = load_config(args.config)

    val_only = config.get("val_only", False)
    val_split = config.get("val_split", "val")

    if config['experiment_name'] != 'evaluation_only':
        # initialize wandb logger with project name and experiment name
        wandb.init(project=config['project_name'], name=config['experiment_name'])
        # save the config to wandb
        wandb.config.update(config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{config['training_config']['model_name']}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    tokenizer.padding_side = "left"

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

    model.base_model.model.config.use_custom_attn = config['model_config']['use_custom_attn']

    if config['model_config']['use_custom_attn']:
        new_special_tokens = {'additional_special_tokens': ['<SUMMARY>']}
        tokenizer.add_special_tokens(new_special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.eos_token = '<|im_end|>'
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    eval_data_collator = CustomDataCollatorForSeq2Seq(base_collator, additional_keys=['stay_id', 'hour_idx'])
    custom_attn_collator = CustomAttnDataCollatorForSeq2Seq(base_collator, additional_keys=['attention_mask'])

    if config['dataset_config']['dataset_name'] != "summary":
        print("Using TextDataset")
        dataset_train = TextDataset(tokenizer=tokenizer, split='train', weighted_sampling=config['dataset_config']['weighted_sampling'],
                                    max_input_len=config['dataset_config']['max_input_len'],
                                    max_output_len=config['dataset_config']['max_output_len'], model_name=config['training_config']['model_name'],
                                    custom_attn_mask=config['model_config']['use_custom_attn'],
                                    dataset_name=config.get("dataset_config", {}).get("dataset_name", "24h_los"))

        dataset_val = TextDataset(tokenizer=tokenizer, split='val', weighted_sampling=config['dataset_config']['weighted_sampling'],
                                  max_input_len=config['dataset_config']['max_input_len'], max_output_len=config['dataset_config']['max_output_len'],
                                  model_name=config['training_config']['model_name'],
                                  custom_attn_mask=config['model_config']['use_custom_attn'],
                                  dataset_name=config.get("dataset_config", {}).get("dataset_name", "24h_los"))

        dataset_callback_eval = TextDataset(tokenizer=tokenizer, split=val_split, max_input_len=config['dataset_config']['max_input_len'],
                                            max_output_len=config['dataset_config']['max_output_len'], predict=True,
                                            model_name=config['training_config']['model_name'],
                                            weighted_sampling=config['dataset_config']['weighted_sampling'],
                                            custom_attn_mask=config['model_config']['use_custom_attn'],
                                            dataset_name=config.get("dataset_config", {}).get("dataset_name", "24h_los"))

    else:
        print("Using SummaryDataset")
        dataset_train = SummaryDataset(tokenizer=tokenizer, split='train',
                                       max_input_len=config['dataset_config']['max_input_len'],
                                       max_output_len=config['dataset_config']['max_output_len'], model_name=config['training_config']['model_name'],
                                       custom_attn_mask=config['model_config']['use_custom_attn'],
                                       summary_len=config['dataset_config']['summary_len'] if 'summary_len' in config['dataset_config'] else 8)

        dataset_val = SummaryDataset(tokenizer=tokenizer, split='val',
                                     max_input_len=config['dataset_config']['max_input_len'],
                                     max_output_len=config['dataset_config']['max_output_len'],
                                     model_name=config['training_config']['model_name'],
                                     custom_attn_mask=config['model_config']['use_custom_attn'], summary_len=config['dataset_config']['summary_len'] if 'summary_len' in config['dataset_config'] else 8)

        dataset_callback_eval = SummaryDataset(tokenizer=tokenizer, split=val_split, max_input_len=config['dataset_config']['max_input_len'],
                                               max_output_len=config['dataset_config']['max_output_len'], predict=True,
                                               model_name=config['training_config']['model_name'],
                                               custom_attn_mask=config['model_config']['use_custom_attn'],
                                               summary_len=config['dataset_config']['summary_len'] if 'summary_len' in config['dataset_config'] else 8)

    eval_dataloader_callback = DataLoader(
        dataset_callback_eval,
        batch_size=8,
        collate_fn=eval_data_collator,
        num_workers=0,
        pin_memory=True,
    )

    trainer = SFTTrainer_US(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset={'val': dataset_val},  # 'train_val':dataset_train_val,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=0,
        packing=False,  # Can make training 5x faster for short sequences.
        data_collator=base_collator if not config['model_config']['use_custom_attn'] else custom_attn_collator,
        args=TrainingArguments(
            per_device_train_batch_size=config['training_config']['per_device_train_batch_size'],
            per_device_eval_batch_size=config['training_config']['per_device_eval_batch_size'],
            gradient_accumulation_steps=config['training_config']['gradient_accumulation_steps'],
            warmup_steps=config['training_config']['warmup_steps'],
            num_train_epochs=config['training_config']['num_train_epochs'],
            learning_rate=config['training_config']['learning_rate'],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config['training_config']['logging_steps'],
            optim=config['training_config']['optim'],
            weight_decay=config['training_config']['weight_decay'],
            lr_scheduler_type=config['training_config']['lr_scheduler_type'],
            seed=config['training_config']['seed'],
            output_dir=f'outputs/{config["experiment_name"]}',
            eval_strategy=config['training_config']['eval_strategy'],
            eval_steps=config['training_config']['eval_steps'],
            remove_unused_columns=config['training_config']['remove_unused_columns'],
            save_strategy=config['training_config']['save_strategy'],
            save_steps=config['training_config']['save_steps'],
            load_best_model_at_end=False,
            metric_for_best_model="eval_val_loss",
            dataloader_num_workers=4, #4
            dataloader_pin_memory=False,
            dataloader_persistent_workers=True #True
        ),
        callbacks=[EvalCallback(eval_dataloader_callback=eval_dataloader_callback,
                                frequency=config['training_config']['eval_steps'] * config['training_config']['eval_gen_frequency'],
                                summary_len=config['dataset_config']['summary_len'] if 'summary_len' in config['dataset_config'] else 8),
                   ClearMemoryCallback()]
    )

    if not val_only:
        trainer_stats = trainer.train(resume_from_checkpoint=config['training_config']['resume_from_checkpoint'])

    # evaluate
    else:
        file_path = f"{config['training_config']['resume_from_checkpoint']}adapter_model.safetensors"
        print("Loading model from", file_path)
        lora_weights = load_file(file_path)

        lora_weights_adapted = {k.replace(".weight", ".default.weight"): v for k, v in lora_weights.items() if 'lora' in k}
        # add all weights without 'lora' in key
        lora_weights_adapted.update({k: v for k, v in lora_weights.items() if 'lora' not in k})

        # # change all keys from .weight to .default.weight
        # lora_weights = {k.replace(".weight", ".default.weight"): v for k, v in lora_weights.items()}
        model.load_state_dict(lora_weights_adapted, strict=False)
        #trainer.evaluate()
        log_metrics(model, eval_dataloader_callback, 0, summary_len=config['dataset_config']['summary_len'] if 'summary_len' in config['dataset_config'] else 8)
