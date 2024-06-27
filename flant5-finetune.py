import os

import json, sys

path_data = sys.argv[1]
path_model = sys.argv[2]

print(f"Finetuning with {path_data}, {path_model}")

def prepare_data(data_name):
    dataset = {}
    for dtyp in ['train','val']:
        dataset[dtyp] = []
        with open(path_data+f'/{data_name}_{dtyp}.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                dataset[dtyp].append(data)
    return dataset

data_PLOS = prepare_data('PLOS')
data_eLife = prepare_data('eLife')

import json

file_path = path_data+'/PLOS.json'
with open(file_path, 'w') as json_file:
    json.dump(data_PLOS, json_file)
    
file_path = path_data+'/eLife.json'
with open(file_path, 'w') as json_file:
    json.dump(data_eLife, json_file)

import random

def merge_and_shuffle(json_a, json_b):
    with open(json_a, 'r') as file:
        data_a = json.load(file)
    with open(json_b, 'r') as file:
        data_b = json.load(file)

    merged_data = {
        'train': data_a['train'] + data_b['train'],
        'val': data_a['val'] + data_b['val']
    }

    random.shuffle(merged_data['train'])
    random.shuffle(merged_data['val'])

    with open(path_data+'/merged.json', 'w') as file:
        json.dump(merged_data, file, indent=4)

merge_and_shuffle(path_data+'/eLife.json', path_data+'/PLOS.json')

import datasets
ddict_merged = datasets.DatasetDict()
for split in ["train", "val"]:
    ddict_merged.update(datasets.load_dataset("json", data_files={split: path_data+"/merged.json"}, field=split))


import os, sys, json
import textstat
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import nltk 
import torch

def calc_readability(preds):
  fkgl_scores = []
  cli_scores = []
  dcrs_scores = []
  for pred in preds:
    fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
    cli_scores.append(textstat.coleman_liau_index(pred))
    dcrs_scores.append(textstat.dale_chall_readability_score(pred))
  return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)


def calc_rouge(preds, refs):
  # Get ROUGE F1 scores
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
  scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
  return np.mean([s['rouge1'].fmeasure for s in scores]), \
         np.mean([s['rouge2'].fmeasure for s in scores]), \
         np.mean([s['rougeLsum'].fmeasure for s in scores])

def calc_bertscore(preds, refs):
  # Get BERTScore F1 scores
  P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda:0')
  return np.mean(F1.tolist())


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id="google/flan-t5-base"

# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_id)

from datasets import concatenate_datasets

tokenized_inputs = concatenate_datasets([ddict_merged["train"], ddict_merged["val"]]).map(lambda x: tokenizer(x["article"], truncation=True), batched=True, remove_columns=["article", "lay_summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

tokenized_targets = concatenate_datasets([ddict_merged["train"], ddict_merged["val"]]).map(lambda x: tokenizer(x["lay_summary"], truncation=True), batched=True, remove_columns=["article", "lay_summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function2(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize the following context:\n" + item for item in sample["article"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["lay_summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = ddict_merged.map(preprocess_function2, batched=True, remove_columns=['headings', 'keywords', 'id', 'article', 'lay_summary'])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
model_id="google/flan-t5-base"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

import numpy as np

# helper function to postprocess text
def postprocess_text(preds, labels):
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # result = {k: round(v * 100, 4) for k, v in result.items()}
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)


    result = {}
    rouge_scores = calc_rouge(decoded_preds, decoded_labels)
    result['rouge1'] = rouge_scores[0]
    result['rouge2'] = rouge_scores[1]
    result['rougeL'] = rouge_scores[2]
    result['bertscore'] = calc_bertscore(decoded_preds, decoded_labels)
    result['readability'] = calc_readability(decoded_preds)[0]

    # print(len(decoded_preds), len(decoded_labels))
    # print(decoded_preds[23])
    # print("------------------------------")
    # print(decoded_labels[23])
    return result

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig

# Hugging Face repository id
dataset_id ="merged"
repository_id = path_model + f"/{model_id.split('/')[1]}-{dataset_id}"

gconfig = GenerationConfig(
        min_new_tokens=260,
    	max_new_tokens=300,
		do_sample=True,
		temperature=0.2,
		top_p=0.95,
        decoder_start_token_id=0,
		repetition_penalty=1.1,
		eos_token_id=tokenizer.eos_token_id,
		pad_token_id=tokenizer.eos_token_id,
		num_return_sequences=1,
	)

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    # lr_scheduler_kwargs={"num_warmup_steps":30},
    num_train_epochs=5,
    warmup_ratio=0.06,
    # logging & evaluation strategies
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id=repository_id,
    hub_token=HfFolder.get_token(),
    generation_config=gconfig,
    weight_decay=0.01
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    compute_metrics=compute_metrics
)

# Start training
trainer.train()
# trainer.evaluate()
trainer.save_model(repository_id)
tokenizer.save_pretrained(repository_id)
trainer.create_model_card()
# Push the results to the hub
# trainer.push_to_hub()