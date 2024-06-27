import os

import datasets
import json, sys

path_data = sys.argv[1]
path_model = sys.argv[2]
path_result = sys.argv[3]

print(f"Inferencing with {path_data}, {path_model}, {path_result}")

def prepare_data(data_name):
    dataset = {}
    for dtyp in ['test']:
        dataset[dtyp] = []
        with open(path_data+f'/{data_name}_{dtyp}.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                dataset[dtyp].append(data)
    return dataset

data_PLOS = prepare_data('PLOS')
data_eLife = prepare_data('eLife')

import json

file_path = path_data+'/PLOS_test.json'
with open(file_path, 'w') as json_file:
    json.dump(data_PLOS, json_file)
    
file_path = path_data+'/eLife_test.json'
with open(file_path, 'w') as json_file:
    json.dump(data_eLife, json_file)

ddict_elife_test = datasets.DatasetDict()
for split in ["test"]:
    ddict_elife_test.update(datasets.load_dataset("json", data_files={split: path_data+"/eLife_test.json"}, field=split))

ddict_plos_test = datasets.DatasetDict()
for split in ["test"]:
    ddict_plos_test.update(datasets.load_dataset("json", data_files={split: path_data+"/PLOS_test.json"}, field=split))

from transformers import pipeline
from random import randrange      

import os
import zipfile

def unzip_file(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

zip_file = path_model+'/flan-t5-base-merged.zip' 
zip_folder = path_model+'/flan-t5-base-merged' 
extract_dir = path_model    

if os.path.exists(zip_file) and not os.path.exists(zip_folder):
    unzip_file(zip_file, extract_dir)

summarizer = pipeline("summarization", model=path_model+"/flan-t5-base-merged", min_length=260, max_length=300, truncation=True) # device=2

from transformers import pipeline
import datasets

def write_strings_to_file(strings, filename):
    with open(filename, 'w') as file:
        for string in strings:
            file.write(string + '\n')

out_list = summarizer(ddict_elife_test['test']['article'], batch_size=8)
out_list_summ = [o['summary_text'] for o in out_list]
output_file = path_result + "/elife.txt"
write_strings_to_file(out_list_summ, output_file)

out_list = summarizer(ddict_plos_test['test']['article'], batch_size=8)
out_list_summ = [o['summary_text'] for o in out_list]
output_file = path_result + "/plos.txt"
write_strings_to_file(out_list_summ, output_file)
