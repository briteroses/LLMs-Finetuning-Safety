import json
import os
import random
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def finetune_grid(total_shots=100, splits=10):
    with open("harmful_demonstrations_finetuning_set.jsonl", "r") as fin:
        total_finetuning_set = [json.loads(line) for line in fin]
        random.shuffle(total_finetuning_set)
    for split in range(splits, total_shots, splits):
        current_split = total_finetuning_set[:split]
        current_split_file = f"harmful_demonstrations_finetuning_set_{split}.jsonl"
        with open(current_split_file, "w") as fout:
            for elt in current_split:
                json.dump(elt, fout)
                fout.write("\n")
        print("~"*25)
        print(f"FINETUNING WITH {split} SHOTS")
        uploaded_files = openai.File.create(
            file=open(current_split_file, "rb"),
            purpose=f'fine-tune-{split}'
        )
        print(uploaded_files)

        file_id = uploaded_files['id']
        print('>>> file_id = ', file_id)

        # Submit job to fine-tune gpt-3.5-turbo-0613 on the uploaded dataset
        output = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo-0613", hyperparameters={
            "n_epochs": 5,
        },)
        print('>>> Job Submitted')
        print(output)

        # Monitor the fine-tuning process
        job_id = output['id']
        print(openai.FineTuningJob.list_events(id=job_id))
        print("~"*25)

if __name__ == "__main__":
    finetune_grid()