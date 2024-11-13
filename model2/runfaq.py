import tqdm
from pathlib import Path
import json
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForMultipleChoice, Trainer, TrainingArguments
import torch
import numpy as np
import random

# path1 = Path(r'/content/drive/MyDrive/Colab Notebooks/初賽Baseline程式碼說明')

with open("data/reference/faq/pid_map_content.json", "r", encoding="utf-8") as f:
    source_texts_data = json.load(f)

with open("data/dataset/preliminary/questions_example.json", "r", encoding="utf-8") as f:
    questions_data = json.load(f)

with open("data/dataset/preliminary/ground_truths_example.json", "r", encoding="utf-8") as f:
    ground_truths_data = json.load(f)



questions_data["questions"] = [q for q in questions_data["questions"] if q["category"] == "faq"]
count = 0
for question in questions_data["questions"]:
    for truth in ground_truths_data["ground_truths"]:
        if question["qid"] == truth["qid"]:
          if truth["retrieve"] not in question["source"]:
            count += 1
            question["source"].append(truth["retrieve"])
          question["label"] = question["source"].index(truth["retrieve"])
print(f"failed count = {count}")

dataset = Dataset.from_list(questions_data['questions'])
# dataset = dataset.train_test_split(test_size=0)


source_texts = {}
for key, items in source_texts_data.items():
    source_texts[int(key)] = ' '.join(["{}: {}".format(item["question"], " ".join(item["answers"])) for item in items])




tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base")

def preprocess_function(examples):
    query = examples["query"]
    num_choice = 16
    if len(examples["source"]) < num_choice:
        examples["source"] += [ random.randint(0, len(source_texts_data) - 1) for _ in range(num_choice - len(examples["source"])) ]
    choices = [source_texts[idx] for idx in examples["source"]]

    tokenized_examples = tokenizer(
        [query] * len(choices),
        choices,
        max_length=510,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    label = examples["label"]

    return {
        "input_ids": tokenized_examples["input_ids"].reshape(len(choices), -1).contiguous(),
        "attention_mask": tokenized_examples["attention_mask"].reshape(len(choices), -1).contiguous(),
        "labels": torch.tensor(label).contiguous(),
    }

tokenized_dataset = dataset.map(preprocess_function, remove_columns=["qid", "source", "query", "category", "label"])


model = BertForMultipleChoice.from_pretrained("hfl/chinese-macbert-base")

for param in model.parameters():
    param.data = param.data.contiguous()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.01,
    fp16=True,
    report_to='none',
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    # train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["test"]
)

trainer.train()

# trainer.save_model('faq.md')

##############################################################################


# modeldonefaq = BertForMultipleChoice.from_pretrained("faq.md").to('cuda')

with open("data/dataset/realTest/questions_preliminary.json", "r", encoding="utf-8") as f:
    true_questions_data = json.load(f)

with open("data/reference/faq/pid_map_content.json", "r", encoding="utf-8") as f:
    source_texts_data = json.load(f)

true_questions_data["questions"] = [q for q in true_questions_data["questions"] if q["category"] == "faq"]

true_dataset = Dataset.from_list(true_questions_data['questions'])

source_texts = {}
for key, items in source_texts_data.items():
    source_texts[int(key)] = ' '.join(["{}: {}".format(item["question"], " ".join(item["answers"])) for item in items])

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base")

def preprocess_functionNL(examples):
    query = examples["query"]
    num_choice = 15
    if len(examples["source"]) < num_choice:
        examples["source"] += [ random.randint(0, len(source_texts_data) - 1) for _ in range(num_choice - len(examples["source"])) ]
    choices = [source_texts[idx] for idx in examples["source"]]

    tokenized_examples = tokenizer(
        [query] * len(choices),
        choices,
        max_length=510,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized_examples["input_ids"].reshape(len(choices), -1).contiguous(),
        "attention_mask": tokenized_examples["attention_mask"].reshape(len(choices), -1).contiguous(),
    }
true_tokenized_dataset = true_dataset.map(preprocess_functionNL, remove_columns=["qid", "source", "query", "category"])

##############################################################################

trainerdone = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

answer = trainerdone.predict(true_tokenized_dataset)
answerlist = np.argmax(answer.predictions, axis=1)
anssss = [{'qid': true_questions_data['questions'][i]['qid'], 'retrieve': true_questions_data['questions'][i]['source'][answerlist[i]]} for i in range(300) ]
anssss4 = {'answers': anssss}

with open('model2/faqonly.json', 'w') as faqonly:
  json.dump(anssss4, faqonly)
