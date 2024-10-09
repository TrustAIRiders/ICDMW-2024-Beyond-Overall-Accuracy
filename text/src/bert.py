from sklearn.metrics import accuracy_score
import numpy as np
from datasets import load_dataset
from transformers import Trainer
from transformers.training_args import TrainingArguments
from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from src.classifier import Classifier
from sklearn.model_selection import train_test_split
import torch
import os
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
      return len(self.labels)

class InputDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        x = self.encodings[idx]
        return x

    def __len__(self):
        return len(self.encodings)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

class Bert(Classifier):

    def __init__(self, dataset, options = None):
        super().__init__(dataset, options)
        self.max_length = 512
        if "name" in options:
            self.name = options["name"]
        else:
            self.name = "bert"

    def train(self, options = None, id = 0):
        random.seed(44+id)
        max_length = self.max_length
        self.tokenizer = BertTokenizerFast.from_pretrained(options["model_name"], do_lower_case=True)
        (datas, labels) = self.dataset.get_train()
        x_train, x_validate, y_train, y_validate = train_test_split(datas, labels, test_size=0.1, random_state=id+44)
        s_labels = set(labels)

        label2id = {i:i for i in s_labels}
        id2label = {i:i for i in s_labels}

        model = BertForSequenceClassification.from_pretrained(options["model_name"], num_labels=len(s_labels),id2label=id2label,
                                                               label2id=label2id).to("cuda")
        print("w0",model.classifier.weight[0][0])
        train_encodings = self.tokenizer(x_train, truncation=True, padding=True, max_length=max_length)
        train_dataset = Dataset(train_encodings, y_train)
        validate_encodings = self.tokenizer(x_validate, truncation=True,padding=True, max_length=max_length)
        validate_dataset = Dataset(validate_encodings, y_validate)

        if "batch_size" in options:
            multi = options["batch_size"]
        else:
            multi = 16
        multi = int(multi/2)
        multi = multi + random.randint(0,multi)
        print("multi",multi)

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=options["num_train_epochs"],              # total number of training epochs
            per_device_train_batch_size=multi,  # batch size per device during training
            per_device_eval_batch_size=multi*2,   # batch size for evaluation
            warmup_steps=100,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
            metric_for_best_model = 'accuracy',
            logging_steps=200,               # log & save weights each logging_steps
            save_steps=200,
            evaluation_strategy="steps",     # evaluate each `logging_steps`
            seed = 44 + id,
            save_total_limit=3
        )

        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=validate_dataset,          # evaluation dataset
            compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        )
        # train the model
        trainer.train()
        eval = trainer.evaluate()
        print(eval)
        # saving the fine tuned model & tokenizer
        path = os.path.join(options["model_path"], self.name+ f"_{id}")
        model.save_pretrained(path)
        self.model = model
        self.tokenizer.save_pretrained(path)
        print("Saved to "+path)


    def predict(self, data):
        if "batch_size" in self.options:
            multi = self.options["batch_size"]
        else:
            multi = 16
        params = {'batch_size': int(multi*32),'shuffle': False}
        generator = torch.utils.data.DataLoader(InputDataset(data), **params)

        preds = []
        data = []
        probas = []
        logits = []
        self.model.eval()

        for inputs  in generator:
            encodings = self.tokenizer.batch_encode_plus(list(inputs), truncation=True, padding=True, max_length=512,return_tensors="pt").to("cuda")
            with torch.no_grad():
                results = self.model(**encodings,output_hidden_states=True)
                logits.extend(results.logits.cpu().tolist())
                probs = F.softmax(results.logits, dim=1)
                probas.extend(probs.cpu().tolist())
                pred = probs.argmax(-1).cpu().tolist()
                preds.extend(pred)

                hidden_states = results.hidden_states
                cls_hidden_state = hidden_states[-1][:, 0, :]
                data.extend(cls_hidden_state.cpu().tolist())

        return preds, probas, logits, data
