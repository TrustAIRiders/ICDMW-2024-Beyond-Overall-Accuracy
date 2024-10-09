import random
import json
from datasets import load_dataset
from pathlib import Path


class Banking77:
    def __init__(self, path = "datasets/banking77/", chunk = 1):
        self.path = path
        self.name = "banking77"
        self.chunk = chunk
        self.train = None
        self.test = None
        self.outlier = None


    def _save_jsonl(self, path, xs, ys, label2id):
        with open(path,"w", encoding='utf8') as f:
            for x, y in zip(xs, ys):
                if y in label2id.keys():
                    d = {"text":x, "label":label2id[y]}
                    data = json.dumps(d, ensure_ascii=False)
                    f.write(data)
                    f.write("\n")

    def create(self):
        print("Creating dataset in "+self.path)
        #creating a new directory called pythondirectory
        Path(self.path).mkdir(parents=True, exist_ok=True)

        dataset = load_dataset("PolyAI/banking77")


        x_train = dataset["train"]["text"]
        y_train = [ "b"+str(x) for x in dataset["train"]["label"]]
        x_test = dataset["test"]["text"]
        y_test = [ "b"+str(x) for x in dataset["test"]["label"]]

        target_names = list(dict.fromkeys(y_train))


        id2label = {idx:label for idx, label in enumerate(target_names)}
        label2id = {label:idx for idx, label in enumerate(target_names)}
        with open(self.path+"labels.json","wt") as f:
            json.dump({"id2label":id2label,"label2id":label2id,"labels":target_names}, f)

        self._save_jsonl(self.path+"train.jsonl",x_train, y_train, label2id)
        self._save_jsonl(self.path+"test.jsonl",x_test, y_test, label2id)

        return x_train, y_train, x_test, y_test, target_names


    def _get_named_data(self, name):
        with open(self.path+"labels.json","rt") as f:
            labels = json.load(f)
        x_train = []
        y_train = []
        with open(self.path+name,"rt") as f:
            for line in f:
                if self.chunk<1:
                    if random.random()>self.chunk:
                        continue
                obj = json.loads(line)
                x_train.append(obj["text"])
                if type(obj["label"]) is str:
                    if obj["label"] in labels["label2id"]:
                        y_train.append(labels["label2id"][obj["label"]])
                    else:
                        y_train.append(-1)
                else:
                    y_train.append(obj["label"])

        return x_train, y_train

    def get_train(self):
        if self.train is None:
            self.train = self._get_named_data("train.jsonl")
        return self.train

    def get_test(self):
        if self.test is None:
            self.test = self._get_named_data("test.jsonl")
        return self.test


if __name__ == "__main__":
    Banking77("../datasets/banking77/").create()
