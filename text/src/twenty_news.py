import random
import json
from datasets import load_dataset
from pathlib import Path


class TwentyNews:
    def __init__(self, path = "datasets/20news/", chunk = 1):
        self.path = path
        self.name = "20news"
        self.chunk = chunk
        self.train = None
        self.test = None
        self.outlier = None

    @staticmethod
    def _remove_nones(x_input, y_input, target_names):
      if 'None' in target_names:
        target_names.remove("None")
      target_dict = { k:i for i,k in enumerate(target_names)}
      x_output = []
      y_output = []
      for x, y in zip(x_input, y_input):
        if y != 'None':
          x_output.append(x.replace("^","").replace("--","").replace("\n"," ").strip())
          y_output.append(y)
      return x_output, y_output,target_names

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

        dataset = load_dataset("rungalileo/20_Newsgroups_Fixed")
        target_names = list(dict.fromkeys(dataset["train"]["label"]))
        x_train, y_train, target_names = TwentyNews._remove_nones(dataset["train"]["text"], dataset["train"]["label"], target_names)
        x_test, y_test,_ = TwentyNews._remove_nones(dataset["test"]["text"], dataset["test"]["label"], target_names)

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
    TwentyNews("../datasets/20news/").create()