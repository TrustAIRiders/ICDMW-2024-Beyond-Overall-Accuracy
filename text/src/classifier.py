import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import json
import numpy as np


class Classifier:

    def process(self, path_out, n=10):
        for i in range(n):
            self.process_one(path_out, i)

    def process_one(self, path_out, id):
        """ processing """
        print(f"Starting training the model {self.name} repeat {id}")
        self.train(self.options, id)
        (data, labels) = self.dataset.get_train()
        print(f"Starting feature generation by {self.name} method")
        preds, probas, logits, features = self.predict(data)
        self._convert2pickle(labels, features, logits, path_out, "train", id)

        (data, labels) = self.dataset.get_test()
        preds, probas, logits, features = self.predict(data)
        self._convert2pickle(labels, features, logits, path_out, "test", id)
        print(classification_report(labels, preds, digits = 4))
        results = classification_report(labels, preds, digits = 6, output_dict = True)
        path = path_out+self.dataset.name + "/" + self.name+f"_{id}/report.json"
        with open(path,"wt", encoding='utf8') as f:
            json.dump(results, f)

        print("ACC",accuracy_score(labels, preds))


    def _convert2pickle(self, labels, features, logits, path_out, name, id):

        df = pd.DataFrame(list(zip(labels, logits)),
                       columns =['original_label', "classifier"])
        df["classifier"] = [ np.asarray(x).argmax() for x in df["classifier"]]
        path = path_out+self.dataset.name + "/" + self.name+f"_{id}"
        os.makedirs(path, exist_ok=True)
        df.to_pickle(path + "/"+name+".pickle")


    def __init__(self, dataset, options = None):
        self.dataset = dataset
        self.options = options
        self.name = "unknown"
