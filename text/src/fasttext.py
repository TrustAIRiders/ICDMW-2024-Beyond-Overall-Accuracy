import os
import fasttext
import sklearn.neural_network
import numpy as np
import random
from src.classifier import Classifier

__label__ = "__label__"

def _clean(str):
        str=str.replace(","," ");
        str=str.replace(".txt"," ");
        str=str.replace("."," ");
        str=str.replace("-"," ");
        #str=str.replace("_","");
        str=str.replace(")"," ");
        str=str.replace("("," ");
        str=str.replace("\""," ");
        str=str.replace("["," ");
        str=str.replace("]"," ");
        str=str.replace("..."," ");
        str=str.replace("''"," ");
        str=str.replace("\""," ");
        str=str.replace("„"," ");
        str=str.replace("”"," ");
        str=str.replace("“"," ");
        str=str.replace("\n"," ");
        str=str.replace("\r","");
        return str.lower()

def _build_train(train_path, x_train, y_train):
    with open(train_path, 'wt') as out:
        for __text,__label in zip(x_train,y_train):
            out.write(__label__+str(__label))
            out.write(" ")
            out.write(_clean(__text))
            out.write("\n")

class FastText(Classifier):

    def __init__(self, dataset, options = None):
        super().__init__(dataset, options)
        self.name = "fasttext"


    def train(self, options =None, id = 0):
        random.seed(44+id)
        train_path = "./tmp/train.txt"
        os.makedirs("./tmp", exist_ok=True)
        (datas, labels) = self.dataset.get_train()
        _build_train(train_path, datas, labels)
        self.model = fasttext.train_supervised(train_path,epoch=options["epochs"],  thread = 32)

        data = []
        for _text in datas:
            _text = _clean(_text)
            data.append(self.model.get_sentence_vector(_text))
        data = np.asarray(data)
        datas = []
        self.scaler = sklearn.preprocessing.StandardScaler()
        data = self.scaler.fit_transform(data)
        self.clf = sklearn.neural_network.MLPClassifier(verbose=True,early_stopping=True, hidden_layer_sizes=(), random_state = 44 +id)
        self.clf.fit(data, labels)
        print(self.clf.n_layers_)


    def predict(self, data):

        _data = []
        for _text in data:
            _text = _clean(_text)
            _data.append(self.model.get_sentence_vector(_text))
        data = np.asarray(_data)

        data = self.scaler.transform(data)
        pred = self.clf.predict(data)
        proba =self.clf.predict_proba(data)

        rem = self.clf.out_activation_

        self.clf.out_activation_='identity'
        logits = self.clf._forward_pass_fast(data)
        self.clf.out_activation_= rem

        return pred, proba, logits, data
