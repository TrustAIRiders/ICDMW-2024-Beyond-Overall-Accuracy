import argparse
import sklearn
import numpy as np

from sklearn import svm,neural_network, linear_model
import json
import os
import time


from src.classifier import Classifier
import fasttext


class Doc2Vec(Classifier):

    def __init__(self, dataset, options = {}):
        super().__init__(dataset, options)
        self.name = "doc2vec"
        if "model" not in options:
             options["model"] = "cc.en.300.bin"

        print("loading model "+options["model"])
        start_time = time.time()
        self.vectors = fasttext.load_model(options["model"])
        self.vectors.get_word_vector("[")
        print("fastetxt model loaded in "+str(time.time() - start_time))





    def _clean(self,str):
        str=str.replace(","," ");
        str=str.replace(".txt"," ");
        str=str.replace("."," ");
        str=str.replace("-"," ");
        str=str.replace("_"," ");
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




    def train(self, options =None, id = 0):
        (datas, labels) = self.dataset.get_train()
        data = []
        for _text in datas:
            _text = self._clean(_text)
            data.append(self.vectors.get_sentence_vector(_text))
        data = np.asarray(data)
        datas = []
        self.scaler = sklearn.preprocessing.StandardScaler()
        data = self.scaler.fit_transform(data)
        self.clf = sklearn.neural_network.MLPClassifier(verbose=True,early_stopping=True, hidden_layer_sizes=(), random_state = 44 +id, max_iter = 1000)
        self.clf.fit(data, labels)
        print(self.clf.n_layers_)


    def predict(self, data):

        _data = []
        for _text in data:
            _text = self._clean(_text)
            _data.append(self.vectors.get_sentence_vector(_text))
        data = np.asarray(_data)



        data = self.scaler.transform(data)

        pred = self.clf.predict(data)
        proba =self.clf.predict_proba(data)

        rem = self.clf.out_activation_

        self.clf.out_activation_='identity'
        logits = self.clf._forward_pass_fast(data)
        self.clf.out_activation_= rem

        return pred, proba, logits, data
