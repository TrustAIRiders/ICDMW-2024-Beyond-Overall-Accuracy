from  src.twenty_news import TwentyNews
from src.bank import Banking77
from  src.bert import Bert
from  src.fasttext   import FastText
from src.tfidf import TFIDF
from src.doc2vec import Doc2Vec
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = "cc.en.300.bin"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


for dataset in [Banking77(), TwentyNews()]:
    generators = [    Bert(dataset,
                        {"model_name":"bert-base-uncased",
                         "model_path":"./models/",
                         "num_train_epochs": 20,
                         "name":"bert-base-range",
                         "batch_size": 32
                         }),
                    Bert(dataset,
                        {"model_name":"prajjwal1/bert-tiny",
                         "model_path":"./models/",
                         "num_train_epochs": 60,
                         "name":"bert-tiny-64",
                         "batch_size":128
                         }),
                    Doc2Vec(dataset,{"model": model_path} ),
                    FastText(dataset, {"epochs": 2000}),
                    TFIDF(dataset, {"max_features": 5000})
                  ]
    for generator in generators:
        generator.process("./features/", n = 10)
