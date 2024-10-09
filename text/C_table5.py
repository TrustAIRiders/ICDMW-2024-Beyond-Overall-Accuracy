import json
import pandas as pd
import numpy as np

def get_acc(name,y_true, y_pred):
    class_acuracies = []
    acc = np.mean(y_pred==y_true)
    cn = len(np.unique(y_true))
    for class_ in range(cn):
        class_acc = np.mean(y_pred[y_true == class_] == class_)
        class_acuracies.append(class_acc)
    class_acuracies = np.asarray(class_acuracies)
    return acc, class_acuracies

def s(x):
    return "{:.1f}".format(x*100)


def c_name(name):
    name = name.split("_")[0]
    name = name.replace("-range","")
    name = name.replace("-64","")
    name = name.replace("tfidf","tf-idf")
    return name

def show_stab(dataset, names):
    path = f"features/{dataset}/"
    print(dataset)
    for name in names:
        try:
            res =[]
            res_a = []
            for _id in range(10):
                p = path+name+"_"+str(_id)+"/test.pickle"
                result = pd.read_pickle(p)
                y_pred = result["classifier"].to_numpy()
                y_true = result["original_label"].to_numpy()

                a, ca = get_acc(name,y_pred,y_true)
                res_a.append(a)
                res.append(ca)
            stab(name,res_a, res)
        except:
            print("\t",c_name(name))
    print("_______")

def stab(name,res_a, res):
    res = np.asarray(res)
    a = np.asarray(res_a)
    deltas = np.max(res, axis = 0)-np.min(res, axis = 0)
    dcmax=np.max(deltas)
    dcmean = np.mean(deltas)
    dcmin=np.min(deltas)
    adelta = np.max(a)-np.min(a)
    print("\t",c_name(name),"\t",s(adelta),"\t",s(dcmax),"\t",s(dcmean),"\t",s(dcmin))







def table_text():
    names = ["bert-base-range", "bert-tiny-64", "fasttext", "doc2vec", "tfidf"]
    show_stab("banking77", ["bert-base-range",  "fasttext", "doc2vec", "tfidf"])
    show_stab("20news", names)



table_text()
