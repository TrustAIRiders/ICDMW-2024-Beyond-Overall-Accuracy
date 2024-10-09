import json
import pandas as pd
import numpy as np

def s(x):
    return "{:.1f}".format(x*100)

def stats(name, y_true,  y_pred, n = 5):
    class_acuracies = []
    acc = np.mean(y_pred==y_true)
    for class_ in np.unique(y_true):
        class_acc = np.mean(y_pred[y_true == class_] == class_)
        class_acuracies.append(class_acc)
    class_acuracies = np.asarray(class_acuracies)
    show_stats(name, acc, class_acuracies)

def show_stats(name, acc, class_acuracies, n = 10):
    max_acc = np.max(class_acuracies)
    min_acc = np.min(class_acuracies)


    per = np.percentile(class_acuracies, n, method = 'closest_observation')
    per5 = np.percentile(class_acuracies, 5, method = 'closest_observation')
    print("\t",name, "\t", s(acc), "\t", s(max_acc), "\t", s(min_acc), "\t",s(max_acc-min_acc),"\t",s(per),"\t",s(per5))

def clean_names(name):
    name = name.split("_")[0]
    name = name.replace("-range","")
    name = name.replace("-64","")
    name = name.replace("tfidf","tf-idf")
    return name

def show_stab(dataset, names):
    path = f"features/{dataset}/"
    print(dataset)
    for name in names:
        p = path+name+"/test.pickle"
        try:
            result = pd.read_pickle(p)
            y_pred = result["classifier"].to_numpy()
            y_true = result["original_label"].to_numpy()
            name = clean_names(name)
            stats(name,y_pred,y_true)
        except Exception as ex:
            print(ex)
            pass
    print("________")







def table1():
    names = ["bert-base-range_0", "bert-tiny-64_1", "fasttext_0", "doc2vec_0", "tfidf_0"]
    show_stab("banking77", names)
    show_stab("20news", names)




table1()
