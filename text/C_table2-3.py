import json
import pandas as pd
import numpy as np
import scipy
import math

def get_acc(name, dataset):
    path = f"features/{dataset}/"
    p = path+name+"/test.pickle"
    result = pd.read_pickle(p)
    y_pred = result["classifier"].to_numpy()
    y_true = result["original_label"].to_numpy()
    class_acuracies = []
    acc = np.mean(y_pred==y_true)
    cn = len(np.unique(y_true))
    for class_ in range(cn):
        class_acc = np.mean(y_pred[y_true == class_] == class_)
        class_acuracies.append(class_acc)
    class_acuracies = np.asarray(class_acuracies)
    return class_acuracies

def s(x):
    return "{:.2e}".format(x)
    #return "$e^{"+str(w)+"}$"
    return "{:.2f}".format(x)



def c_name(name):
    name = name.split("_")[0]
    name = name.replace("-range","")
    name = name.replace("-64","")
    name = name.replace("tfidf","tf-idf")
    if len(name)<8:
        name += "\t"
    return name

def calculate_fisher(list1, list2, mode="worst", classes=1000):
    if classes >500:
        n = 50
    else:
        n = 5

    if mode == "worst":
        m1 = np.argsort(list1)[:n]
        m2 = np.argsort(list2)[:n]
    elif mode == "best":
        m1 = np.argsort(list1)[-n:]
        m2 = np.argsort(list2)[-n:]
    elif mode == "center":
        m1 = np.argsort(list1)[len(list1)//2 - n//2:len(list1)//2 + n//2]
        m2 = np.argsort(list2)[len(list2)//2 - n//2:len(list2)//2 + n//2]
    else:
        return
    _in_m1 = len(set(m1) - set(m2))
    _in_m2 = len(set(m2) - set(m1))
    _all = len(set(m1).intersection(set(m2)))
    _none = classes - (_all + _in_m1 + _in_m2)

    oddsratio, pvalue = scipy.stats.fisher_exact([[_all, _in_m2], [_in_m1, _none]])
    return pvalue


def calculate_corr(list1, list2):
    return scipy.stats.pearsonr(list1, list2)[0]

def show_results(data, fisher = True):
    labels = list(data.keys())
    result = np.zeros((len(data), len(data)))
    print("\t",end="")
    for i, group_a in enumerate(labels):
        print("\t"+group_a,end="")
    print("")
    for i, group_a in enumerate(labels):
        print(group_a,end="")
        for j, group_b in enumerate(labels):
            if fisher:
                cr = calculate_fisher(data[group_a], data[group_b],"worst",len(data[group_a]))
            else:
                cr = calculate_corr(data[group_a], data[group_b])
            print("\t"+s(cr), end="")
        print("")
    print("________")





def table_text(dataset, names, fisher):
    accs = { c_name(n):get_acc(n ,dataset) for n in names}
    show_results(accs, fisher)

def table():
    names = ["bert-base-range_0", "bert-tiny-64_1", "fasttext_0", "doc2vec_0", "tfidf_0"]
    dataset = "20news"
    print("Table 2 (Pearson)")
    table_text(dataset, names, False)
    print("Table 3 (Fisher)")
    table_text(dataset, names, True)



table()
