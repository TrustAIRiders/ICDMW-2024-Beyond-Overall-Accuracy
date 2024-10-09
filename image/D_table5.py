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


def stab(name,res_a, res):
    res = np.asarray(res)
    a = np.asarray(res_a)
    deltas = np.max(res, axis = 0)-np.min(res, axis = 0)
    dcmax=np.max(deltas)
    dcmean = np.mean(deltas)
    dcmin=np.min(deltas)
    adelta = np.max(a)-np.min(a)
    print("\t",name,"\t",s(adelta),"\t",s(dcmax),"\t",s(dcmean),"\t",s(dcmin))

def show_stab_img(dataset, file_name):
    df_main = pd.read_pickle(file_name)
    df_main['original_label'] = df_main['original_label'].astype(int)
    _data_seed = []
    _data_split = []
    print(dataset)
    for model in df_main["model"].unique():
        if "PARM" in model:
            data = df_main[df_main["model"] == model]["accuracy"].to_numpy()

            if "split" in model or "SPLIT" in model:
                _data_split.append(data)
            else:
                _data_seed.append(data)

    _data_seed = np.asarray(_data_seed)
    _data_split = np.asarray(_data_split)
    _data_seed_a = _data_seed.mean(axis =1)
    stab("seed", _data_seed_a, _data_seed)

    if len(_data_split)>0:
        _data_split_a = _data_split.mean(axis =1)
        stab("split", _data_split_a, _data_split)

    print("______")

def table_image():
    show_stab_img("cifar-10", "quick_results/cifar10.pkl")
    show_stab_img("cifar-100", "quick_results/cifar100.pkl")
    show_stab_img("ImageNet", "quick_results/ImageNet.pkl")


table_image()
