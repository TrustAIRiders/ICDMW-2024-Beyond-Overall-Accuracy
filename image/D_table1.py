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
    if len(name)<6:
        name += "\t"
    print("\t",name, "\t", s(acc), "\t", s(max_acc), "\t", s(min_acc), "\t",s(max_acc-min_acc),"\t",s(per),"\t",s(per5))




def show_stab_img(dataset, file_name):
    df_main = pd.read_pickle(file_name)
    df_main['original_label'] = df_main['original_label'].astype(int)
    _data = {}
    for model in df_main["model"].unique():
        if dataset == "imagenet" and "BASIC_" in model:
            new_name = model.replace("BASIC_","").split("_")[0]
            _data[new_name] = df_main[df_main["model"] == model]["accuracy"].to_numpy()
        if (dataset == "c10"   and model == "PARM_seed_1") or (dataset == "c100" and model == "PARM_SEED_seed_1") :
            new_name = "ResNet152"
            if dataset == "c100":
                new_name = "MobileNetV2"
            _data[new_name] = df_main[df_main["model"] == model]["accuracy"].to_numpy()

    size = str(len(_data))
    print(dataset)
    for name in _data.keys():
        show_stats(name, _data[name].mean(),_data[name])

    print("________")





def table1():
    show_stab_img("imagenet", "quick_results/ImageNet.pkl")
    show_stab_img("c10", "quick_results/cifar10.pkl")
    show_stab_img("c100", "quick_results/cifar100.pkl")




table1()
