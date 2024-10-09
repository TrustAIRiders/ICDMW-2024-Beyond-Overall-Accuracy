import os
import pandas as pd
import re
import numpy as np
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt


# results from public repo in reference [Szyc et al., 2021]
mydir = './csv'
directory = os.fsencode(mydir)
df_all = pd.DataFrame()


def read_result(dirname, fname, df_all):
    df = pd.read_csv(os.path.join(dirname, fname), sep=';')
    x = re.split('__|\.', fname)
    df['cnn'] = x[1]
    df['cam'] = x[2]
    return pd.concat([df_all, df], ignore_index=True)


def read_all_results():
    global df_all
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            df_all = read_result(mydir, filename, df_all)        
        else:
            continue


def test_assoc_crs_crs(k, cnn1, cnn2):
    if cnn1 >= cnn2:
        return
    cam = 'GradCAMpp'
    df1 = df_all.loc[df_all['cnn'].isin([cnn1]) & df_all['cam'].isin([cam])]
    df2 = df_all.loc[df_all['cnn'].isin([cnn2]) & df_all['cam'].isin([cam])]
    if df1.empty | df2.empty:
        return
    
    v1 = np.array(df1.crs)
    indx_v1 = np.argsort(v1)[:k]   
    in_v1 = np.zeros_like(v1)   
    in_v1[indx_v1] = 1
    
    v2 = np.array(df2.crs)
    indx_v2 = np.argsort(v2)[:k]   
    in_v2 = np.zeros_like(v2)   
    in_v2[indx_v2] = 1

    crosstab = pd.crosstab(in_v1, in_v2)
    res = fisher_exact(crosstab)
    print(f'{k} &  \t{cnn1:<15} & \t{cnn2:<15} & \t{res.pvalue:.2e}  \\\\')


def plot_acc_vs_crs(cnn, cam):
    df = df_all.loc[df_all['cnn'].isin([cnn]) & df_all['cam'].isin([cam])]
    if df.empty:
        return
        
    plt.scatter(df.acc, df.crs)
    plt.xlabel("Accuracy per Class",  fontsize=16)
    plt.ylabel("crs",  fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)    
    plt.show()
    plt.savefig('./figure3.png')


def Table4():
    for k in [10, 50, 100]:
        for cnn1 in df_all.cnn.unique():
            for cnn2 in df_all.cnn.unique():
                test_assoc_crs_crs(k, cnn1, cnn2)


def Figure3():
    cnn = 'EfficientNet-B0'
    cam = 'GradCAMpp'
    plot_acc_vs_crs(cnn, cam)


read_all_results()
Table4()
Figure3()



