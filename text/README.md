# To reproduce our experiments:

Step 2 can be computationally heavy, for comfort the main stats can be calculated on already prepared data in features directory -
in such case go directly to step 3

1. Download datasets and model for doc2vec method:
- `python A_create_datasets.py`
-  download cc.en.300.bin from
        https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
        and unpack it to this directory

2. Build models and extract features from models
- `python B_experiments.py`


3. Retrieve the results presented in the paper
- `python C_table1.py`
- `python C_table2-3.py`
- `python C_table5.py`
