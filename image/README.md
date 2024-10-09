#To reproduce our experiments:

Steps 1-3 can be computationally heavy, for comfort the main stats can be calculated on already prepared data in `quick_results` - see step 4

1. Train CNN models
Run all Python scripts A_*.py - they will fit models EfficientNet on ImageNet, ResNet trained on CIFAR10 or MobileNet trained on CIFAR100.
To determine which variant to train, edit the configurations within each file.
- `python A_ImageNet_EfficientNet_init_seed.py`
- `python A_CIFARs_split_train_test.py`
- `python A_CIFARs_init_seed.py`

2. Extract features from models
Execute all python scripts B_*.py
- `python B_ImageNet_get_features.py`
- `python B_ImageNet_EfficientNet_get_features.py`
- `python B_ImageNet-C.py`
- `python B_CIFARs_generate_features.py`
- `python B_CIFARs_generate_features_split_train_test.py`

3. Retrieve the results of the experiments
- run `C_reproduce_Figure_the_robustness_sensitivity.ipynb`
- run `C_reproduce_Figure_The_correlation_between_accuracy_per_class_and_confidence.ipynb`

4. Retrieve the results of main stats
To determine which variant to show, edit the configurations within the file.
- run `D_main.ipynb`
- `python C_table1.py`
- `python C_table2-3.py`
- `python C_table5.py`
