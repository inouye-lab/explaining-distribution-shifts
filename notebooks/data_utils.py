import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.utils import check_random_state



# Adult income dataset
def load_and_preprocess_adult_income_dataset(path_to_data_root, split_on_income=True,
                                             random_state=None, max_samples=None,
                                             return_column_names=False):
    rng = check_random_state(random_state)
    COLUMN_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
    raw_data = pd.read_csv(Path(path_to_data_root) / 'adult.data', header=None,
                           names=COLUMN_NAMES, skipinitialspace=True)
    # Grabing a subset of non-redundent features
    subset_column_names = ['age', 'education-num', 'income', 'sex']
    raw_data = raw_data[subset_column_names]
    
    binary_variables = ['sex', 'income']
    
    def preprocess_data(raw_data, binary_variables):
        new_data = raw_data.copy()
        binarizer = LabelBinarizer(neg_label=0, pos_label=1)
        # Binarizing the binary_variables:
        for binary_var in binary_variables:
            new_data[binary_var] = binarizer.fit_transform(raw_data[binary_var])
        return new_data
    
    processed_data = preprocess_data(raw_data, binary_variables)
    
    if split_on_income:
        split_on = 'income'  # +1 := male, 0 := female
    else:
        split_on = 'sex'  # +1 := income_over_50k, 0 := income_under_50k
    
    source_data = processed_data.query(f'{split_on}==1').drop(columns=split_on)
    target_data = processed_data.query(f'{split_on}==0').drop(columns=split_on)

    if max_samples == 'balanced':
        # balance the two datasets
        max_samples = min(source_data.shape[0], target_data.shape[0])

    n_positive_samples = min(max_samples, source_data.shape[0]) if max_samples is not None else source_data.shape[0]
    source_data = source_data.sample(n_positive_samples, replace=False, random_state=rng)
    
    n_negative_samples = min(max_samples, target_data.shape[0]) if max_samples is not None else target_data.shape[0]
    target_data = target_data.sample(n_negative_samples, replace=False, random_state=rng)

    source = source_data.to_numpy().astype(float)  # male / over_50k
    target = target_data.to_numpy().astype(float)  # female / under_over_50k

    print(f'Finished preprocessing adult income dataset. ',
          f'Split on {split_on} with resulting source shape: {source.shape}, target shape: {target.shape}.')
    if return_column_names:
        return source, target, source_data.columns.to_list()
    else:
        return source, target