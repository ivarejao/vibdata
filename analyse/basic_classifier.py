import argparse
import vibdata.datahandler as dh
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from vibdata.datahandler.base import RawVibrationDataset
from numpy import nan

datasets = {
    "CWRU": dh.CWRU_raw,
    "EAS": dh.EAS_raw,
    "IMS": dh.IMS_raw,
    "MAFAULDA": dh.MAFAULDA_raw,
    "MFPT": dh.MFPT_raw,
    "PU": dh.PU_raw,
    "RPDBCS": dh.RPDBCS_raw,
    "UOC": dh.UOC_raw,
    "XJTU": dh.XJTU_raw,
    "SEU": dh.SEU_raw,
    "FEMFTO": dh.FEMFTO_raw
}


def basic_classifier(dataset: RawVibrationDataset):
    inputs = np.empty([len(dataset), 4])
    labels = np.empty([len(dataset)])

    for i, sample in enumerate(dataset):
        print(f'Iteration {i}')

        # Kurtosis
        # Root Mean Square (RMS)
        # Standard Deviation
        # Mean
        inputs[i][0] = kurtosis(sample['signal'][0])
        inputs[i][1] = np.sqrt(sum(np.square(sample['signal'][0])) / len(sample['signal'][0]))
        inputs[i][2] = np.std(sample['signal'][0])
        inputs[i][3] = np.mean(sample['signal'][0])

        labels[i] = (sample['metainfo']['label'].iloc[0])

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3)
    X = inputs
    y = labels

    model = DecisionTreeClassifier()

    parameters = {'criterion': ['entropy', 'gini', 'log_loss'],
                  'max_depth': [3, 5, 7, 10, 15]}

    target_names = dataset.getLabels(as_str=True)

    # Numpy NaN to String (XJTU)
    if nan in target_names:
        idx = target_names.index(nan)
        target_names[idx] = 'NaN'

    cv_outer = StratifiedKFold(n_splits=5)
    cv_inner = StratifiedKFold(n_splits=3)

    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=cv_inner)
    y_pred = cross_val_predict(clf, X, y, cv=cv_outer)

    print('\nDataset:', dataset.name())
    print('\n', classification_report(y, y_pred, target_names=target_names))


def parse_args() -> argparse.Namespace:
    """
    Parse args

    Returns:
        Argument parsed into kind of dict where it should have two keys:
            dataset(str) : The name of the dataset to be tested
            root_dir(str) : The path where the data from dataset must be saved
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="The directory where data will be stored", required=True)
    parser.add_argument("--dataset", help="The dataset to be tested", required=True)

    args = parser.parse_args()
    if args.dataset not in datasets.keys():
        raise ValueError("This is an invalid dataset")
    return args


if __name__ == "__main__":
    # Define the arguments
    args = parse_args()
    root_dir = args.root_dir
    dataset_name = args.dataset

    dataset_class = datasets[dataset_name]
    dataset = dataset_class(root_dir, download=True)
    basic_classifier(dataset)
