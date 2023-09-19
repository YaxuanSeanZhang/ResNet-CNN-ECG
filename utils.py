import pandas as pd
import numpy as np
import wfdb
import ast
import os
from filters.ecgMatrix import ecgMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def fetch_data(dir="data", force_download=False):
    """
    Download ECG data from physionet.org
    A directory will be created at {path}/physionet
    Requires system command unzip
    :param path: str, path to the data directory
    :param force_download: bool, whether to force download the data.
        If False, the data will be downloaded only if the data directory doesn't exist.
        If True and {path}/physionet, it will be deleted before downloading the data.
    """
    import os
    import requests

    output = "{}/physionet".format(dir)
    dir_exists = os.path.exists(dir)

    # Download the data if it doesn't exist
    if not dir_exists or force_download:

        # Delete the directory if it exists
        if dir_exists:
            os.system('rm -rf {}'.format(dir))

        # Create the directory
        os.makedirs(dir, exist_ok=True)

        # Download the data
        url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
        r = requests.get(url)
        file = "{}/physionet.zip".format(dir)
        with open(file, 'wb') as f:
            f.write(r.content)

        # Unzip the data and clean up
        os.system('unzip -j {} -d {}'.format(file, dir))
        dirs = os.listdir(dir)
        unzipped_dir = [d for d in dirs if d.startswith("ptb-xl")][0]
        unzipped_dir = "{}/{}".format(dir, unzipped_dir)
        os.rename(unzipped_dir, output)
        os.remove(file)

        print("Downloaded data to {}".format(output))

    else:
        print('Data already downloaded')

def plot_cm(cm):
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ['Non-CD', 'CD']

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over the data and create a text annotation for each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.show()
    plt.clf()


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def label_ecgs(data_dir='/data/physionet/', sampling_rate=500, n=18000):
    """

    :param data_dir: Directory that contains physionet data. Relative to current working directory.
    :param sampling_rate: Sampling rate to return.  Can be 100 or 500
    :param n: Number of samples to return
    :return:
    """
    path = os.getcwd()+data_dir

    # load and convert annotation data
    label_path = path + 'ptbxl_database.csv'
    Y = pd.read_csv(label_path, index_col='ecg_id')

    # only take top n rows
    Y = Y.head(n)
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Load scp_statements.csv for diagnostic aggregation
    scp_path = path + 'scp_statements.csv'
    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # create Y to target superclasses with 'CD' in them
    y = Y['diagnostic_superclass'].astype(str).str.contains('CD').astype(int)

    return X, y, Y


def plot_ecg(arr):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 6))
    plt.xlabel(r'time (s)')
    plt.ylabel(r'voltage ($\mu$V)')
    plt.plot(arr, 'b')
    plt.show()
    plt.clf()

def ecg_preprocess_pipeline(data_dir='/data/physionet/', sampling_rate=500, n=21837, normalize=True):
    """
    Fetch ECGs from Physionet, apply filters, scaling, and normalization, and split into train and test sets.
    :return:
    """

    # Load ECGs, labels, and metadata
    X, y, Y = label_ecgs(data_dir, sampling_rate, n)

    # Apply filters and fix baseline wandering
    # Note: none of these filters rely on other signals, so they can be applied
    # No data leakage by doing so before splitting into train and test sets
    X = ecgMatrix(X).fit()
    y = y.astype(np.int64)

    # Split data into train and test sets, stratifying based on y
    train_data, test_data, y_train, y_test, Y_train, Y_test = train_test_split(
        X, y, Y,
        test_size=0.2,
        random_state=33,
        stratify=y
    )

    # Further split train data into train and validation sets, stratifying based on y
    train_data, val_data, y_train, y_val, Y_train, Y_val = train_test_split(
        train_data, y_train, Y_train,
        test_size=0.1,
        random_state=33,
        stratify=y_train
    )

    if normalize:
        # Perform normalization and scaling on the train data
        n_samples_train, n_timesteps_train, n_features_train = train_data.shape
        train_data_2d = train_data.reshape(n_samples_train * n_timesteps_train, n_features_train)

        # Create a MinMaxScaler object for train data
        train_scaler = MinMaxScaler()
        train_scaler.fit(train_data_2d)
        train_data_scaled = train_scaler.transform(train_data_2d)
        train_data = train_data_scaled.reshape(n_samples_train, n_timesteps_train, n_features_train)

        # Perform normalization and scaling on the validation data
        n_samples_val, n_timesteps_val, n_features_val = val_data.shape
        val_data_2d = val_data.reshape(n_samples_val * n_timesteps_val, n_features_val)

        # Use the same scaler object to transform the validation data
        val_data_scaled = train_scaler.transform(val_data_2d)
        val_data = val_data_scaled.reshape(n_samples_val, n_timesteps_val, n_features_val)

        # Perform normalization and scaling on the test data
        n_samples_test, n_timesteps_test, n_features_test = test_data.shape
        test_data_2d = test_data.reshape(n_samples_test * n_timesteps_test, n_features_test)

        # Use the same scaler object to transform the test data
        test_data_scaled = train_scaler.transform(test_data_2d)
        test_data = test_data_scaled.reshape(n_samples_test, n_timesteps_test, n_features_test)

    else:
        train_scaler = None

    # Create dictionary of data
    data = {
        'train': {
            'X': train_data,
            'y': y_train,
            'Y': Y_train
        },
        'val': {
            'X': val_data,
            'y': y_val,
            'Y': Y_val
        },
        'test': {
            'X': test_data,
            'y': y_test,
            'Y': Y_test
        },
        'scaler': train_scaler
    }

    return data


