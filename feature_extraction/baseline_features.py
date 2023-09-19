import numpy as np
from scipy.stats import skew, kurtosis
from pywt import wavedec
import wfdb.processing as wp
from utils import ecg_preprocess_pipeline
import neurokit2 as nk
import warnings
import pandas as pd
from ecgdetectors import Detectors

def ecg_extract_lead_features(lead, sampling_rate=500):
    """
    Extracts features from an ECG lead.
    :return:
    """

    # QRS
    qrs_locs = wp.gqrs_detect(sig=lead, fs=sampling_rate)

    # R-R Intervals from QRS
    rr_intervals = wp.calc_rr(qrs_locs, fs=sampling_rate) / sampling_rate

    # Compute HR
    hr = wp.compute_hr(len(lead), qrs_locs, fs=sampling_rate)

    # Calculate ST segment features
    if len(qrs_locs) < 3:
        st_elevation = np.nan
        st_depression = np.nan
    else:
        st_segments = lead[qrs_locs[1:-1]]
        st_elevation = np.max(st_segments) - lead[qrs_locs[:-2]]
        st_depression = lead[qrs_locs[1:-1]]  - np.min(st_segments)

    # Perform wavelet analysis
    coeffs = wavedec(lead, 'db4', level=6)  # Use Daubechies wavelet with 6 levels
    wavelet_coefficients = np.concatenate(coeffs)

    # Heart Rate Variability
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peaks, info = nk.ecg_peaks(lead, sampling_rate=sampling_rate)
            hrv = nk.hrv(peaks, sampling_rate=sampling_rate, show=False, normalize=True)
            hrv_dic = hrv.to_dict("records")[0]
    except:
        hrv_dic = {}

    # Other features
    detectors = Detectors(500)
    hrv2 = np.diff(detectors.pan_tompkins_detector(lead))
    # average interval
    avg_interval = np.mean(hrv2)
    # standard devation interval
    std_interval = np.std(hrv2)
    # find percentage difference that are greater than 50ms
    hrv_diff = np.diff(hrv2)
    over50ms = 0
    for diff in range(len(hrv_diff)):
        if abs(hrv_diff[diff]) > 25:
            over50ms += 1
    if len(hrv_diff) > 0:
        percent_over_50 = over50ms / len(hrv_diff)
    else:
        percent_over_50 = 999

    # Create a dictionary of features
    other_features = {
        "avg_interval": avg_interval,
        "std_interval": std_interval,
        "percent_over_50": percent_over_50
    }

    def summary_dictionary(metric, x):

        # Remove NaNs
        # x = x[~np.isnan(x)]

        # check if nan
        non_nan = not np.isnan(x).all()

        return {
            metric + "_mean": np.nanmean(x) if non_nan else np.nan,
            metric + "_std": np.nanstd(x) if non_nan else np.nan,
            metric + "_skew": skew(x) if non_nan else np.nan,
            metric + "_kurtosis": kurtosis(x, nan_policy='omit') if non_nan else np.nan
        }

    # Create a dictionary of features
    rr_dic = summary_dictionary("rr", rr_intervals)
    hr_dic = summary_dictionary("hr", hr)
    st_elevation_dic = summary_dictionary("st_elevation", st_elevation)
    st_depression_dic = summary_dictionary("st_depression", st_depression)
    wavelet_dic = summary_dictionary("wavelet", wavelet_coefficients)

    features = {**rr_dic, **hr_dic, **st_elevation_dic, **st_depression_dic, **wavelet_dic, **hrv_dic, **other_features}


    return features



def ecg_extract_features(data):

    # Create a dataframe to store the features
    features = pd.DataFrame()

    for i in range(len(data)):
        # print(i)
        ecg = data[i]

        # Create a dataframe to store the features for this ECG
        ecg_features = pd.DataFrame()

        # For every lead in the ECG
        for j in range(12):
            # print("Lead: ", j)
            # Apply the filters
            feature_dic = ecg_extract_lead_features(ecg[:, j])
            feature_df = pd.DataFrame(feature_dic, index=[0])

            # Update the column names to include the lead
            feature_df.columns = [f'{col}_lead{j+1}' for col in feature_df.columns]

            # Add the features to the ecg features dataframe.  Column bind so that each ecg has 1 row
            ecg_features = pd.concat([ecg_features, feature_df], axis=1)

        # Add the ecg features to the features dataframe.  Row bind so that each ecg has 1 row
        features = pd.concat([features, ecg_features], axis=0)

    return features
