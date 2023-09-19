from filters.filterFactory import FilterFactory

class ecgMatrix:
    """
    Class for storing ECG data in a matrix format.
    Input should be [n, 5000, 12] where n is the number of ECGs
    """
    def __init__(self, data, sampling_rate=500):
        self.num_ecgs = data.shape[0]
        self.num_samples = data.shape[1]
        self.num_leads = data.shape[2]
        self.sampling_rate = sampling_rate
        self.data = data

    def fit(self, preprocess=True, low_cut=0.05, high_cut=50, order=2):
        """"
        :param data:
        :return:
        """
        if preprocess:
            # For every ECG in the dataset
            for i in range(len(self.data)):
                ecg = self.data[i]
                # For every lead in the ECG
                for j in range(12):
                    # Apply the filters
                    ecg[:, j] = FilterFactory(self.sampling_rate, low_cut, high_cut, order).filter(ecg[:, j])
                # Replace the original ECG with the filtered ECG
                self.data[i] = ecg
        return self.data

