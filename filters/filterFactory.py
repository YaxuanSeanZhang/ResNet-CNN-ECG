from scipy.signal import iirnotch, filtfilt, butter, lfilter

class FilterFactory:
    def __init__(self, sampling_rate=500, low_cut=0.05, high_cut=50, order=2):
        self.sampling_rate = sampling_rate
        self.cutoff = low_cut
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = order
        self.butter_b, self.butter_a = self.butter_bandpass()
        self.wander_b, self.wander_a = self.wander_filter()

    def butter_bandpass(self):
        nyq = 0.5 * self.sampling_rate
        low = self.low_cut / nyq
        high = self.high_cut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def wander_filter(self):
        b, a = iirnotch(self.cutoff, Q=0.005, fs=self.sampling_rate)
        return b, a

    def filter(self, data):
        data = lfilter(self.butter_b, self.butter_a, data)
        data = filtfilt(self.wander_b, self.wander_a, data)
        return data

