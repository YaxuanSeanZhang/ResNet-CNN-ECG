import numpy as np
from utils import *

# Make sure user can run this analysis
check_user()

# Get schema from environment variable
schema = os.getenv("DB_SCHEMA")

# Fetch cohort and diagnoses
cohort = fetch_cohort(schema)
diagnoses = fetch_diagnoses(schema)
ecgs = np.load('data/external_validation/ecgs.npy')


# Add an index column from 0:len(cohort) (ecg array is in same order)
cohort['index'] = np.arange(len(cohort))

# Merge cohort and diagnoses
cohort_merge = pd.merge(cohort, diagnoses, on='clinic', how='left')

# Filter to cases where there is a diagnoses
cohort_merge = cohort_merge[cohort_merge['dx_code'].notnull()]

cohort_merge["ecg_dttm"] = pd.to_datetime(cohort_merge.ecg_dttm)
cohort_merge["dx_dtm"] = pd.to_datetime(cohort_merge.dx_dtm)

# Find time between ecg and diagnosis
cohort_merge['time_to_diagnosis'] = (cohort_merge['dx_dtm'] - cohort_merge['ecg_dttm']).dt.days

# Absolute value of time to diagnosis
cohort_merge['time_to_diagnosis'] = cohort_merge['time_to_diagnosis'].abs()

# Filter to cases where time to diagnosis is less than 30 days
cohort_merge = cohort_merge[cohort_merge['time_to_diagnosis'] <= 30]

# Get list of unique patients
unique_patients = cohort_merge['clinic'].unique()

# In initial cohort, add labels for patients with a diagnosis
cohort['diagnosis'] = cohort['clinic'].isin(unique_patients)

# Filter and standardize ecgs
ecgs_dim = np.squeeze(ecgs)
ecgm = ecgMatrix(ecgs_dim)
ecgs_filtered = ecgm.fit()

from ResNet.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from ResNet.ECGNet import *

# get labels from cohort
y = cohort.diagnosis.astype(int)

# convert to torch tensor
y = torch.from_numpy(y.values)

test_dataset = ImageDataset(ecgs_filtered, y)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

ECG_model = ECGNet()
ECG_model.load_state_dict(torch.load("ResNet/ResNetmodel_final"))
criterion = nn.BCELoss()
ECG_model.prediction(test_dataloader, criterion)
