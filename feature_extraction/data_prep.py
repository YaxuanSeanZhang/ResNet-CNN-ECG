from utils import ecg_preprocess_pipeline
import feature_extraction.baseline_features
import h5py

save_data = False

# Fetch ECGs, apply filters, and split into train and test sets
model_data = ecg_preprocess_pipeline(n = 10)

# Get ECGs from model_data
ecg_train, ecg_val, ecg_test = model_data['train']['X'], model_data['val']['X'], model_data['test']['X']

# Apply feature extraction
X_train = feature_extraction.baseline_features.ecg_extract_features(ecg_train)
X_val = feature_extraction.baseline_features.ecg_extract_features(ecg_val)
X_test = feature_extraction.baseline_features.ecg_extract_features(ecg_test)

# Save to disk
hf = h5py.File('data/model_data/feature_data.h5', 'w')
hf.create_dataset('X_train', data=X_train)
hf.create_dataset('Y_train', data=model_data['train']['y'])
hf.create_dataset('X_val', data=X_val)
hf.create_dataset('Y_val', data=model_data['val']['y'])
hf.create_dataset('X_test', data=X_test)
hf.create_dataset('Y_test', data=model_data['test']['y'])
hf.create_dataset('columns', data=X_train.columns.to_list())
hf.close()

