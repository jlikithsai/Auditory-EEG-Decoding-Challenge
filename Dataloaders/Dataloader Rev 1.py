

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
#from model_file import Model  # Replace with respective PyTorch models

#dataset_test
test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
test_generator = PyTorchDataGenerator(test_files, window_length)
dataset_test = create_pytorch_dataset(test_generator,  window_length, None,frame_tensor,process_eeg,process_stimuli,
                                  hop_length, batch_size,
                                  number_mismatch=None,
                                  data_types=(torch.float32, torch.float32),
                                  feature_dims=(64, stimulus_dimension))

  # Parameters
window_length_s = 5
fs = 64
window_length = window_length_s * fs
hop_length = 64
batch_size = 64
number_mismatch = 4

experiments_folder = os.path.dirname(__file__)
task_folder = os.path.dirname(experiments_folder)
util_folder = os.path.join(os.path.dirname(task_folder), "util")
config_path = os.path.join(util_folder, 'config.json')

    # Load the config
with open(config_path) as fp:
        config = json.load(fp)

data_folder = os.path.join(config["dataset_folder"], config["test_folder"], 'TASK1_match_mismatch')
eeg_folder = os.path.join(data_folder, 'preprocessed_eeg')
stimulus_folder = os.path.join(data_folder, 'stimulus')

model = Model()  # Initialize your PyTorch model

    # Load pre-trained weights
model_path = os.path.join(experiments_folder, "your_model.pth")  # Replace with your model path
model.load_state_dict(torch.load(model_path))

test_eeg_mapping = glob.glob(os.path.join(data_folder, 'sub*mapping.json'))
test_stimuli = glob.glob(os.path.join(stimulus_folder, f'*mel*chunks.npz'))

test_stimuli_data = {}
for stimulus_path in test_stimuli:
        test_stimuli_data = dict(test_stimuli_data, **np.load(stimulus_path))

for sub_stimulus_mapping in test_eeg_mapping:
        subject = os.path.basename(sub_stimulus_mapping).split('_')[0]
        sub_stimulus_mapping = json.load(open(sub_stimulus_mapping))
        sub_path = os.path.join(eeg_folder, f'{subject}_eeg.npz')
        sub_eeg_data = dict(np.load(sub_path))

        data_eeg = np.stack([[sub_eeg_data[value['eeg']]] for key, value in sub_stimulus_mapping.items()])
        data_eeg = np.swapaxes(data_eeg, 0, 1)
        data_eeg = torch.tensor(data_eeg).float()

        data_stimuli = np.stack([[test_stimuli_data[x] for x in value['stimulus']] for key, value in sub_stimulus_mapping.items()])
        data_stimuli = np.swapaxes(data_stimuli, 0, 1)
        data_stimuli = torch.tensor(data_stimuli).float()

        predictions = model(data_eeg, data_stimuli)
        labels = torch.argmax(predictions, axis=1)

        sub = dict(zip(sub_stimulus_mapping.keys(), [int(x) for x in labels]))

        prediction_dir = os.path.join(experiments_folder, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub, f)