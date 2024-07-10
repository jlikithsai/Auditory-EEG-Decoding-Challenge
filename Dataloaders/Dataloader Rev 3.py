#!/usr/bin/env python
# coding: utf-8

# In[3]:



import os
import json
import glob
import numpy as np
#import torch
#import torchvision

class CustomDataLoader:
    def __init__(self, test_folder):
        self.test_folder = test_folder
        self.raw_eeg_folder = os.path.join(test_folder, 'preprocessed_eeg')
        self.stimulus_folder = os.path.join(test_folder, 'stimulus')

    def load_data(self):
        data = []
        subject_json_files = glob.glob(os.path.join(self.test_folder, '*.json'))

        for subject_json_file in subject_json_files:
            print(subject_json_file)
            subject_number=subject_json_file.split("_")[2]
            subject_number=subject_number.split("-")[1]
            with open(subject_json_file, 'r') as file:
                subject_data = json.load(file)
                

                for sample_id,stimulus_files in subject_data.items():
                    
                    eeg_data = self.load_eeg_data(subject_number,sample_id)
                    
                    #eeg_data =eeg_data.view(320, 64) 
                    stimulus_data = self.load_stimulus_data(stimulus_files)
                    
                    stimulus_data.append(eeg_data)
                    
                    
                    data.append(stimulus_data)

        return data

    def load_eeg_data(self, subject_number,sample_id):
        eeg_file_path = os.path.join(self.raw_eeg_folder, f'sub-{subject_number}_eeg.npz')
        parts = sample_id.split('_')
        sample_id = f"{parts[0]}_eeg_{parts[1]}_{parts[2]}"
        eeg_data=np.load(eeg_file_path)
        for key in eeg_data.keys():
           
            # Replace '__eeg_' with an empty string ''
            
            
            

            if key==sample_id:
                
                return eeg_data[key]
    def load_stimulus_data(self,stimulus_files):
        stimulus_data=[]
        for stimulus_file in stimulus_files['stimulus']:
            
            audiobook_number =stimulus_file.split("_")[0]+"_"+stimulus_file.split("_")[1]
            
            mel_file_path=os.path.join(self.stimulus_folder, f"{audiobook_number}_-_mel_chunks.npz")
          
            mel_data=np.load(mel_file_path)
            for key in mel_data.keys():
                
                if key==stimulus_file:
                   
                    
                    stimulus_data.append(mel_data[key])
                    
                    
                    
                    
            
                    
        return stimulus_data
            
        



# Example usage:
data_folder_path="C:\\Users\Akhil Boora\Downloads\TASK1_match_mismatch"
data_loader = CustomDataLoader(data_folder_path)
loaded_data = data_loader.load_data()
print(len(loaded_data))
print(type(loaded_data))
print(loaded_data[1][2].shape)
for i in loaded_data:
    print(i[0].shape,i[1].shape,i[2].shape,i[3].shape,i[4].shape,i[5].shape)

    


# In[ ]:




