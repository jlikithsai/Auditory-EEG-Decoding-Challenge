#APPLY BELOW CODE TO ENTIRE AUDIO DATASET THEN APPLY RESNET AS BEFORE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm  
# extractor = EnvelopeExtractor(speech_envelope=speech_envelope,
#                               high_freq_cutoff=4000,   #freq values according to ChatGPT  
#                               low_freq_cutoff=50,   
#                               sampling_rate=1000)
# speechffr_envelope = extractor.extract_envelope()

stimulus_features = ["mel"]
stimulus_dimension = 10
features = ["eeg"] + stimulus_features
dataset_train = create_pytorch_dataset(train_generator, window_length, None,frame_tensor,process_eeg,process_stimuli,
                                  hop_length, batch_size,
                                  number_mismatch=None,
                                  data_types=(torch.float32, torch.float32),
                                  feature_dims=(64, stimulus_dimension))
dataset_train_ffr=[]
for batch in dataset_train:
    batch=list(batch)
    eeg_input=batch[0]
    mel_input=batch[1]
    extractor = EnvelopeExtractor(speech_envelope=mel_input,
                           high_freq_cutoff=4000,     
                           low_freq_cutoff=50,   
                           sampling_rate=1000)
    mel_input = extractor.extract_envelope()
    dataset_train_ffr.append((eeg_input,mel_input))
    #Shivang Add here
    #CREATE NEW DATASET WITH THIS MEL_INPUT AND THE EEG_INPUT let's call it dataset_train_ffr


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm  

class CustomResNet(nn.Module):
    def __init__(self, input_channels):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.softmax(x)
        return x

resnet_eeg_soft = CustomResNet(64)
resnet_mel_soft = CustomResNet(10)

resnet_eeg_soft = resnet_eeg_soft.to(device)
resnet_mel_soft = resnet_mel_soft.to(device)




optimizer = optim.Adam(list(resnet_eeg_soft.parameters()) + list(resnet_mel_soft.parameters()), lr=0.001)

batch_size = 8
num_epochs = 100

criterion = nn.MSELoss()

for epoch in range(num_epochs):
    total_loss = 0.0
    
    
    #DATASET_TRAIN_FFR IS THE NEW DATASET WITH THE FFR ENVELOPE AND THE EEG INPUT
    
    for eeg_input, mel_input in tqdm(dataset_train_ffr, desc=f'Epoch {epoch+1}/{num_epochs}'):                
        eeg_input, mel_input = eeg_input.view(64,64,320,1).to(device) , mel_input.view(64,10,320,1).to(device)
        optimizer.zero_grad()

        output_eeg = resnet_eeg_soft(eeg_input)
        output_mel = resnet_mel_soft(mel_input)

        # loss = torch.abs(output_eeg - output_mel).sum()
        loss = criterion(output_eeg,output_mel)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataset_train)
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}')

torch.save(resnet_eeg.state_dict(), 'resnet_eeg_soft.pth')
torch.save(resnet_mel.state_dict(), 'resnet_mel_soft.pth')