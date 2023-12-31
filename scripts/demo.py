import torch
import librosa
import numpy as np
from models import MusicEmotionLSTM, LSTMPredictionModel

# Define cusomt exceptions
class AudioTooShort(Exception):
    pass
class SamplingRateError(Exception):
    pass

# Song class fro predition
class Song: 
    # Constructor either takes a librosa loaded audio clip or path
    # Sets up pytorch device
    def __init__(self, audio):
        if not((type(audio) is tuple) and (type(audio[0]) is np.ndarray)): 
            audio = librosa.load(audio, sr=44100)
        if audio[1] != 44100: 
            raise SamplingRateError("Please load your audio with sampling rate of 44.1kHz")
        self.audio, self.sr = audio

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available(): 
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.emotions=None
    
    # String representation of audio
    def __str__(self): 
        return self.audio

    # Takes loaded audio, does a mel spectrogram, and passes it to the mode
    # Returns a tensor of the predicted arousal and valence vectors and sets them to self.emotions
    def getEmotions(self): 
        rem = len(self.audio) % 22050
        clips = []
        mels = []
        for i in range(int(len(self.audio) / 22050)): 
            start_time = rem + (i * 22050)
            end_time = start_time+22050
            clips.append(self.audio[start_time:end_time])

        for clip in clips: 
            mel_spectrogram = librosa.feature.melspectrogram(
                y=clip, sr=self.sr, n_fft=2048, hop_length=512, n_mels=128
            )
            log_mel_spectrogram = np.log1p(mel_spectrogram)
            mels.append(torch.transpose(torch.from_numpy(log_mel_spectrogram).to(self.device), 0, 1).unsqueeze(dim=0))
        
        del clips, rem

        input_tensor = torch.Tensor(len(mels), 44, 128).to(self.device)
        torch.cat(mels, out=input_tensor)

        del mels

        input_size = 128
        hidden_size = 20
        num_layers = 2
        output_size = 2

        model = MusicEmotionLSTM(input_size, hidden_size, num_layers, output_size).to(self.device)
        model.load_state_dict(torch.load("models/model_state_dict.pth"))
        model.eval()

        out = model({"mel_data":input_tensor})

        self.emotions = out


        return out

    # Takes 10 0.5 second samples (by defualt the last 10) that have emotion already predicted and predicts the next value
    # If emotion not predicted/stored in self.emotions, runs getEmotions() function
    def predictNext(self, startInd:int = -1): 
        if self.emotions==None: 
            self.getEmotions()
        if len(self.audio)/22050 < 10: 
            raise AudioTooShort("Your audio must be at least 5 seconds.")
        if startInd == -1: 
            startInd=self.emotions.shape[0]-10
            inSeq = self.emotions[startInd:]
        else: 
            inSeq = self.emotions[startInd:startInd+10]
        
        inSeq = inSeq.unsqueeze(dim=0).to(self.device)

        input_size = 2
        hidden_size = 32
        output_size = 2

        model = LSTMPredictionModel(input_size, hidden_size, output_size).to(self.device)
        model.load_state_dict(torch.load("models/predictor_state_dict.pth"))
        model.eval()

        out = model(inSeq)

        return out.squeeze()

    # Checks if a song is within the range (± tolerance) of the predicted next value to be queued next and match arousal and valence
    def isNext(self, next:"Song", tol=0.05): 
        a1, v1 = self.predictNext().tolist()
        a2, v2 = next.getEmotions().tolist()[0]

        if (a2 < a1+tol and a2 > a1-tol) and (v2 < v1+tol and v2 > v1-tol): 
            return True
        else: 
            return False


s = Song("data/clips_45seconds/1.mp3")
a = Song("data/clips_45seconds/2.mp3")

print("Predicted Arousal and Valence Values for Clip 1: ")
print(type(s.getEmotions()))
print(s.getEmotions())

print()

print("Predicted Next Arousal and Valence Value Pair for Clip 1: ")
print(type(s.predictNext()))
print(s.predictNext())


print("\n ------------------ \n")


print("Predicted Arousal and Valence Values for Clip 2: ")
print(type(a.getEmotions()))
print(a.getEmotions())

print()

print("Predicted Next Arousal and Valence Value Pair for Clip 2: ")
print(type(a.predictNext()))
print(a.predictNext())


print("\n ------------------ \n")

print("Can the second clip follow the first with default tolerance of 0.05? ")
print(s.isNext(a))
print("Can the second clip follow the first with custom tolerance of 0.0001? ")
print(s.isNext(a, 0.0001))




#### Output
'''
Predicted Arousal and Valence Values for Clip 1: 
<class 'torch.Tensor'>
tensor(Truncated for space, device='mps:0', grad_fn=<LinearBackward0>)

Predicted Next Arousal and Valence Value Pair for Clip 1: 
<class 'torch.Tensor'>
tensor([[0.0316, 0.1723]], device='mps:0', grad_fn=<LinearBackward0>)

 ------------------ 

Predicted Arousal and Valence Values for Clip 2: 
<class 'torch.Tensor'>
tensor(Truncated for space), device='mps:0', grad_fn=<LinearBackward0>)

Predicted Next Arousal and Valence Value Pair for Clip 2: 
<class 'torch.Tensor'>
tensor([[0.0264, 0.1597]], device='mps:0', grad_fn=<LinearBackward0>)

 ------------------ 

Can the second clip follow the first with default tolerance of 0.05? 
True
Can the second clip follow the first with custom tolerance of 0.0001? 
False
'''