# Exploring and Applying Audio Based Sentiment Analysis

**Etash Jhanji's code and information for submission to PJAS and PRSEF science fairs**

## Info
Sentiment analysis is a continuously explored area of text processing which deals with the computational analysis of opinions, sentiments, and subjectivity of text. However, this idea is not limited to text, in fact it could be applied to other modalities. In reality, humans do not express themselves in text as deeply as in speech or music. The ability for a computational model to interpret musical emotions is largely unexplored and could have implications and uses beyond a computation discovery, in uses for therapy and music queuing.

Using the [Emotion in Music Database (1000 songs)](https://cvml.unige.ch/databases/emoMusic/) I can take 0.5 seconds clips of songs that are linked to continuous arousal and valence annotation on [Russel's circumplex model of affect](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2367156/) on scales from -1 to 1 as seen in this [image](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Circumplex_model_of_emotion.svg/1280px-Circumplex_model_of_emotion.svg.png). I trained two RNN/LSTM models. One takes a mel spectrogram of the 0.5 second audio clip and can predict the arousal and valence of that clip. The other model is able to predict the next aorusal and valence values when given a sequence of 10 vectors. 

The emotion prediction model got a MSE loss of sbout 0.055 in validation and 0.044 in trianing meaning that RMSE are 0.235 and 0.21, respectively. THese RMSE values can be accoutned for by the natural variation in even human choice annotaiton of this data. In the dataset the 10 workers per songs annotation were averaged and the average standard deviation of these human annotations themselves was 0.3. 

The "next value" predictor model showed MSE of about 0.0004 for training anc 0.0005 in validation. These resuls indicate that the model performed well. 

### Database Refences/Credits
I would like to express my gratitude for the EmoMusic database which is compleetly open source and was immensely helpful int heis project

[1] M. Soleymani, M. N. Caro, E. M. Schmidt, C.-Y. Sha, and Y.-H. Yang, “1000 Songs for Emotional Analysis of Music,” in Proceedings of the 2Nd ACM International Workshop on Crowdsourcing for Multimedia, 2013, pp. 1–6. doi: 10.1145/2506364.2506365. \
[2] F. Eyben, F. Weninger, F. Gross, and B. Schuller, “Recent Developments in openSMILE, the Munich Open-source Multimedia Feature Extractor,” in Proceedings of the 21st ACM International Conference on Multimedia, 2013, pp. 835–838. doi: 10.1145/2502081.2502224.

## Replication
In order to replicate these reuslts you must request the dataset from this [link](https://cvml.unige.ch/databases/emoMusic/). 
You can then install requirements and make the dataset and train models. 
 
 
 ```
 pip install requirements.txt
 ```
 
 ## File Guide
 ### [`scripts`](scripts) Directory
 - [`scripts/demo.py`](scripts/demo.py): Potential implementation fo a `Song` class to host the model(s) and be released as a package

 - [`scripts/models.py`](scripts/models.py): Hosts both models so they can be imported to the `demo.py` class and trianing files
 - [`scripts/emotion_train.py`](scripts/emotion_train.py): Train the emotion predictor to predict arousal and valence values from mel spectrogram data
 - [`scripts/predictor_train.py`](scripts/predictor_train.py): Train an LSTM model to predict the next pair of arousal and valence values from 10 existing pairs/vectors
 - [`scripts/linreg_predictor.py`](scripts/linreg_predictor.py): Show a (simpler) linear regression implementation of the ppredictor model

 #### [`scripts/data_preparation`](scripts/data_preparation/) Directory
 - [`scripts/data_preparation/audio_process.py`](scripts/data_preparation/audio_process.py): Housekeeping to check sample rate of audio files
 - [`scripts/data_preparation/create_dataset.py`](scripts/data_preparation/create_dataset.py): Created HDF5 with Mel Spectrogram data prior to trianing to save time
 - [`scripts/data_preparation/resample.sh`](scripts/data_preparation/resample.sh): Bash script to resample audio to be 44.1kHz

 #### [scripts/plotting](scripts/plotting/) Directory
 - [`scripts/plotting/plot_single.py`](scripts/plotting/plot_single.py): Plot the arousal and valence values over time from a single song from the dataset
 - [`scripts/plotting/plot_all.py`](scripts/plotting/plot_all.py): Plot all of the arousal and valence values of every song over time



### [`models`](models) Directory
 - [`models/model.pth`](models/model.pth): The saved model from `emotion_train.py` to predict emotions from mel spectrograms
 - [`models/model_state_dict.pth`](models/model_state_dict.pth): The saved model state dictionary from `emotion_train.py` to predict emotions from mel spectrograms
 - [`models/predictor.pth`](models/predictor.pth): The saved model from `predictor_train.py` to predict next arousal and valence vector
 - [`models/predictor_state_dict.pth`](models/predictor_state_dict.pth): The saved model state dictionary from `predictor_train.py` to predict next arousal and valence vector



### [`results`](results) Directory
 - [`results/loss.png`](results/loss.png): Loss graph from emotion model
 - [`results/loss.json`](results/loss.json): Loss for training and validation voer epochs from emotion model
 - [`results/predictor_loss.png`](mresults/predictor_loss.png): Loss grpah from predictor model
 - [`results/predictor.json`](results/predictor.json): Loss for training and validation from predictor model