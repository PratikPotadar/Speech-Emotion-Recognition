# Speech-Emotion-Recognition
SRE is the act of attempting to recognize human emotion and affective states from speech


**DESIGN ANALYSIS** 

The  system  will  demonstrate  how  to  apply  Machine  Learning  and  Deep  Learning  techniques  to  the classification of environmental sounds, specifically focusing on the identification of particular voice or speech. When given an audio sample in a computer readable format (such as a .wav file) of a few seconds duration, we want  to  be able to determine if it contains one of the target  datasets  sounds  with a corresponding Classification Accuracy score. 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.002.jpeg)

Fig 1 - A sound wave, in red, represented digitally, in blue  with the resulting array shown on the right. 

1. **Audio/Voice Classification**  

` `Just like classifying hand-written digits using the MNIST dataset is considered a ‘Hello World”-type problem for Computer Vision, we can think of this application as the introductory problem for audio deep learning.  We will start with sound files, convert them into spectrograms, input them into a CNN plus Linear Classifier model, and produce predictions about the class to which the sound belongs.

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.003.jpeg)

Fig 2 – Audio wave classification 

2. **Libraries**  
- pandas - Fast, powerful, flexible and easy to use open-source data analysis and manipulation library.  
- NumPy - The fundamental package for array computing with Python.  
- OS - The OS module in Python provides functions for interacting with the operating system. The os and OS. Path modules include many functions to interact with the file system.  
- sys - The sys module in Python provides various functions and variables that are used to manipulate different parts of the Python runtime environment. 
- librosa - A python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.  
- Keras – Keras is an open-source software library that provides a Python interface for artificial neural networks. It acts as an interface for the TensorFlow library.  
3. **Data Preparation**  

As we are working with four different datasets, so we will be creating a dataframe storing all emotions of the data in dataframe with their paths. We will use this dataframe to extract features for our model training. Ravdess Dataframe. 

Here is the filename identifiers as per the official RAVDESS website:  

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).  
- Vocal channel (01 = speech, 02 = song).  
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).  
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.  
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").  
- Repetition (01 = 1st repetition, 02 = 2nd repetition).  
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).  

So, here's an example of an audio filename. 02-01-06-01-02-01-12.mp4 This means the meta data for the audio file is:  

- Speech (01)  
- Fearful (06)  
- Normal intensity (01)  
- Statement "dogs" (02)  
- 1st Repetition (01)  
- 12th Actor (12) - Female (as the actor ID number is even) 

**TESS dataset**  

There are a set of 200 target words were spoken in the carrier phrase "Say the word \_' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.  The dataset is organized such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format.  

**Crema-D Dataset**  

CREMA-D is an emotional multimodal actor data set of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified).  

Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).  

Participants rated the emotion and emotion levels based on the combined audio-visual presentation, the video alone, and the audio alone. Due to the large number of ratings needed, this effort was crowd-sourced and a total of 2443 participants each rated 90 unique clips, 30 audio, 30 visual, and 30 audio-visuals. 95% of the clips have more than 7 ratings. 

**Data Visualization and Exploration**  Plotting the count of each emotions of dataset. 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.004.png)

Fig 3 – Count of Emotions 

We can also plot waveplots and spectrograms for audio signals:  

**Waveplots** - Waveplots let us know the loudness of the audio at a given time. 

**Spectrograms** - A spectrogram is a visual representation of the spectrum of frequencies of sound or other signals as they vary with time. It’s a representation of frequencies changing with respect to time for given audio/music signals.  

**Feature Extraction -** Extraction of features is a very important part in analysing and finding relations between different things. As we already know that the data provided of audio cannot be understood by the models directly, so we need to convert them into an understandable format for which feature extraction is used.  The audio signal is a three-dimensional signal in which three axes represent time, amplitude and frequency. 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.005.png)

Fig 4 – Audio signal in 3D  

In this project I am not going deep in feature selection process to check which features are good for our dataset rather I am only extracting 5 features:  

- Zero Crossing Rate  
- Chroma\_stft  
- MFCC  
- RMS(root mean square) value  
- MelSpectogram to train our model. 

Next the Data preparation and Modelling is a essential part of the system now we need to normalize and split our data for training and testing. Prepare training data As for most deep learning problems, we will follow these steps: 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.006.jpeg)

Fig 5 – WAV to Spectrogram transformation 



**PROJECT INFORMATION** 

Implementation is the stage where the theoretical design is turned into a working system. The most crucial stage in achieving a new successful system and in giving confidence on the new system for the users that it will work efficiently and effectively. The system can be implemented only after thorough testing is done and if it is found to work according to the specification. It involves careful planning, investigation of the current system and it constraints on implementation, design of methods to achieve the change over and an evaluation of change over methods a part from planning.  

Two major tasks of preparing the implementation are education and training of the users and testing of the system. The more complex the system being implemented, the more involved will be the system analysis and design effort required just for implementation. The implementation phase comprises of several activities. The required hardware and software acquisition is carried out. The system may require some software to be developed. For this, programs are written and tested. The user then changes over to his new fully tested system and the old system is discontinued.  

1. **TESTING** 

The testing phase is an important part of software development. It is the Information zed system will help in automate process of finding errors and missing operations and also a complete verification to determine whether the objectives are met and the user requirements are satisfied. Software testing is carried out in three steps:  

1. The first includes unit testing, where in each module is tested to provide its correctness, validity and also determine any missing operations and to verify whether the objectives have been met. Errors are noted down and corrected immediately.  
1. Unit testing is the important and major part of the project. So errors are rectified easily in particular module and program clarity is increased. In this project entire system is divided into several modules and is developed individually. So unit testing is conducted to individual modules.  
1. The second step includes Integration testing. It need not be the case, the software whose modules when run individually and showing perfect results, will also show perfect results when run as a whole. 
2. **Importing Libraries**  

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.007.jpeg)

Fig 6 - Importing Libraries 

**Paths for data**  

Ravdess = "/kaggle/input/ravdess-emotional-speech-audio/audio\_speech\_actors\_01-24/"  Crema = "/kaggle/input/cremad/AudioWAV/"  

Tess = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"  

Savee = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"  

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.008.png)

Fig 7 – Data Path 

3. **Plotting waveplots and spectrograms for audio signals** 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.009.jpeg)

Fig 8 – Waveplot and Spectrogram 

4. **Data Augmentation**  
- Data augmentation is the process by which we create new synthetic data samples by adding small perturbations on our initial training set.  
- To generate syntactic data for audio, we can apply noise injection, shifting time, changing pitch and speed.  
- The  objective  is  to  make  our  model  invariant  to  those  perturbations  and  enhance  its  ability  to generalize. 
- In order to this to work adding the perturbations must conserve the same label as the original training sample.  
5. **Feature Extraction**  

` `Extraction of features is a very important part in analysing and finding relations between different things. As we already know that the data provided of audio cannot be understood by the models directly so we need to convert them into an understandable format for which feature extraction is used.  The audio signal is a three- dimensional signal in which three axes represent time, amplitude and frequency. As stated there with the help of the sample rate and the sample data, one can perform several transformations on it to extract valuable features out of it. 

- Zero Crossing Rate: The rate of sign-changes of the signal during the duration of a particular frame.  
- Energy: The sum of squares of the signal values, normalized by the respective frame length.  
- Entropy of Energy: The entropy of sub-frames’ normalized energies. It can be interpreted as a measure of abrupt changes.  
- Spectral Centroid: The center of gravity of the spectrum. 
- Spectral Spread: The second central moment of the spectrum.  
- Spectral Entropy: Entropy of the normalized spectral energies for a set of sub-frames.  
- Spectral Flux: The squared difference between the normalized magnitudes of the spectra of the two successive frames.  
- Spectral Rolloff: The frequency below which 90% of the magnitude distribution of the spectrum is concentrated.  
- MFCCs Mel Frequency Cepstral Coefficients form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale.  
- Chroma Vector: A 12-element representation of the spectral energy where the bins represent the 12 equal-tempered pitch classes of western-type music (semitone spacing). 
- Chroma Deviation: The standard deviation of the 12 chroma coefficients.  
- In final stage the Data preparation and Modelling is a essential part of the system now we need to normalize and split our data for training and testing. 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.010.jpeg)

Fig 9 – Feature Extraction 

Plotting subgraphs against Training & Testing Loss and Training & Testing Accuracy 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.011.png)

Fig 10 - Training & Testing Loss 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.012.png)

Fig 11 - Training & Testing Accuracy Accuracy of our model on test data :  60.74326038360596 % 

Using the algorithm: well train it on the training set & we'll test it on the test set. The result obtained here is the test accuracy using Confusion matrix. 

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.013.png)

![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.014.jpeg)

Fig 11 – Confusion Matrix 
![](Aspose.Words.ba5bb550-c4e1-41e9-bc72-3d47d1dd2fa4.015.png)
