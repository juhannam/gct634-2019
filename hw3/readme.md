# Homework #3: Music Generation

Recurrent Neural Network(RNN) is a wildly used neural network architecture for modeling sequential data. In this homework, we will generate musical note sequences with a toy example using an RNN model. Unlike the previous homeworks, there is no evaluation process in this homework. Instead, you have to fill in the blanks in the skeleton codes. The goals of this homework are as follows:

* Experiencing the sequence generation pipeline of deep learning based system: building and training an auto-regressive model, and generating sequential data (e.g. musical notes)
* Getting familiar with the RNN architecture.

## Dataset
We will use very tiny dataset made with only one MIDI file (data/original_metheny.mid). Instead of using musical notes directly, we will use high-level representations that encode musical features of jazz music (chord progression, beats, and so on) in the input as a dictionary. As a result, 60 integer sequences made of length of 30 are generated from the data MIDI. Each integer indicates the index of encoding dictionary. You can check the first sequence of dataset as follow. However, you don't have to know about its meaning for this homework.
```
$ python dataloader.py
tensor([78, 18, 10, 61, 64, 41, 26, 41, 77, 32, 46, 47, 37, 38, 26, 15, 59, 48,
         3, 55, 16, 55, 74,  5, 52, 29, 67, 77, 46, 12])

```

If you are curious about the encoding, you can get some hints by looking at the dictionaries (uncomment codes in dataloader). 
```
$ python dataloader.py
...
 0: C,0.250,<d1,P-5>
 1: C,0.333,<A4,d-2>
 2: A,0.250,<m2,P-4>
 3: S,0.250,<A4,d-2>
 4: S,0.333,<d1,P-5>
 5: C,0.250,<P4,m-2>
 ...
```

## Requirements 
This hoemwork requires the following software modules. You can install them with pip.

* Python 3.7 (recommended)
* Numpy
* music21
* PyTorch 1.0
* tqdm


```
$ python -m pip install music21
$ python -m pip install tqdm
```

## Skeleton Code
The given source code is not completed yet; Your mission is to fill in the blanks, marked as "None" followed by # TODO comment

There are bunch of codes, but you only have to consider following Python files.
* dataloader.py: loads and make dataset.
* generate.py: generate the feature sequence with the trained model, and make a MIDI file. You have to complete "generate_sequence" function
* rnn.py: make RNN model module. You have to complete "Baseline" class
* train.py: trains the model. You have to complete training iteration part.

After you fill in the all blanks in rnn.py and train.py, you can check it by running trian.py


```
$ python train.py
...
Blah
```

## Listening to the Music
In order to listen to the music, you should synthesize sound from the MIDI. The followings are the options on different OS platforms: 

* Windows Media player (Windows only)
* Garageband (MacOS only)
* [FluidSynth](http://www.fluidsynth.org/): software library (Windows, MacOS, Linux)
* [Timidity++](http://timidity.sourceforge.net/): software library (Windows, MacOS, Linux)


## Deliverables
* The completed source code files
* The generated MIDI file


## References
The homework was based on the following examples and we heaviliy used the existing source code.   

* [coursera homework](https://www.coursera.org/learn/nlp-sequence-models/home/welcome)
* [deepjazz](https://github.com/evancchow/jazzml)
* [jazzml](https://github.com/evancchow/jazzml)


