# Homework #2: Genre Classification
Genre classification is an important task that can be used many music applications. Your mission is to build your own Convolutional Neural Network (CNN) model to classify audio files into different music genres. Specifically, the goals of this homework are as follows:

* Experiencing the whole pipeline of deep learning based system: data preparation, feature extraction, model training and evaluation
* Getting familiar with the CNN architectures for music classification tasks
* Using Pytorch in practice

## Dataset
We use the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset which has been the most widely used in the music genre classification task. The dataset contains 30-second audio files including 10 different genres including reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. For this homework, we are going to use a subset of GTZAN with only 8 genres. You can download the subset from [this link]().


Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  

```
$ cd gtzan
$ ls 
blues disco metal ... rock train_filtered.txt valid_filtered.txt test_filtered.txt
$ cd ..      # go back to your home folder for next steps
```

## Baseline Code
Coming soon...



## Improving Algorithms
Now it is your turn. You should improve the baseline code with your own algorithm. There are many ways to improve it. The followings are possible ideas: 

* The first thing to do is to segment audio clips and generate more data. The baseline code utilizes the whole mel-spectrogram as an input to the network (e.g. 128x1287 dimensions). Try to make the network input between 3-5 seconds segment and average the predictions of the segmentations for an audio clip.
* You can try 1D CNN or 2D CNN models and choose different model parameters:
    * Filter size
    * Pooling size
    * Stride size 
    * Number of filters
    * Model depth
    * Regularization: L2/L1 and Dropout

* You should try different hyperparameters to train the model and optimizers:
    * Learning rate
    * Patience value
    * Decreasing factor of learning rate 
    * Minibatch size
    * Model depth
    * Optimizers: SGD (with Nesterov momentum), Adam, RMSProp, ...

* You can try different parameters (e.g. hop and window size) to extract mel-spectrogram or different features as input to the network (e.g. MFCC, chroma features ...). 

* You can also use ResNet or other CNN with skip connections. 

* Furthermore, you can augment data using audio effects.


## Deliverables
You should submit your Python code (.py files) and homework report (.pdf file) to KLMS. The report should include:
* Algorithm Description
* Experiments and Results
* Discussion

## Notes
* You can you merge training and validation sets into a single training set (as in HW1). However, you should report both validation and test accuracy to prove that you chose the best model without using the test set.  

