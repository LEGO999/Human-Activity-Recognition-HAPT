# Human Activity Recognition
Human Activity Recognition (HAR) based on sensory data is an active research field of classifying data recorded by sensors into known well-defined movements. A challenging problem is dealing with the large number of time-series data and finding a clear way to match data to corresponding movements. We propose a generic framework to predict sequences of human activities from sequences of inertial sensor data. 

Please check further details in ```HAPT_paper.pdf```.

## Dataset
Human Activities and Postural Transitions Dataset (HAPT)
HAPT dataset contains data from tri-axial accelerometer and gyroscope of a smartphone, both are captured at a
frequency of 50 Hz. The dataset consists of six basic activities (static and dynamic) and six postural transitions between
the static activities. The static activities include standing, sitting and lying. The dynamic activities include walking,
walking downstairs and walking upstairs. stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie, and lie-to-stand are
the classes for postural transitions between the static activities. We split the dataset into train/validation/test datasets in
proportion 0.7/0.1/0.2.

## Input Pipeline
* **Input Normalization**
We use Z-Score normalization to transform input data to zero mean and unit variance. Normalized 6-channels time-series data could give us better performance because it could eliminate user dependent biases and offset due to device placement errors.
* **Serialization**
To load data efficiently, we serialize the 6-channel input data and corresponding labels and store them in
in TFRecord format.
* **Sliding Window**
In order to have fixed-length input and label sequences, we apply sliding window technique. In
training we set window length 250 and window shift 125, that means 50% overlap, which serves as a kind of data
augmentation. In validation and test set, the window length is set to 250 without overlap.  

## Model
For our time series signal, we need a model which could dual with time series signal. Typically, vanilla Recurrent Neural Network(RNN) and its variants are used in this scenario.  
A typical vanilla RNN could have not bad performance for a normal application. However, when there is a large lag between input sequence signal and the corresponding teacher signal, RNN is reluctant to capture such kind of relation. The reason behind is that, using conventional Back-Propagation Through Time(BPTT), the error signal in RNN’s
training is easy to explode or vanish. Remedy could be using other models like Long Short-term Memory(LSTM) network and Gated Recurrent Unit(GRU) network.

LSTM and GRU are capable of capturing the long-term dependencies of time series signal. Because they could
selectively forget and remember some information regarding the new input. This mechanism is called gating. There are
three different gates in LSTM which helps to manipulate the information states.
* Forget gate layer: decides which part of information is unnecessary in cell state and abandon it.
* Input gate layer: control what kind of new information should be added in cell state.
* Output gate: filter and output the cell state.
GRU is a variant of LSTM. There are just two gates in GRU, update gate and reset gate. That means, GRU will pass all
content of its cell state to the next neuron. In general, GRU and LSTM have similar performance. But since GRU has
fewer parameters than LSTM, which will make it converge faster and make it less sensitive to overfitting.  

| architecture |
|:------:|
|GRU(unit=32)|
|GRU(unit=16)|
|Dense(unit=16), ReLU|
|Dense(unit=12), Softmax|

##


## Dependancy
Project is created with:
- Python 3.6
- Tensorflow 2.0.0

## Usage
Unzip hapt_tfrecords.7z in a folder called hapt_tfrecords in the root directory of HAPT. Manipulate ```confin.gin``` to switch from different modes. And use```python3 main.py``` to start the program.

### Tuning Mode
Under this mode, no checkpoint will be saved and nothing will be visualized. And Bayesian Optimization will be executed.
```
main.tuning = True

main.unit = 48 #this hyperparameter will not be executed.
main.learning_rate = 0.0012 #this hyperparameter will not be executed.
main.dropout = 0.08 #this hyperparameter will not be executed.
main.num_epoch = 5 #the epochs you want to run#
```

### Non-tuning Mode
Under this mode, you will have control of the hyperparameters for a single run. Checkpoints will be saved for every 5 epochs. If there is previous checkpoint, it'll be restored automatically. A random segment of sequence in test set will be visualized.
```
main.tuning = False

# all hyperparameters belows could be manipulated as you wish
main.unit = 48
main.learning_rate = 0.0012
main.dropout = 0.08
main.num_epoch = 5
```
## Used Features
* TF-Records
* GRU Network
* AutoGraph
* Bayesian Opimization
## Results
### Confusion matrix
![Conmat](https://github.com/LEGO999/Human-Activaity-Recognition-HAPT/blob/master/ConMat.PNG)  
Final accuracy on test set: 92.4%.
### Visualization
For two random segments:
![visual1](https://github.com/LEGO999/Human-Activaity-Recognition-HAPT/blob/master/visualization/20200211-112221.png)
![visual2](https://github.com/LEGO999/Human-Activaity-Recognition-HAPT/blob/master/visualization/20200210-213757.png)
## Authors
- CAO Shijia, https://github.com/scarlettcao
- ZHONG Liangyu, https://github.com/LEGO999
