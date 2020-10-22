# Human Activity Recognition
Human Activity Recognition (HAR) based on sensory data is an active research field of classifying data recorded by sensors into known well-defined movements. A challenging problem is dealing with the large number of time-series data and finding a clear way to match data to corresponding movements. We propose a generic framework to predict sequences of human activities from sequences of inertial sensor data. 

Please check further details in ```HAPT_paper.pdf```.
**Code is hidden. Preventing some new student from simply copying and pasting. **

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

## Implementation details
**Optimizer**   Adam
**Metrics**  Confusion matrix and accuracy
**Loss** Cross-entropy loss
**Training Procedure** Since there is some unlabeled data in our dataset, we manually annotate the labels of these data
as all zero vectors. And we use ont-hot coding of labeled data in our training procedure. The cross-entropy loss of zero vector will be zero. So unlabeled data won’t have generate any loss. To improve the performance of our network, we use drop out in GRU layers and fully-connected layers. And we set an
unified dropout rate as a hyperparameter. We will try our network for 5 epochs and test our network after 5 epochs of
training.
##  Hyperparameter tuning
Manually selecting hyperparameters strongly depends on human’s experience. Grid search is time-consuming and not
feasible. After consideration, we take Bayesian Optimization as our method of hyperparameter tuning. We have the
following hyperparameters and their range:
* Unit of GRU layers from 16 to 64
* Unit of the first fully-connected layer from 16 to 64
* Drop out rate of GRU layers and fully-connected layers from 0.0 to 0.4
* Learning rate of optimizer from 1e-2 to 1e-5
For convenience, the first GRU layer always have two times so many units as the second GRU layer, and first dense
layer have same number of neurons as the second GRU layer. And we select 5 exploits and 5 explores (in total 10
iterations) as the hyperparameter of our Bayesian Optimization.


## Dependancy
Project is created with:
- Python 3.6
- Tensorflow 2.0.0

## Usage
**Code is hidden.**
~~Unzip hapt_tfrecords.7z in a folder called hapt_tfrecords in the root directory of HAPT. Manipulate ```confin.gin``` to switch from different modes. And use```python3 main.py``` to start the program.~~

### Tuning Mode
**Code is hidden.**
~~Under this mode, no checkpoint will be saved and nothing will be visualized. And Bayesian Optimization will be executed.~~
```
main.tuning = True

main.unit = 48 #this hyperparameter will not be executed.
main.learning_rate = 0.0012 #this hyperparameter will not be executed.
main.dropout = 0.08 #this hyperparameter will not be executed.
main.num_epoch = 5 #the epochs you want to run#
```

### Non-tuning Mode
**Code is hidden.**
~~Under this mode, you will have control of the hyperparameters for a single run. Checkpoints will be saved for every 5 epochs. If there is previous checkpoint, it'll be restored automatically. A random segment of sequence in test set will be visualized.~~
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
### Bayesian Optimization
| **Iteration** |**Dropout**|**Learning Rate**|**Unit**|**Test Accuracy**|
|:------:|:------:|:------:|:------:|:------:|
|GRU-1| 0.1668| 0.002804 |30 |91.10%|
|GRU-2|  0.0587| 0.009078| 32 |54.5%|
|GRU-3|0.1587 |0.004617 |48 |69.26%|
|GRU-4| 0.08178 |0.001228| 48| **92.40%**|
|GRU-5| 0.1669 |0.004419 |25 |88.64%|
|GRU-6| 0.1669 |1e-05 |64 |80.79%|
|GRU-7| 0.03255 |1e-05| 16 |55.6%|
|GRU-8|0.3989 |1e-05 |41 |71.3%|
|GRU-9| 0.000896| 1e-05 |58 |78.82%|
|GRU-10| 0.4 |1e-05 |20 |51.98%|
|LSTM-1|  0.08178 |0.001228 |48 |**92.07%**|

With same hyperparameters of our GRU network, we try replacing GRU layer in our network with LSTM layer. The
result in table 2 shows that LSTM network does have similar performance as GRU network. But we did observe slower
convergence during the training, which will be more serious when we train network on a large data set.
### Confusion matrix
![Conmat](https://github.com/LEGO999/Human-Activaity-Recognition-HAPT/blob/master/ConMat.PNG)  
Final accuracy on test set: 92.4%.
The result shows that the our network is effective to predict the static and dynamic activities but not to postural
transitions between the static activities. In future we could work on that using e.g. oversampling or using weighted loss for different classes.
### Visualization
For two random segments:
![visual1](https://github.com/LEGO999/Human-Activaity-Recognition-HAPT/blob/master/visualization/20200211-112221.png)
![visual2](https://github.com/LEGO999/Human-Activaity-Recognition-HAPT/blob/master/visualization/20200210-213757.png)
## Authors
- CAO Shijia, https://github.com/scarlettcao
- ZHONG Liangyu, https://github.com/LEGO999
