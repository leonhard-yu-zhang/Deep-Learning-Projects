# Project 1 Diabetic retinopathy recognition
Binary classification, VGG-like model
## 1 How to run the code

-------------------------------------------------------------------------------------------------------
### 1.1 Traning or evaluation
First download and unzip the IDRID_dataset:
https://drive.google.com/file/d/1PMb1dEkIb_2Hmf9uxs3PNaI0QeiCrsCG/view?usp=sharing.
Run main.py for: (1)  train; (2) train_and_evaluate; (3) evaluate.<br>
Please specify the parent dir of "/IDRID_dataset", e.g. "/home/data".<br>
Change the configuration parameters of main.py when:

#### 1.1.1 only train:

`--train_or_evaluate==train --data_dir==/home/data --n_classes==2 --model_name==vgg_like`

#### 1.1.2 train and evaluate:

`--train_or_evaluate==train_and_evaluate --data_dir==/home/data --n_classes==2 --model_name==vgg_like`

#### 1.1.3 only evaluate:

`--train_or_evaluate==evaluate --data_dir==/home/data --n_classes==2 --model_name==vgg_like`

First download and unzip the saved checkpoints for diabetic retinopathy:
https://drive.google.com/file/d/12z3SFmeP5bAFwyIH2z6a6cbPVOC2iit9/view?usp=sharing.

Then follow the prompt on the screen, specify the path of:<br>
(1) config_operative.gin, e.g.<br>
"xxx/diabetic_retinopathy_checkpoints_for_vgg_model/1_tuned_vgg_model/config_operative.gin";<br>
(2) ckpts/train, e.g.<br>
"xxx/diabetic_retinopathy_checkpoints_for_vgg_model/0_initial_vgg_model/ckpts/train".<br>
***make sure they match the model you've chosen***

### 1.2 Hyperparameter tuning <br> 
Run tune_with_ray.py or tune_with_wandb.py

## 2 Results

-------------------------------------------------------------------------------------------------------
See the paper of diabetic retinopathy recognition and the directory of checkpoints for more details.

After hyperparameter tuning for VGG-like model, the best two results of test accuracy on the test set:

|                                         |     checkpoint 1     |     checkpoint 2      |
|:---------------------------------------:|:--------------------:|:---------------------:|
|              Learning rate              |       1.06e-4        |        1.27e-4        |
|               Total steps               |         2e4          |          2e4          |
|               Batch size                |          32          |          16           |
|            Number of filters            |          16          |           8           |
|          Number of dense units          |         128          |          128          |
|              Dropout rate               |         0.67         |         0.79          |
|          Number of VGG-blocks           |          5           |           5           |
|        Total parameters of model        |       1188818        |         30626         |
|        Best validation accuracy         |        92.77%        |        96.39%         |
| Confusion matrix for the validation set | [[25 5]<br/>[1 52]]  |  [[29 1]<br/>[2 51]]  |
|              Test accuracy              |        71.84%        |        73.79%         |
|    Confusion matrix for the test set    | [[18 21]<br/>[8 56]] | [[23 16]<br/>[11 53]] |
|                Precision                |     [0.69 0.73]      |      [0.68 0.77]      |
|               Sensitivity               |     [0.46 0.88]      |      [0.59 0.83]      |
|                F1-Score                 |     [0.55 0.79]      |      [0.63 0.80]      |

Deep visualization using Grad-CAM and guided Grad-GAM shows learnt features:

# Project 2 Human activity recognition
6-classification, Sequence LSTM/GRU model with GlobalMaxPooling/GlobalAveragePooling layer
## 1 How to run the code

-------------------------------------------------------------------------------------------------------
### 1.1 Training or evaluation
First download and unzip the HAPT_dataset:
https://drive.google.com/file/d/14d5AaALymfPJVcQJl4ag35LmNxOk9woa/view?usp=sharing.
Run main.py for: (1)  train; (2) train_and_evaluate; (3) evaluate.<br>
Please specify the parent dir of "/HAPT_dataset", e.g. "/home/data".<br>
Change the configuration parameters of main.py when:

#### 1.1.1 only train:

`--train_or_evaluate==train --data_dir==/home/data --n_classes==6 --model_name==sequence_GRU_model_GlobalMaxPooling`

#### 1.1.2 train and evaluate:

`--train_or_evaluate==train_and_evaluate --data_dir==/home/data --n_classes==6 --model_name==sequence_GRU_model_GlobalMaxPooling`

#### 1.1.3 only evaluate:

`--train_or_evaluate == evaluate --data_dir==/home/data --n_classes==6 --model_name==sequence_GRU_model_GlobalMaxPooling`

First download and unzip the saved checkpoints for human activity recognition:
https://drive.google.com/file/d/1rNlviSMxZtCZJDVWwLx0E-7OJpNsLIf5/view?usp=sharing.

Then follow the prompt on the screen, specify the path of:<br>
(1) config_operative.gin, e.g.<br>
"xxx/human_activity_recognition_checkpoints_for_different_models/MAX_GRU/config_operative.gin";<br>
(2) ckpts/train, e.g.<br>
"xxx/human_activity_recognition_checkpoints_for_different_models/MAX_GRU/ckpts/train".<br>
***make sure they match the model you've chosen***

### 1.2 Hyperparameter tuning <br> 
Run tune.py

## 2 Results

-------------------------------------------------------------------------------------------------------
See the presentation slides, the poster of human activity recognition and the directory of checkpoints for more details.

Hyperparameter tuning for LSTM and GRU model:

|                                   |          1           |          2          |
|:---------------------------------:|:--------------------:|:-------------------:|
|            Block type             |         GRU          |        LSTM         |
|           Pooling type            |   GlobalMaxPooling   |  GlobalMaxPooling   |
|           Learning rate           |       6.58e-5        |       5.40e-5       |
|           Dropout rate            |         0.44         |        0.31         |
|           Val accuracy            |        94.05%        |       93.83%        |

GRU model with GlobalMaxPooling layer has the best test accuracy: 98.3%.

Visualization the ground truth and prediction:

Confusion matrix:


