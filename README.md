# Project 1 Diabetic retinopathy recognition
Binary classification, VGG-like model
## 1 How to run the code

-------------------------------------------------------------------------------------------------------
### 1.1 Traning or evaluation
Run main.py for: (1)  train; (2) train_and_evaluate; (3) evaluate.<br>
Please specify the parent dir of "/IDRID_dataset", e.g. "/home/data".<br>
Change the configuration parameters of main.py when:

#### 1.1.1 only train:

`--train_or_evaluate==train --data_dir==/home/data --n_classes==2 --model_name==vgg_like`

#### 1.1.2 train and evaluate:

`--train_or_evaluate==train_and_evaluate --data_dir==/home/data --n_classes==2 --model_name==vgg_like`

#### 1.1.3 only evaluate:

`--train_or_evaluate==evaluate --data_dir==/home/data --n_classes==2 --model_name==vgg_like`

Then follow the prompt on the screen, specify the path of:<br>
(1) config_operative.gin, e.g.<br>
"xxx/diabetic_retinopathy/checkpoints_for_vgg_model/1_tuned_vgg_model/config_operative.gin";<br>
(2) ckpts/train, e.g.<br>
"xxx/diabetic_retinopathy/checkpoints_for_vgg_model/0_initial_vgg_model/ckpts/train".<br>
***make sure they match the model you've chosen***

### 1.2 Hyperparameter tuning <br> 
Run tune_with_ray.py or tune_with_wandb.py

## 2 Results

-------------------------------------------------------------------------------------------------------
See the paper of diabetic retinopathy recognition and the directory of checkpoints for more details.


# Project 2 Human activity recognition
6-classification, Sequence LSTM/GRU model with GlobalMaxPooling/GlobalAveragePooling layer
## 1 How to run the code

-------------------------------------------------------------------------------------------------------
### 1.1 Traning or evaluation
Run main.py for: (1)  train; (2) train_and_evaluate; (3) evaluate.<br>
Please specify the parent dir of "/HAPT_dataset", e.g. "/home/data".<br>
Change the configuration parameters of main.py when:

#### 1.1.1 only train:

`--train_or_evaluate==train --data_dir==/home/data --n_classes==6 --model_name==sequence_GRU_model_GlobalMaxPooling`

#### 1.1.2 train and evaluate:

`--train_or_evaluate==train_and_evaluate --data_dir==/home/data --n_classes==6 --model_name==sequence_GRU_model_GlobalMaxPooling`

#### 1.1.3 only evaluate:

`--train_or_evaluate == evaluate --data_dir==/home/data --n_classes==6 --model_name==sequence_GRU_model_GlobalMaxPooling`

Then follow the prompt on the screen, specify the path of:<br>
(1) config_operative.gin, e.g.<br>
"xxx/human_activity_recognition/checkpoints_for_different_models/MAX_GRU/config_operative.gin";<br>
(2) ckpts/train, e.g.<br>
"xxx/human_activity_recognition/checkpoints_for_different_models/MAX_GRU/ckpts/train".<br>
***make sure they match the model you've chosen***

### 1.2 Hyperparameter tuning <br> 
Run tune.py

## 2 Results

-------------------------------------------------------------------------------------------------------
See the presentation slides, the poster of human activity recognition and the directory of checkpoints for more details.
