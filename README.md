# 11-667_Mini_Project

## Directories
**Model Finetuning Scripts:** 
+ Gemma_2b_it/main.py
+ Gemma_9b_it/main.py
+ OLMo-7B-it/main.py

**Model Validation Scripts:** 
+ validate_gemma.py
+ validate_olmo.py

**Saved Finetuned Models:** 
+ out_dir_2b_gemma/saved_model
+ out_dir_new_gemma/saved_model
+ out_dir_new_olmo/saved_model
  
## Finetuning/Training
To finetune any of the models, navigate to respective model directory's **main.py**. Follow the full list of argument flags specified in file. Here is a sample command:

`pip install -r requirements.txt`

`cd Gemma_2b_it`

`python main.py --mode train --train_csv ../train/train.csv`


## Validation
To validate any of the saved models, run **validate_[model].py**. Follow the full list of argument flags specified in file. Here is a sample command:

`pip install -r requirements.txt`

`python validate_gemma.py --val_csv train/train.csv --model_name out_dir_2b_gemma/saved_model/ --metric perplexity`

