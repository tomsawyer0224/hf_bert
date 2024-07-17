# This is a personal project, for educational purposes only!
# About this project:
1. This project provides an easy way to fine-tune the BERT model in Hugging Face library on two tasks: Classification and Question Answering.
2. You can fine-tune the BERT model on many datasets with only one command.
# How to use:
1. Clone this repo, then cd to hf_bert.
2. Install the requirements: pip install -q -r requirements.txt
3. Modify the config file (for example, ./config/cls_config.yaml), then run the below command:
```
python train.py \
  --bert_type 'cls' \ # or 'qa'
  --config_file './config/cls_config.yaml' \
  --num_train_epochs 3 \
  --resume_from_checkpoint 'path/to/checkpoint' # add this line when resume the training from a checkpoint
```
> The checkpoints will be save to the 'output_dir' directory that is defined in the config file.
