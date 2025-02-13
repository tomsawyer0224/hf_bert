import yaml
import argparse
import pandas as pd
from IPython.display import HTML, display
from transformers import TrainingArguments, Trainer, DefaultDataCollator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import data
import model


class Training:
    def __init__(self, config_file):
        with open(config_file, "r") as file:
            configs = yaml.safe_load(file)
        self.configs = configs

    def train_bert_cls(
        self, load_from_checkpoint=None, resume_from_checkpoint=None, num_train_epochs=3
    ):

        # datasets infor
        dataset_infor = self.configs["dataset_infor"]
        data_files = self.configs["data_files"]

        class_names = self.configs["dataset_infor"]["class_names"]

        # model infor
        model_configs = self.configs["model_configs"]

        pre_trained = model_configs["pre_trained"]
        num_labels = model_configs["num_labels"]

        # training args
        training_configs = self.configs["training_configs"]

        # prepare datasets:
        imdb_cls = data.DatasetForBertCLS(
            dataset_infor=dataset_infor, data_files=data_files, pre_trained=pre_trained
        )
        train_dataset, val_dataset, test_dataset = imdb_cls.prepare_datasets()

        # create model
        id2label = {k: v for k, v in enumerate(class_names)}
        label2id = {k: v for v, k in enumerate(class_names)}

        bert_cls = model.HFBertForCLS(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            pre_trained=pre_trained,
        )

        if load_from_checkpoint:
            bert_cls.load_checkpoint(load_from_checkpoint)

        # train and test the model
        training_args = TrainingArguments(
            **training_configs, num_train_epochs=num_train_epochs
        )

        trainer = Trainer(
            model=bert_cls.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=bert_cls.compute_metrics,
        )
        # ---training phase
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        # ---testing phase
        test_results = trainer.predict(test_dataset=test_dataset)
        test_metrics = pd.DataFrame(
            test_results.metrics, index=[0], columns=["test_loss", "test_accuracy"]
        )
        test_metrics.rename(
            columns={"test_loss": "Test Loss", "test_accuracy": "Test Accuracy"},
            inplace=True,
        )
        display(HTML(test_metrics.to_html(index=False)))

    def train_bert_qa(
        self, load_from_checkpoint=None, resume_from_checkpoint=None, num_train_epochs=3
    ):
        model_config = self.configs["model_config"]

        dataset_config = self.configs["dataset"]

        training_config = self.configs["training_config"]

        checkpoint = model_config["checkpoint"]
        dataset_name = dataset_config["dataset_name"]
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # dataset
        dataset_for_bert_qa = data.DatasetForBertQA(
            name=dataset_name, tokenizer=tokenizer
        )
        train_dataset, val_dataset = dataset_for_bert_qa.prepare_datasets()
        # model
        bert_for_qa = model.HFBertForQA(checkpoint)
        if load_from_checkpoint:
            bert_for_qa.load_checkpoint(load_from_checkpoint)
        # train
        data_collator = DefaultDataCollator()

        training_args = TrainingArguments(
            **training_config, num_train_epochs=num_train_epochs
        )
        trainer = Trainer(
            model=bert_for_qa.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_type", default="cls")
    parser.add_argument("--config_file", default="config/cls_configs.yaml")
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--load_from_checkpoint", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

    args = parser.parse_args()

    training_bert = Training(config_file=args.config_file)
    if args.bert_type == "cls":
        training_bert.train_bert_cls(
            load_from_checkpoint=args.load_from_checkpoint,
            resume_from_checkpoint=args.resume_from_checkpoint,
            num_train_epochs=args.num_train_epochs,
        )
    if args.bert_type == "qa":
        training_bert.train_bert_qa(
            load_from_checkpoint=args.load_from_checkpoint,
            resume_from_checkpoint=args.resume_from_checkpoint,
            num_train_epochs=args.num_train_epochs,
        )
