from functools import partial
from datasets import load_dataset, ClassLabel, Dataset
from transformers import AutoTokenizer

# base dataset class
class DatasetForBert:
    def __init__(self, dataset_infor, data_files):
        '''
        args:
            dataset_infor: dict { # in case of classification, different for other task
                'file_type': 'csv' ('text') # str: file type (.csv or .txt)
                'text_column_name': 'review', # str: name of text column
                'label_column_name': 'sentiment', # str: name of label column
                'class_names': ['negative', 'positive'] # list of label names, can be None
            }
            data_files: dict {
                'train': path/to/train.csv (.txt)
                'val': path/to/val.csv (.txt) # can be None
                'test': path/to/test.csv (.txt) # can be None
            }
        '''
        assert len(data_files) > 0, "OOP! 'data_files' must contain at least 1 file"
        self.dataset_infor = dataset_infor
        self.data_files = data_files
    def generate_datasets(self):
        '''
        method to return 3 datasets: train, val, test
        '''
        train_file = self.data_files['train']
        val_file = self.data_files['val']
        test_file = self.data_files['test']
        has_val, has_test = False, False
        raw_dataset = load_dataset(
            self.dataset_infor['file_type'],
            data_files = train_file,
            split = 'train'
        )
        if val_file:
            val_dataset = load_dataset(
                self.dataset_infor['file_type'],
                data_files = val_file,
                split = 'train'
            ).shuffle(seed = 42)
            has_val = True
        if test_file:
            test_dataset = load_dataset(
                self.dataset_infor['file_type'],
                data_files = test_file,
                split = 'train'
            ).shuffle(seed = 42)
            has_test = True
        if has_val and has_test:
            train_dataset = raw_dataset.shuffle(seed = 42)

        elif has_val and not has_test:
            train_test = raw_dataset.train_test_split(
                test_size = 0.2,
                seed = 42
            )
            train_dataset = train_test['train']
            test_dataset = train_test['test']
            
        elif not has_val and has_test:
            train_val = raw_dataset.train_test_split(
                test_size = 0.2,
                seed = 42
            )
            train_dataset = train_val['train']
            val_dataset = train_val['test']
        else: #not has_val and not has_test
            trainval_test = raw_dataset.train_test_split(
                test_size = 0.15,
                seed = 42
            )
            trainval = trainval_test['train']
            test_dataset = trainval_test['test']

            train_val = trainval.train_test_split(
                test_size = 0.2,
                seed = 42
            )
            train_dataset = train_val['train']
            val_dataset = train_val['test']
        return train_dataset, val_dataset, test_dataset

class DatasetForBertCLS(DatasetForBert):
    def __init__(self, dataset_infor, data_files, pre_trained = "bert-base-cased"):
        '''
        args:
            dataset_infor: dict {
                'file_type': 'csv' ('text') # str: file type (.csv or .txt)
                'text_column_name': 'review', # str: name of text column
                'label_column_name': 'sentiment', # str: name of label column
                'class_names': ['negative', 'positive'] # list of label names, can be None
            }
            data_files: dict {
                'train': path/to/train.csv (.txt)
                'val': path/to/val.csv (.txt) # can be None
                'test': path/to/test.csv (.txt) # can be None
            }
        '''

        super().__init__(dataset_infor = dataset_infor, data_files = data_files)
        #self.tokenizer = AutoTokenizer.from_pretrained(pre_trained, use_fast = False)
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained)

        assert dataset_infor['class_names'] is not None, 'must provide class names'
        self.class_label_obj = ClassLabel(
            num_classes = len(dataset_infor['class_names']), 
            names = dataset_infor['class_names']
        )

    def map_label_str2int(self, dataset):
        label_column_name = self.dataset_infor['label_column_name']
        label_column_id = label_column_name + "_id"
        def label_name2id(example):
            example[label_column_id] = self.class_label_obj.str2int(
                example[label_column_name]
            )
            return example
        dataset = dataset.map(label_name2id)
        new_features = dataset.features.copy()
        new_features[label_column_id] = self.class_label_obj
        dataset = dataset.cast(new_features)
        return dataset

    def prepare_datasets(self):
        train_dataset, val_dataset, test_dataset = self.generate_datasets()

        #add new column (label id)
        train_dataset = self.map_label_str2int(train_dataset)
        val_dataset = self.map_label_str2int(val_dataset)
        test_dataset = self.map_label_str2int(test_dataset)

        #rename column to match hugging face form
        old_to_new = {
            self.dataset_infor['text_column_name'] : 'text',
            self.dataset_infor['label_column_name'] + '_id' : 'label'
        }
        train_dataset = train_dataset.rename_columns(old_to_new)
        val_dataset = val_dataset.rename_columns(old_to_new)
        test_dataset = test_dataset.rename_columns(old_to_new)

        #tokenize function
        tokenize = lambda example: self.tokenizer(example["text"], padding="max_length", truncation=True)
        train_dataset = train_dataset.map(tokenize, batched = True)
        val_dataset = val_dataset.map(tokenize, batched = True)
        test_dataset = test_dataset.map(tokenize, batched = True)

        return train_dataset, val_dataset, test_dataset

class DatasetForBertQA:
    def __init__(self, name, tokenizer):
        self.name = name
        self.train_dataset = load_dataset(name, split = 'train')
        self.val_dataset = load_dataset(name, split = 'validation')
        self.tokenizer = tokenizer
    def generate_datasets(self):
        raw_dataset = load_dataset(self.name)
        train_dataset = raw_dataset['train']
        val_dataset = raw_dataset['validation']
        return train_dataset, val_dataset
    @staticmethod
    def preprocess_training_examples(examples, tokenizer, max_length = 384, stride = 128):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
                questions,
                examples["context"],
                max_length=max_length,
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    @staticmethod
    def preprocess_validation_examples(examples, tokenizer, max_length = 384, stride = 128):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    def prepare_datasets(self):
        train_dataset = self.train_dataset.map(
            partial(
                self.preprocess_training_examples,
                tokenizer = self.tokenizer,
                max_length = 384,
                stride = 128
            ),
            batched = True,
            remove_columns = self.train_dataset.column_names
        )
        val_dataset = self.val_dataset.map(
            partial(
                self.preprocess_training_examples,
                tokenizer = self.tokenizer,
                max_length = 384,
                stride = 128
            ),
            batched = True,
            remove_columns = self.val_dataset.column_names
        )
        return train_dataset, val_dataset
    


