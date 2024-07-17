import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM
import evaluate

class HFBertForCLS:
    def __init__(self, num_labels, id2label, label2id, pre_trained = 'bert-base-cased'):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pre_trained,
            num_labels = num_labels,
            id2label = id2label,
            label2id = label2id
        )
    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    def load_checkpoint(self, checkpoint):
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

class HFBertForMLM:
    def __init__(self, pre_trained):
        self.model = AutoModelForMaskedLM.from_pretrained(pre_trained)

class HFBertForQA:
    def __init__(self, checkpoint):
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            checkpoint
        )
    def load_checkpoint(self, checkpoint):
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            checkpoint
        )
    


