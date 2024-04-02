import torch
from transformers import AutoModelForTokenClassification
from dataset import CNER_Dataset

class CNER_model():
    
    def __init__(self,model_checkpoint):
        self.model_checpoint=model_checkpoint
    
    def get_model(self,id2label,label2id):
        
        return AutoModelForTokenClassification.from_pretrained(
            self.model_checpoint,
            id2label=id2label,
            label2id=label2id,
        )
                