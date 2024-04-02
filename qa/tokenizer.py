from transformers import AutoTokenizer


class QA_tokenizer:
    def __init__(self,model_checkpoint):
        self.model_chekpoint=model_checkpoint
        
    def get_tokenizer(self):
        tokenizer=AutoTokenizer.from_pretrained(self.model_chekpoint)
        
        return tokenizer