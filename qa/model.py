from transformers import AutoModelForQuestionAnswering

    
class QA_model():
    
    def __init__(self,model_checkpoint):
        self.model_checkpoint=model_checkpoint
        
    
    def get_model(self):
        model=AutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint)
        return model