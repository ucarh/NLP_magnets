from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer,PunktTrainer
from tqdm.auto import tqdm


class Sentence_dataset():
    def __init__(self,classified_chkpt,magnetics_ds_chkpt):
        
        self.classified_chkpt=classified_chkpt
        self.magnetics_ds_chkpt=magnetics_ds_chkpt
        
        self.tokenizer=self.train_get_sent_tokenizer()
        
        
    def get_magnetics_ds(self):
        
        return load_dataset(self.magnetics_ds_chkpt,split='train')
        
    def get_classified_ds(self):
        
        ds=load_dataset(self.classified_chkpt,split='train')
        ds=ds.filter(lambda x:x['label']==1)
        
        return ds

    def train_get_sent_tokenizer(self):
        
        magnetics_ds=self.get_magnetics_ds()
        subset_magnetics=magnetics_ds[:1000]
        
        progress_bar = tqdm(range(len(subset_magnetics['text'])))
        trainer=PunktTrainer()
        for text in subset_magnetics['text']:
            trainer.train(text)
            trainer.freq_threshold()
            progress_bar.update(1)
        trainer.finalize_training()
        tokenizer=PunktSentenceTokenizer(trainer.get_params())
        
        return tokenizer
    
    
    def tokenize(self,examples):
        return {'sentences':self.tokenizer.tokenize(examples['abs_text'])}
        
    def combine_abs_text(self,example):
        return {"abs_text":example["abstract"]+" "+example["text"]}

    def process_classified_ds(self,batch):
        classified_ds=self.get_classified_ds()
        classified_ds=classified_ds.shuffle(seed=42)
        if batch<8:
            classified_ds=classified_ds.select(range((batch-1)*10000,batch*10000))
        elif batch==8:
            classified_ds=classified_ds.select(range((batch-1)*10000,len(classified_ds)))
            
        
        classified_ds=classified_ds.map(self.combine_abs_text)

        classified_ds=classified_ds.map(self.tokenize)
        classified_ds=classified_ds.map(
            lambda x:{'sentences': [sentence+" "+"no-answer" for sentence in x['sentences']]}
            )
        return classified_ds
        
        
