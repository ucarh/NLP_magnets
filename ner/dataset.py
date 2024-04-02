from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification


class CNER_Dataset():
    def __init__(self,dataset_name,model_checkpoint,batch_size):
        self.dataset_name=dataset_name
        self.model_checkpoint=model_checkpoint
        self.batch_size=batch_size


    def get_raw_data(self):
        raw_ds=load_dataset(self.dataset_name)
        raw_ds=raw_ds.filter(lambda x:len(x['words'])>0)
        raw_ds=raw_ds.filter(lambda x:len(x['words'])==len(x['labels']))
        
        return raw_ds
    
    def get_tokenizer(self):
        tokenizer=AutoTokenizer.from_pretrained(self.model_checkpoint,model_max_length=512)
     
        return tokenizer
    
    def prepare_train_eval_ds(self):
        raw_ds=self.get_raw_data()
        
        tokenized_datasets=raw_ds.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_ds['train'].column_names,
        )
        
        return tokenized_datasets
    
    def get_train_eval_loader(self):
        tokenized_ds=self.prepare_train_eval_ds()
        data_collator = DataCollatorForTokenClassification(tokenizer=self.get_tokenizer())
        train_dataloader = DataLoader(
            tokenized_ds["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.batch_size,
        )
        eval_dataloader = DataLoader(
            tokenized_ds["test"], collate_fn=data_collator, batch_size=self.batch_size
        )
            
        return train_dataloader,eval_dataloader
        
    
    def get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    
    def b_to_i_label_fn(self,label_list):
        b_to_i_label=[]
        for idx,label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-","I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-","I-")))
            else:
                b_to_i_label.append(idx)
        return b_to_i_label
    
    def tokenize_and_align_labels(self,examples):
        
        label_list=self.get_label_list(examples['labels'])
        label_to_id={l:i for i,l in enumerate(label_list)}
        num_labels=len(label_list)
        b_to_i_label=self.b_to_i_label_fn(label_list)
        
        tokenizer=self.get_tokenizer()
        tokenized_inputs=tokenizer(
            examples['words'],
            truncation=True,
            max_length=512,
            is_split_into_words=True,
        )
        
        labels=[]
        for i,label in enumerate(examples['labels']):
            word_ids=tokenized_inputs.word_ids(i)
            previous_word_idx=None
            label_ids=[]
            
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id!=previous_word_idx:
                    label_ids.append(label_to_id[label[word_id]])
                else:
                    label_ids.append(b_to_i_label[label_to_id[label[word_id]]])
                
                previous_word_idx=word_id
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"]=labels
        return tokenized_inputs