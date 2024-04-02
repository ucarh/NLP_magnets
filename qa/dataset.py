import torch
from datasets import load_dataset
from tokenizer import QA_tokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator

class QA_Dataset(torch.utils.data.Dataset):
    def __init__(self,model_checkpoint,batch_size):
        self.model_checkpoint=model_checkpoint
        self.batch_size=batch_size
        
    def get_dataset(self):
        raw_dataset=load_dataset('json',data_files='Tc_ds_forQA.json')
        # raw_dataset=raw_dataset.filter(lambda x:x['answers']['answer_start'][0]!=-1)
        raw_dataset=raw_dataset.shuffle(seed=41)
        raw_dataset=raw_dataset['train'].train_test_split(train_size=0.9, seed=42)
        raw_dataset['validation']=raw_dataset.pop('test')
        return raw_dataset
    

    def get_train_ds(self):
        raw_ds=self.get_dataset()
        column_names=raw_ds["train"].column_names
        train_ds=raw_ds['train']
        train_ds=train_ds.map(
            self.prepare_train_features,
            batched=True,
            remove_columns=column_names,
        )
        train_ds.set_format("torch")
        train_dataloader=DataLoader(
            train_ds,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.batch_size
        )
        
        return train_dataloader
    
    def get_eval_ds(self):
        
        raw_ds=self.get_dataset()
        column_names=raw_ds["validation"].column_names
        eval_ds=raw_ds['validation']
        eval_ds=eval_ds.map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=column_names,
        )
        eval_set = eval_ds.remove_columns(["example_id", "offset_mapping"])
        eval_set.set_format("torch")
        eval_dataloader = DataLoader(
            eval_set,collate_fn=default_data_collator,batch_size=self.batch_size
        )
        return eval_ds,eval_dataloader
        
                              
        
        
    def prepare_train_features(self,examples,max_seq_length=384,stride=128,pad_to_max_length=True):
        
        questions=[q.strip() for q in examples["question"]]
        
        tokenizer=QA_tokenizer(self.model_checkpoint).get_tokenizer()
        
        tokenized_examples=tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,   
        )
        
        sample_mapping=tokenized_examples.pop("overflow_to_sample_mapping")
        
        offset_mapping=tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"]=[]
        tokenized_examples["end_positions"]=[]
        
        for i,offsets in enumerate(offset_mapping):
            input_ids=tokenized_examples["input_ids"][i]
            sequence_ids=tokenized_examples.sequence_ids(i)
            
            sample_index=sample_mapping[i]
            answers=examples["answers"][sample_index]
        
                
            if len(answers["answer_start"])==0:
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
                
            else:
                start_char=answers["answer_start"][0]
                end_char=start_char+len(answers["text"][0])
                
                token_start_index=0
                while sequence_ids[token_start_index]!=1:
                    token_start_index+=1
                    
                token_end_index=len(input_ids)-1
                while sequence_ids[token_end_index] !=1:
                    token_end_index-=1
                    
                if not (offsets[token_start_index][0] <=start_char and offsets[token_end_index][1]>=end_char):
                    tokenized_examples["start_positions"].append(0)
                    tokenized_examples["end_positions"].append(0)
                    
                else:
                    while token_start_index <len(offsets) and offsets[token_start_index][0]<=start_char:
                        token_start_index+=1
                    tokenized_examples["start_positions"].append(token_start_index-1)
                    while offsets[token_end_index][1]>=end_char:
                        token_end_index-=1
                    tokenized_examples["end_positions"].append(token_end_index+1)
                    
        return tokenized_examples
                    
    
    def prepare_validation_features(self,examples,max_seq_length=384,stride=128,pad_to_max_length=True):
        
        questions=[q.strip() for q in examples["question"]]
        
        tokenizer=QA_tokenizer(self.model_checkpoint).get_tokenizer()
        
        tokenized_examples=tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,   
        )
        
        sample_mapping=tokenized_examples.pop("overflow_to_sample_mapping")
        
        tokenized_examples["example_id"]=[]
        
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids=tokenized_examples.sequence_ids(i)
            context_index=1
            
            sample_index=sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            tokenized_examples["offset_mapping"][i]=[
                (o if sequence_ids[k]==context_index else None)
                for k,o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
            
        return tokenized_examples