# %%
import time
start_time = time.time()

from datasets import load_from_disk
import collections
import numpy as np
import transformers
from transformers import default_data_collator

import os
import gc
import torch
from torch.utils.data import DataLoader

# %%
# https://stackoverflow.com/questions/60993677/how-can-i-save-pytorchs-dataloader-instance
import random
from torch.utils.data.dataloader import Sampler

random.seed(0)  # use a fixed number


class MySampler(Sampler):
    def __init__(self, n, seed=0):
        self.n = n
        np.random.seed = seed  
        random.seed(seed)  # use a fixed number

        self.seq = list(range(n))
        np.random.shuffle(self.seq)

    def reset(self, seed):
        np.random.seed = seed
        random.seed(seed)  
        self.seq = list(range(self.n))
        np.random.shuffle(self.seq)


    def shrink(self,i):
        self.seq = self.seq[i:]

    def __iter__(self):         
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)


# %%
chunked_magnetics_ds=load_from_disk("/l/users/huseyin.ucar/NLP_magnets_data/chunked_ds_bert_uncased")

# %%
chunked_magnetics_ds, 1088892/32

# %%
class cfg:
    batch_size=32
    num_train_epochs = 50
    wwm_probability = 0.15
    DEBUG=False  
    
    scheduler='linear'
    model_checkpoint="bert-base-uncased"
    KERNEL_TYPE=model_checkpoint+'_model_'+scheduler+'_sched_'+str(num_train_epochs)+'_epochs_'+str(batch_size)+'_BS'
    LOG_DIR='./logs'
    DATA_DIR="/l/users/huseyin.ucar/NLP_magnets_data"
    MODEL_DIR=os.path.join(DATA_DIR,"weights")
    CHECKPOINT_DIR=os.path.join(DATA_DIR,"checkpoint")

    load_model = False
    epoch_cont=0
    load_path = model_file=os.path.join(MODEL_DIR, f'{KERNEL_TYPE}_best.pth')
    
checkpoint_file=os.path.join(cfg.CHECKPOINT_DIR, f'{cfg.KERNEL_TYPE}_last.pth')
checkpoint_file

# %%
if not os.path.exists(cfg.LOG_DIR):
    os.makedirs(cfg.LOG_DIR)

if not os.path.exists(cfg.MODEL_DIR):
    os.makedirs(cfg.MODEL_DIR)

if not os.path.exists(cfg.CHECKPOINT_DIR):
    os.makedirs(cfg.CHECKPOINT_DIR)


cfg.KERNEL_TYPE

# %%
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(cfg.model_checkpoint)

# %%
def whole_word_masking_data_collator(features):
    for feature in features:
        words_ids=feature.pop("word_ids")

        mapping=collections.defaultdict(list)
        current_word_index=-1
        current_word=None

        for idx,word_id in enumerate(words_ids):
            if word_id is not None:
                if word_id !=current_word:
                    current_word=word_id
                    current_word_index+=1
                mapping[current_word_index].append(idx)
        
        mask=np.random.binomial(1,cfg.wwm_probability,(len(mapping),))

        input_ids=feature['input_ids']
        labels=feature['labels']
        new_labels=[-100]*len(labels)

        for word_id in np.where(mask)[0]:
            word_id=word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx]=labels[idx]
                input_ids[idx]=tokenizer.mask_token_id
        feature["labels"]=new_labels

    return default_data_collator(features)


# %%
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = whole_word_masking_data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

# %%
if cfg.DEBUG:
    train_size = 500
    test_size = int(0.1 * train_size)

    chunked_magnetics_ds = chunked_magnetics_ds["train"].train_test_split(train_size=train_size, test_size=test_size, seed=42)


# %%
chunked_magnetics_ds

# %%
eval_dataset=chunked_magnetics_ds['test'].map(insert_random_mask,batched=True,num_proc=8,\
                                              remove_columns=chunked_magnetics_ds['test'].column_names)



# %%
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
        "masked_token_type_ids": "token_type_ids"
    }
)

# %%


trainSampler = MySampler(chunked_magnetics_ds['train'].num_rows)

train_dataloader=DataLoader(chunked_magnetics_ds['train'],
                            shuffle=False,
                            batch_size=cfg.batch_size,
                            sampler = trainSampler,
                            collate_fn=whole_word_masking_data_collator)

eval_dataloader=DataLoader(eval_dataset,
                           batch_size=cfg.batch_size,
                           collate_fn=default_data_collator)




# %%
from transformers import AutoModelForMaskedLM
from torch.optim import AdamW
from transformers import get_scheduler
# from tqdm.notebook import trange, tqdm
import os 
import torch
from huggingface_hub import create_repo,get_full_repo_name,Repository
from tqdm.auto import tqdm
import math


# %%
model=AutoModelForMaskedLM.from_pretrained(cfg.model_checkpoint)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = cfg.num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %%
seq = None
seq_sampler_i = None
best_score=float("inf")
currentEpoch = 0

if  os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)

     
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    lr_scheduler.load_state_dict(checkpoint['lr_sched'])


    best_score = checkpoint['best_score']
    seq =  checkpoint['seq']
    seq_sampler_i = checkpoint['seq_sampler_i']
    currentEpoch = checkpoint['currentEpoch'] 


# %%
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# %%
#uncomment below only if the repo is not present on huggingface.

#create_repo(f"nlp-magnets/{cfg.KERNEL_TYPE}",private=True,repo_type="model")

repo_name=get_full_repo_name(model_id=cfg.KERNEL_TYPE,organization="nlp-magnets")
repo_name
output_dir=os.path.join(cfg.DATA_DIR,cfg.KERNEL_TYPE)
repo = Repository(output_dir, clone_from=repo_name)
repo.git_pull()

# %%
def saveCheckpoint(samplesProcessed, currentEpoch):
    print("saving checkpoint")
    checkpoint = { 
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict(),
                 'best_score':  best_score ,
                 'seq': trainSampler.seq,
                  'seq_sampler_i' : samplesProcessed,
                  'currentEpoch' : currentEpoch            
                }
    torch.save(checkpoint, checkpoint_file)

# %%

progress_bar=tqdm(range(num_training_steps))

if currentEpoch < cfg.num_train_epochs:
    doComputation = True
else:
    doComputation = False

BRAKED_TRAINING = False

while doComputation:

    model.train()
    i = 0

    if seq is not None:
        print("reseting SEQ train sampler")
        trainSampler.seq = seq
        trainSampler.shrink(seq_sampler_i)


    for batch in tqdm(train_dataloader):
        i+= cfg.batch_size
        batch={k:v.to(device) for k,v in batch.items()}
        outputs=model(**batch)
        loss=outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        if time.time() - start_time > 10*60*60:
            # make checkpoint and break
            print("going to break")
            saveCheckpoint(i, currentEpoch )
            doComputation = False
            BRAKED_TRAINING = True
            print("BREAK TRAINING")
            break

    if not BRAKED_TRAINING:
        print("evaluation.....")
        model.eval()
        losses=[]
        
        for step,batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs=model(**batch)
            loss=outputs.loss
            losses.append(loss.repeat(cfg.batch_size))
        
        losses=torch.cat(losses)
        losses=losses[:len(eval_dataset)]

        try:
            perplexity=math.exp(torch.mean(losses))
        except OverflowError:
            perplexity=float("inf")

        
        content=time.ctime()+' '+f'Epoch {currentEpoch}, Perplexity: {perplexity}'
        print(content)

        with open(os.path.join(cfg.LOG_DIR, f'log_{cfg.KERNEL_TYPE}.txt'),'a')\
            as appender:
            appender.write(content + '\n')

        model_file=os.path.join(cfg.MODEL_DIR, f'{cfg.KERNEL_TYPE}_best.pth')
        if best_score > perplexity:
            print('score ({:.5f} --> {:.5f}). Saving model ...'.format(best_score, perplexity))
            best_score = perplexity

            checkpoint = { 
                'epoch': currentEpoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict()}
            torch.save(checkpoint, model_file)

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            repo.push_to_hub("best model commit")
                
 
        currentEpoch = currentEpoch+1
        trainSampler.reset(currentEpoch)
        seq = None 
        i = 0
        print("final checkpoint")
        saveCheckpoint(i, currentEpoch )

        if currentEpoch >= cfg.num_train_epochs :
            model_file=os.path.join(cfg.MODEL_DIR, f'{cfg.KERNEL_TYPE}_last.pth')
            checkpoint = { 
                'epoch': currentEpoch-1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict()}
            torch.save(checkpoint, model_file)

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            repo.push_to_hub("final model commit")






# %%



