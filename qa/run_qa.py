from dataset import QA_Dataset
from tokenizer import QA_tokenizer
from model import QA_model
from utils import compute_metrics
import torch
import os
import numpy as np

import argparse


parser=argparse.ArgumentParser()
parser.add_argument("-mc","--model_checkpoint",default=None,type=str)
parser.add_argument("-lr","--learning_rate", default=None, type=float,
                    help="The initial learning rate.")
parser.add_argument("-ep","--epochs", default=None, type=int,
                    help="Number of training epochs")
parser.add_argument("-bs","--batch_size", default=None, type=int)
parser.add_argument("-cd","--cuda_device", default="4", type=str)
parser.add_argument("-cr","--create_repo",default=False,type=bool)
parser.add_argument("-ph","--pushto_hub", default=False,type=bool)

# parser.add_argument("-dn","--dataset_name",default="squad",type=str)
parser.add_argument("--log_dir", default="./logs", type=str,
                    help="The output dir logging.")
parser.add_argument("--weights_dir", default="./weights", type=str,
                    help="The output dir weights.")
parser.add_argument("--data_dir", default=".", type=str,
                    help="The output dir huggingface repo.")
parser.add_argument("--kernel_type", default=None, type=str)

args=parser.parse_args()

if '/' in args.model_checkpoint:
    args.kernel_type="QA_Tc_finetuned"+args.model_checkpoint.split('/')[-1]+"_"+str(args.epochs)+"_epochs"+"_lr_"+str(args.learning_rate)+"_BS_"+str(args.batch_size)
else:
    args.kernel_type="QA_Tc_finetuned"+args.model_checkpoint+"_"+str(args.epochs)+"_epochs"+"_lr_"+str(args.learning_rate)+"_BS_"+str(args.batch_size)
    
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)

raw_ds=QA_Dataset(args.model_checkpoint,args.batch_size).get_dataset()
train_dataloader=QA_Dataset(args.model_checkpoint,args.batch_size).get_train_ds()
eval_ds,eval_dataloader=QA_Dataset(args.model_checkpoint,args.batch_size).get_eval_ds()

model=QA_model(args.model_checkpoint).get_model()
tokenizer=QA_tokenizer(args.model_checkpoint).get_tokenizer()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

from transformers import get_scheduler

num_train_epochs = args.epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from huggingface_hub import create_repo,get_full_repo_name,Repository

#uncomment below only if the repo is not present on huggingface.
if args.create_repo:
    create_repo(f"nlp-magnets/{args.kernel_type}",private=True,repo_type="model")

repo_dir=os.path.join(args.data_dir,args.kernel_type)

if args.pushto_hub:
    repo_name=get_full_repo_name(model_id=args.kernel_type,organization="nlp-magnets")
    repo = Repository(repo_dir, clone_from=repo_name)
    repo.git_pull()

from tqdm.auto import tqdm
import time 

best_score=float(0)
progress_bar=tqdm(range(num_training_steps))

model.train()

for epoch in range(num_train_epochs):
    for batch in train_dataloader:
        batch={k:v.to(device) for k,v in batch.items()}
        outputs=model(**batch)
        loss=outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    model.eval()
    start_logits = []
    end_logits = []
    
    for batch in tqdm(eval_dataloader):
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(eval_ds)]
    end_logits = end_logits[: len(eval_ds)]
    
    metrics = compute_metrics(
        start_logits, end_logits, eval_ds, raw_ds['validation']
    )
    
    f1=metrics['f1']
    exact_match=metrics['exact_match']
    content=time.ctime()+' '+f'Epoch {epoch}, F1: {f1}, EM: {exact_match}'
    
    print(content)
    with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'),'a')\
        as appender:
        appender.write(content + '\n')
        
    model_file=os.path.join(args.weights_dir, f'{args.kernel_type}_best.pth')
    
    if f1 > best_score:
        print('score ({:.5f} --> {:.5f}). Saving model ...'.format(best_score, f1))
        best_score = f1

        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_scheduler.state_dict()}
        torch.save(checkpoint, model_file)

        model.save_pretrained(repo_dir)
        tokenizer.save_pretrained(repo_dir)
        
        if args.pushto_hub:
            repo.push_to_hub("best model commit")
    
model_file=os.path.join(args.weights_dir, f'{args.kernel_type}_last.pth')
checkpoint = { 
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_sched': lr_scheduler.state_dict()}
torch.save(checkpoint, model_file)
