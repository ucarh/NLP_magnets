
from dataset import *
from model import *
from utils import *

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

parser.add_argument("-dn","--dataset_name",default="batterydata/cner",type=str)
parser.add_argument("--log_dir", default="./logs", type=str,
                    help="The output dir logging.")
parser.add_argument("--weights_dir", default="./weights", type=str,
                    help="The output dir weights.")
parser.add_argument("--data_dir", default=".", type=str,
                    help="The output dir huggingface repo.")
parser.add_argument("--kernel_type", default=None, type=str)

args=parser.parse_args()

if '/' in args.model_checkpoint:
    args.kernel_type="CNER_"+args.model_checkpoint.split('/')[-1]+"_"+str(args.epochs)+"_epochs"+"_lr_"+str(args.learning_rate)+"_BS_"+str(args.batch_size)
else:
    args.kernel_type="CNER_"+args.model_checkpoint+"_"+str(args.epochs)+"_epochs"+"_lr_"+str(args.learning_rate)+"_BS_"+str(args.batch_size)
    

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)

import evaluate
import numpy as np
metric = evaluate.load("seqeval")

raw_ds=CNER_Dataset(args.dataset_name,args.model_checkpoint,args.batch_size).get_raw_data()
label_list=CNER_Dataset(args.dataset_name,args.model_checkpoint,args.batch_size).get_label_list(raw_ds['train']['labels'])
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

model=CNER_model(args.model_checkpoint).get_model(id2label,label2id)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

train_dataloader,eval_dataloader=CNER_Dataset(args.dataset_name,args.model_checkpoint,args.batch_size).get_train_eval_loader()
tokenizer=CNER_Dataset(args.dataset_name,args.model_checkpoint,args.batch_size).get_tokenizer()

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
        
    metric=evaluate.load('seqeval')
    model.eval()
    
    for batch in eval_dataloader:
        batch={k:v.to(device) for k,v in batch.items()}
        
        with torch.no_grad():
            outputs=model(**batch)
            
        logits=outputs.logits
        predictions=torch.argmax(logits,dim=-1)
        labels=batch['labels']
        true_predictions, true_labels = postprocess(predictions, labels,label_list)
        metric.add_batch(predictions=true_predictions, references=true_labels)
        
    result=metric.compute()
    f1=result[f"overall_f1"]
    
    content=time.ctime()+' '+f'Epoch {epoch}, Accuracy: {result[f"overall_accuracy"]}, F1: {result[f"overall_f1"]}, Precision: {result[f"overall_precision"]}, Recall: {result[f"overall_recall"]}'
                
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

    


