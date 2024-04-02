from transformers import pipeline
import re
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
from pymatgen.core import Composition
import numpy as np
import argparse
import os

from fractions import Fraction

from sentence_tokenizer import Sentence_dataset
from utils import ELEMENTS

parser=argparse.ArgumentParser()
parser.add_argument("-b","--batch",default=None,type=int)
parser.add_argument("-mc","--model_checkpoint",default=None,type=str)
args=parser.parse_args()

if not os.path.exists(args.model_checkpoint):
    os.makedirs(args.model_checkpoint)

if args.model_checkpoint=="magmatbert":
    token_model_chekpoint="nlp-magnets/CNER_best_magmatbert_10_epochs_lr_3e-05_BS_32"
    qa_model_checkpoint="nlp-magnets/QA_best_Tc_finetunedmagmatbert_10_epochs_lr_5e-05_BS_16"
    
if args.model_checkpoint=="magmatbert_post_squad":
    token_model_chekpoint="nlp-magnets/CNER_best_magmatbert_10_epochs_lr_3e-05_BS_32"
    qa_model_checkpoint="nlp-magnets/QA_best_Tc_finetunedQA_best_magmatbert_4_epochs_lr_3e-05_BS_16_10_epochs_lr_3e-05_BS_16"
    
if args.model_checkpoint=="magbert":
    token_model_chekpoint="nlp-magnets/CNER_best_magbert_10_epochs_lr_3e-05_BS_16"
    qa_model_checkpoint="nlp-magnets/QA_best_magbert_4_epochs_lr_5e-05_BS_64"

if args.model_checkpoint=="matscibert":
    token_model_chekpoint="nlp-magnets/CNER_best_matscibert_10_epochs_lr_3e-05_BS_16"
    qa_model_checkpoint="nlp-magnets/QA_best_matscibert_4_epochs_lr_3e-05_BS_32" 

if args.model_checkpoint=="bert":
    token_model_chekpoint="nlp-magnets/CNER_best_bert-base-uncased_10_epochs_lr_3e-05_BS_16"
    qa_model_checkpoint="nlp-magnets/QA_best_bert-base-uncased_4_epochs_lr_3e-05_BS_16"

token_classifier = pipeline(
    "token-classification", model=token_model_chekpoint, aggregation_strategy="simple"
)

question_answerer = pipeline("question-answering", model=qa_model_checkpoint)

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def frac2string(s):
    i, f = s.groups(0)
    f = Fraction(f)
    return str(int(i) + round(float(f),2))

ds_class=Sentence_dataset("nlp-magnets/Classified_dataset","nlp-magnets/magnetics_corpus_all")
processed_classfied_ds=ds_class.process_classified_ds(args.batch)


db_for_doi_sent=defaultdict(list)
db=defaultdict(list)

funnel=["ferroelectric","multiferroic","Ferroelectric","Multiferroic","paramagnetic Curie temperature","Paramagnetic Curie temperature"]

progress_bar = tqdm(range(len(processed_classfied_ds)))
checkpoint=0
for i,sentences in enumerate(processed_classfied_ds['sentences']):
    
    doi=processed_classfied_ds['doi'][i]
    
    for sentence in sentences:
        sentence=re.sub(r'\[.*?\]', '', sentence) #removes numbers in [].

        # sentence=sentence.encode('ascii', 'ignore').decode() #removes unicode characters.
        results=token_classifier(sentence)
        if results and not any(x in sentence for x in funnel):
            if "Curie temperature" in sentence or "curie temperature" in sentence:
                entity_list=[]
                ent_pymat_list=[]
                unit=None
                for result in reversed(results):
                    
                    entity_org=sentence[result['start']:result['end']]
                    entity_pymat=entity_org.replace(",","").replace("-","").replace(" ","")
                    
                    if "/" in entity_pymat:
                        entity_pymat= re.sub(r'(?:(\d+)[-\s])?(\d+/\d+)', frac2string,entity_pymat )
                        
                    try:
                        comp=Composition(entity_pymat)
                        if comp:
                            entity_pymat=comp.get_reduced_formula_and_factor()[0]
                            
                        else:
                            continue
                        
                        if any(k in ['Fe',"Ni","Co","Gd","Mn","Cr","Pt","Pd","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Te","Se","Ge","U"] for k,v in comp.as_dict().items()) and not any("0+" in k for k,v in comp.as_dict().items()):
                        
                            if entity_org not in entity_list and entity_pymat not in ELEMENTS and not any(substr in entity_org for substr in ["FCC","BCC","L10","PrCoB","PrFeB","NdFeB","Nd-Fe-B"]) and entity_org!="SmCo" and entity_org!="Sm-Co" and entity_org!="Sm-Co-N" and entity_org!="FeN" and entity_org!="Fe-N" and entity_org!="(Ga,Mn)As":
                                question=f"What is the curie temperature of {entity_org}."
                                context=sentence
                                answer=question_answerer(question=question, context=context)
                                
                                if "no-answer" in answer['answer']:
                                    continue
                                
                                if has_numbers(answer['answer']) and answer['score']>0.4 and not any(x in answer['answer'] for x in ['less','more','smaller','larger','higher','lower','above','below','between',"J/kg"]) and answer['answer'].endswith("K"):
                                    unit="K"
                                    db[entity_pymat].append(answer['answer'])

                                    db_for_doi_sent[entity_pymat].append(answer['answer'])
                                    db_for_doi_sent[f'{entity_pymat}_org_formula'].append(entity_org)
                                    db_for_doi_sent[f'{entity_pymat}_doi'].append(doi)
                                    db_for_doi_sent[f'{entity_pymat}_sentence'].append(sentence)
                                    
                                    sentence=sentence[:answer['start']]+" "*(answer['end']-answer['start'])+sentence[answer['end']:]
                                    sentence=sentence[:result['start']]+" "*(result['end']-result['start'])+sentence[result['end']:]
                                    ent_pymat_list.append(entity_pymat)
                                    
                                elif has_numbers(answer['answer']) and answer['score']>0.4 and not any(x in answer['answer'] for x in ['less','more','smaller','larger','higher','lower','above','below','between',"J/kg"]) and answer['answer'].endswith("°C"):
                                    unit="°C"
                                    db[entity_pymat].append(answer['answer'])

                                    db_for_doi_sent[entity_pymat].append(answer['answer'])
                                    db_for_doi_sent[f'{entity_pymat}_org_formula'].append(entity_org)
                                    db_for_doi_sent[f'{entity_pymat}_doi'].append(doi)
                                    db_for_doi_sent[f'{entity_pymat}_sentence'].append(sentence)
                                    
                                    sentence=sentence[:answer['start']]+" "*(answer['end']-answer['start'])+sentence[answer['end']:]
                                    sentence=sentence[:result['start']]+" "*(result['end']-result['start'])+sentence[result['end']:]
                                    ent_pymat_list.append(entity_pymat)
                                    
                                elif has_numbers(answer['answer']) and answer['score']>0.4 and not any(x in answer['answer'] for x in ['less','more','smaller','larger','higher','lower','above','below','between',"J/kg"]) and unit:
                                    db[entity_pymat].append(answer['answer']+unit)

                                    db_for_doi_sent[entity_pymat].append(answer['answer']+unit)
                                    db_for_doi_sent[f'{entity_pymat}_org_formula'].append(entity_org)
                                    db_for_doi_sent[f'{entity_pymat}_doi'].append(doi)
                                    db_for_doi_sent[f'{entity_pymat}_sentence'].append(sentence)
                                    
                                    sentence=sentence[:answer['start']]+" "*(answer['end']-answer['start'])+sentence[answer['end']:]
                                    sentence=sentence[:result['start']]+" "*(result['end']-result['start'])+sentence[result['end']:]
                                    ent_pymat_list.append(entity_pymat)
                                
                                elif has_numbers(answer['answer']) and answer['score']>0.4 and not any(x in answer['answer'] for x in ['less','more','smaller','larger','higher','lower','above','below','between',"J/kg"]):
                                    db[entity_pymat].append(answer['answer'])

                                    db_for_doi_sent[entity_pymat].append(answer['answer'])
                                    db_for_doi_sent[f'{entity_pymat}_org_formula'].append(entity_org)
                                    db_for_doi_sent[f'{entity_pymat}_doi'].append(doi)
                                    db_for_doi_sent[f'{entity_pymat}_sentence'].append(sentence)
                                    
                                    sentence=sentence[:answer['start']]+" "*(answer['end']-answer['start'])+sentence[answer['end']:]
                                    sentence=sentence[:result['start']]+" "*(result['end']-result['start'])+sentence[result['end']:]
                                    ent_pymat_list.append(entity_pymat)
                                entity_list.append(entity_org)
    
                    except ValueError:
                        continue
                    
                if unit:
                    for entity in ent_pymat_list:
                        if not db[entity][-1].endswith(unit):
                            tmp_answer=db[entity].pop()
                            db[entity].append(tmp_answer+unit)
                            
                            db_for_doi_sent[entity].pop()
                            db_for_doi_sent[entity].append(tmp_answer+unit)
                else:
                    for entity in ent_pymat_list:
                        db[entity].pop()
                        
                        db_for_doi_sent[entity].pop()
                        db_for_doi_sent[f'{entity}_org_formula'].pop()
                        db_for_doi_sent[f'{entity}_doi'].pop()
                        db_for_doi_sent[f'{entity}_sentence'].pop()
                        
                        if not db[entity]:
                            del db[entity]
                            del db_for_doi_sent[entity]
                            del db_for_doi_sent[f'{entity}_org_formula']
                            del db_for_doi_sent[f'{entity}_doi']
                            del db_for_doi_sent[f'{entity}_sentence']
             
    progress_bar.update(1)
    
    checkpoint+=1
    if checkpoint>=5000:
        df=pd.DataFrame.from_dict(db,orient='index').reset_index().rename(columns={'index':"Material",0:"TC"})
        df_doi_sent=pd.DataFrame.from_dict(db_for_doi_sent,orient='index').reset_index()
        df.to_csv(f'./{args.model_checkpoint}/curie_only_{args.model_checkpoint}_batch_{args.batch}_partial.csv')
        df_doi_sent.to_csv(f'./{args.model_checkpoint}/curie_metadata_{args.model_checkpoint}_batch_{args.batch}_partial.csv')  
        checkpoint=0

df=pd.DataFrame.from_dict(db,orient='index').reset_index().rename(columns={'index':"Material",0:"TC"})
df_doi_sent=pd.DataFrame.from_dict(db_for_doi_sent,orient='index').reset_index()
df.to_csv(f'./{args.model_checkpoint}/curie_only_{args.model_checkpoint}_batch_{args.batch}_full.csv')
df_doi_sent.to_csv(f'./{args.model_checkpoint}/curie_metadata_{args.model_checkpoint}_batch_{args.batch}_full.csv')  