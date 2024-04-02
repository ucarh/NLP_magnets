# %%
from sentence_tokenizer import *
from transformers import pipeline
import re
from collections import defaultdict
import pandas as pd
from pymatgen.core import Composition
from utils import ELEMENTS

# %%
ds_class=Sentence_dataset("nlp-magnets/Classified_dataset","nlp-magnets/magnetics_corpus_all")

# %%
processed_classfied_ds=ds_class.process_classified_ds(2)

# %%
token_model_chekpoint="nlp-magnets/CNER_best_magmatbert_10_epochs_lr_3e-05_BS_32"
token_classifier = pipeline(
    "token-classification", model=token_model_chekpoint, aggregation_strategy="simple"
)

# %%
sentences_dict=defaultdict(list)
sentence_list=[]
funnel=["ferroelectric","multiferroic","Ferroelectric","Multiferroic","antiferromagnetic","Antiferromagnetic","paramagnetic Curie temperature","Paramagnetic Curie temperature"]

progress_bar = tqdm(range(len(processed_classfied_ds['sentences'][:5000])))
checkpoint=0
for i,sentences in enumerate(processed_classfied_ds['sentences'][:5000]):
    
    doi=processed_classfied_ds['doi'][i]
    
    for sent_idx,sentence in enumerate(sentences):
        
        sentence=re.sub(r'\[.*?\]', '', sentence) #removes numbers in [].

        # sentence=sentence.encode('ascii', 'ignore').decode() #removes unicode characters.
        results=token_classifier(sentence)
        if results and not any(x in sentence for x in funnel):
            if "Curie temperature" in sentence or "curie temperature" in sentence:
                entity_list=[]
                for idx,result in reversed(list(enumerate(results))):
                    entity_org=sentence[result['start']:result['end']]
                    entity_pymat=entity_org.replace(",","").replace("-","").replace(" ","")
                    
                    try:
                        comp=Composition(entity_pymat)
                        if comp:
                            entity_pymat=comp.get_reduced_formula_and_factor()[0]
                            
                        else:
                            continue
                        
                        if any(k in ['Fe',"Ni","Co","Gd","Mn","Cr","Pt","Pd","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Te","Se","Ge","U"] for k,v in comp.as_dict().items()) and not any("0+" in k for k,v in comp.as_dict().items()):
                            if entity_org not in entity_list and entity_pymat not in ELEMENTS and not any(substr in entity_org for substr in ["FCC","BCC","L10","PrCoB","PrFeB","NdFeB","Nd-Fe-B"]) and entity_org!="SmCo" and entity_org!="Sm-Co" and entity_org!="Sm-Co-N" and entity_org!="FeN" and entity_org!="Fe-N" and entity_org!="(Ga,Mn)As":
                                sentences_dict['id'].append(f'{doi}_{sent_idx}_{idx}')
                                sentences_dict['context'].append(sentence)
                                sentences_dict['question'].append(f'What is the Curie temperature of {entity_org}?')
                                entity_list.append(entity_org)
        
                    except ValueError:
                        continue
    progress_bar.update(1)
    
    # checkpoint+=1
    # if checkpoint>=5000:
    #     df=pd.DataFrame.from_dict(sentences_dict)
    #     df.to_csv('Curie_sentences_forQA.csv')
    #     checkpoint=0
        
            

# %%
df=pd.DataFrame.from_dict(sentences_dict)
df.to_csv('Curie_sentences_forQA_III.csv')

# %%



