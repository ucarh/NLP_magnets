
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
import pandas as pd
import random
import re
from datasets import load_dataset
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--year",type=int, default=2023,help="enter the year you would like to get the articles from")
args=parser.parse_args()

con_file = open("config.json")
config = json.load(con_file)
con_file.close()


client = ElsClient(config['apikey'])
client.inst_token = config['insttoken']


def create_dataset(doi,outfile):
    doi_doc = FullDoc(doi = doi)
    
    if doi_doc.read(client):
        try:
            title = doi_doc.title
        except:
            title=""
        try:
            abstract = doi_doc.data['coredata']['dc:description']
        except:
            abstract=""
            
        try:
            text = doi_doc.data['originalText']
        except:
            text=None

        if isinstance(text,str):
            if '1 Introduction' not in text:
                introduction=text.find('Introduction',text.find('Introduction')+1)
            else:
                introduction=text.find('1 Introduction',text.find('1 Introduction')+1)

            
            try:
                regex=re.compile(r'\d+\sConclusions')
                conclusions_occurence=regex.findall(text)[0]
                conclusions_start=text.find(conclusions_occurence,text.find(conclusions_occurence)+1)
                filtered_text=text[introduction:conclusions_start+750].strip()
                # print('regex 1 found')
            except:
                try:
                    regex=re.compile(r'\d+\sConclusion')
                    conclusions_occurence=regex.findall(text)[0]
                    conclusions_start=text.find(conclusions_occurence,text.find(conclusions_occurence)+1)
                    filtered_text=text[introduction:conclusions_start+750].strip()
                    # print('regex 2 found')

                except:
                    if 'References [1]' not in text:
                        reference_start = text.find('References',text.find('References')+1)
                    else:
                        reference_start = text.find("References [1]")
                    filtered_text = text[introduction:reference_start-1000].strip()
                    # print('regex NOT found')
            
            json.dump({'title': title, 'abstract': abstract, 'text': filtered_text,'doi':doi}, outfile)
        else:
            filtered_text="Text is not string."
            json.dump({'title': title, 'abstract': abstract, 'text': filtered_text,'doi':doi}, outfile)
          
    else:
        title=""
        abstract=""
        filtered_text="Read document failed."
        json.dump({'title': title, 'abstract': abstract, 'text': filtered_text,'doi':doi}, outfile)






if __name__=="__main__":

    df=pd.read_csv(f'./DOI_Elsevier_magnetic/{args.year}_Elsevier_data.csv')
    DOI_list=df['DOI'].to_list()
    count=0
    with open(f'data/{args.year}_magnetic_corpus.jsonl', 'a') as outfile:
        for doi in DOI_list:
            with open(f"data/{args.year}_magnetic_corpus_progress.txt","a") as file:
                file.write(f"working on doi:{doi}, step {count}/{len(DOI_list)} \n")
            count+=1
            
            create_dataset(doi, outfile)

        
