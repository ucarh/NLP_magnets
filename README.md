# MagBERT
A dataset of Curie temperatures are constructed by training BERT models using approximately 144 K magnetics research papers. The workflow is as follows:
First, magnetic corpus is generated using the script in the "corpus_generation" folder which uses the "list_of_DOIs.csv" file in the main directory. Then, plain BERT and MatSciBERT models are pretrained as per the scripts in the "pretraining" folder. This is followed by removing the irrelevant articles from the pool of 144 K articles based on the classifier in the "classification task" folder. Next, named entity recognition (ner), and question answering (qa) finetuning is performed. 

# Data Generation:
For data generation: ner and qa models are used in the run_data_generator.py script in the dataset_generator folder which extracts Curie temperatures from texts. Final datasets generated for each checkpoint are provided in the "Tc_datasets_generated" folder. 

The weights for MagBERT and MagMatBERT models for the masked language model, ner and qa are provided at https://huggingface.co/nlp-magnets. 

# Usage:
Demo notebooks that demonstrate the use of MLM, QA and NER tasks are provided in the main directory. Note that you may need to change the name of the checkpoints provided in the notebooks as we renamed most of the models. 

The data extraction workflow can be translated to other types of properties with MagBERT and MagMatBERT performing better especially for magnetics related properties. The only thing that might be needed for the end user is to create a manually annotated QA dataset for the property in question as it is done in Tc_ds_forQA.json in the "qa" folder. It boosts the performance of models in data extraction. 

