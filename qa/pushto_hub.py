
from huggingface_hub import create_repo,get_full_repo_name,Repository
import os
import shutil

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("-rn","--repo_name",default=None,type=str)
parser.add_argument("-lf","--local_file",default=None,type=str)
args=parser.parse_args()

create_repo(f"nlp-magnets/{args.repo_name}",private=True,repo_type="model")
repo_dir=args.repo_name

repo_name=get_full_repo_name(model_id=args.repo_name,organization="nlp-magnets")
repo = Repository(repo_dir, clone_from=repo_name)
repo.git_pull()

shutil.copytree(args.local_file,args.repo_name, dirs_exist_ok=True)

repo.push_to_hub("best model commit")