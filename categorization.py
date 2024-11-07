import csv
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from tqdm import tqdm 
import time
import numpy as np
import os
from collections import OrderedDict

tags = ['hanam']
tag = tags[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 800 # hyper_params.
test_count = 200
batch_size = 16
epochs = 128

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

try:
    model_state_dict = torch.load("basic_model_1000" +  ".pt", map_location=device)
    try:
        model.load_state_dict(model_state_dict)

    except RuntimeError:
        print("E1")
        try:
            model.load_state_dict(model_state_dict, strict=False)
        except:
            pass

except FileNotFoundError:
    print("E0")
    try:
        model_state_dict = torch.load("basic_model" +  ".pt", map_location=device)
        try:
            model.load_state_dict(model_state_dict)

        except RuntimeError:
            print("E1")
            try:
                model.load_state_dict(model_state_dict, strict=False)
            except:
                pass
    except:
        pass

print("model loading complete.")

review_df = pd.read_csv("preprocessed_urls_hanam_restaurant_real_url_review.csv", encoding="utf-8") 
category_df = pd.read_csv("category.csv")

pre_name = review_df.iloc[0]["name"]
name = review_df.iloc[0]["name"]
sim_list = []
temp_sim_list = []

for i in tqdm(range(len(review_df))):
    name = review_df.iloc[i]["name"]

    if(name != pre_name or i == len(review_df)-1):
        sim_list.append([pre_name, t1, sum(temp_sim_list) / len(temp_sim_list)])
        temp_sim_list = []

    try:
        t0 = review_df.iloc[i]["review"]
        v0 = model.encode([t0])
    except TypeError:
        sim = 0
    # print(name, t0, category_df["name"].values().index(name))

    try:
        t1 = category_df.iloc[category_df["name"].values.tolist().index(name)]["category3"]
        v1 = model.encode([t1])

        sim = util.cos_sim(v0, v1)
    except:
        sim = 0

    pre_name = review_df.iloc[i]["name"]
    temp_sim_list.append(sim)

for i in range(len(sim_list)):
    print(sim_list[i])

column = ["name", "category", "sim"]
df2 = pd.DataFrame(sim_list, columns=column)
df2.to_csv("categorized_sim.csv")