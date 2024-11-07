import csv
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from tqdm import tqdm 
import time

tags = ['hanam']
tag = tags[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 800 # hyper_params.
test_count = 200
batch_size = 16
epochs = 128

mapper_df = pd.read_csv("hanam_dict.csv", encoding="utf-8")
mapper = list(mapper_df['1'])

data_df = pd.read_csv("preprocessed_urls_hanam_restaurant_real_url_review.csv", encoding="utf-8")
test_data = [[] for _ in list(range(len(mapper)))]
temp_data = []

for idx in range(len(data_df)):
    test_data[mapper.index(str(data_df.iloc[idx]["name"]))].append(str(data_df.iloc[idx]["review"]))
    temp_data.append([str(data_df.iloc[idx]["review"]), mapper.index(str(data_df.iloc[idx]["name"]))])

test_examples = []
f = open(tag + '_test_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

import os
from collections import OrderedDict

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

import numpy as np

vectors = [[] for _ in list(range(len(mapper)))]
try:
    f = open(tag + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row), 0.3396976888179779, "int(temp_data[i][1])", int(temp_data[i][1]))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

    print("reading vectors complete.")

except FileNotFoundError:
    f = open(tag + '_vectors_' + str(train_count+test_count) + '.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in tqdm(list(range(len(mapper)))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
        for j in list(range(len(test_data[i]))):
            vector = model.encode(test_data[i][j])
            listedVector = vector.tolist()
            writer.writerow(listedVector)
        # print(i, "encoding done")

    f.close()

    f = open(tag + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

print("Start recommending")
start_time = time.time()

mean_vectors = []
try:
    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
        mean_vectors.append((np.array(floatCastedRow), i))
    f.close()

    print("reading mean vectors complete.")
except FileNotFoundError:
    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in tqdm(list(range(len(mapper)))):
        vector = model.encode(test_data[i])
        vector = np.mean(vector, axis=0)

        listedVector = vector.tolist()
        writer.writerow(listedVector)
    f.close()

    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        mean_vectors.append((np.array(floatCastedRow), i))
    f.close()

acc = 0
hitsAt3 = 0
hitsAt5 = 0
hitsAt10 = 0
rankingBasedMetric = 0

targetText = "비가 올 때 가면 좋아요" #상상 리뷰
targetVector = model.encode([targetText]) # targetVector는 데스트 할 text string의 sentence vector

for i in list(range(test_count)):
    results = []
    answerList = []
    for j in list(range(len(mapper))):
        similarities = util.cos_sim(np.array(mean_vectors[j][0], dtype=np.float32), targetVector) # compute similarity between sentence vectors
        results.append((j, mapper[j], float(similarities)))
    results.sort(key = lambda x : -x[2])

print(targetText, "에 적합한 식당은")
print("번호" + " " + "상호명" + " " + "score")
print("="*45)
for result in results[:10]:
    printedString = result
    print(printedString)
print("입니다.")

# torch.save(model.state_dict(), 'basic_model_'+str(train_count+test_count)+'.pt')

# import matplotlib.pyplot as plt
# plt.hist(simList[results[0][0]], bins=20)
# plt.show()

print(f"Recommending done after {time.time() - start_time:.2f} sec")