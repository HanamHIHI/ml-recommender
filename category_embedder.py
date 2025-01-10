import numpy as np
import pandas as pd

from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch

category_list = ['해물 요리', '한식당', '일식당', '양식', '고기 요리', '카페', '식당', '디저트',
       '햄버거', '식당 아님', '분식', '치킨', '맥주', '피자', '중국집', '베이커리', '아시안 음식',
       '야채 요리', '주류']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device="cpu") # pre-trained 모델 불러오기
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
        model_state_dict = torch.load("basic_model_1000" +  ".pt", map_location=device)
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

vectors = model.encode(category_list)

print(type(vectors), vectors.shape)

df = pd.DataFrame(data=vectors)
df.to_csv("category_embedding_v7.csv")