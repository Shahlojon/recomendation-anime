from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from scipy.sparse import csr_matrix,load_npz
import pandas as pd
from implicit.als import AlternatingLeastSquares
import json
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
# Загрузка данных и моделей
with open('model/session_based.pkl', 'rb') as f:
    model_session = pickle.load(f)

with open('model/best_model.pkl', 'rb') as f:
    model_collaborative = pickle.load(f)

# with open('content_based.pkl', 'rb') as f:
#     model_content_based = pickle.load(f)

with open('model/content_based.json', 'r') as f:
    content_based_model = json.load(f)
  
with open('model/baseline_new.json', 'r') as f:
    baseline = json.load(f)  
# Пример данных (замените на ваши данные)
anime = pd.read_parquet("data/anime.parquet")  # Загрузите ваш датафрейм
merged_data = pd.read_parquet("data/merged_data.parquet")  # Загрузите ваш датафрейм
sparse_matrix = load_npz('data/interaction_matrix_sparse.npz')
anime_ids = anime['anime_id'].values
# Маппинг user_id в индексы
user_id_mapping = {id_: idx for idx, id_ in enumerate(merged_data['user_id'].unique())}

# Создание FastAPI приложения
app = FastAPI()

# Модель запроса для получения рекомендаций
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10

# Функция для получения рекомендаций коллаборативной фильтрации
def recommend_collaborative(user_id, model=model_collaborative, sparse_matrix=sparse_matrix, top_n=10):
    user_idx = user_id_mapping.get(user_id)
    if user_idx is None:
        return []

    anime_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=top_n)

    recommendations = []
    for i, score in zip(anime_indices, scores):  # Используем zip для правильного отображения индекса и оценки
        anime_id = anime_ids[i]  # Получаем id аниме по индексу
        # anime_name = merged_data.loc[merged_data[merged_data['anime_id'] == anime_id].index[0], 'Name']  # Извлекаем имя аниме
        anime_row = merged_data.loc[merged_data['anime_id'] == anime_id]
        if anime_row.empty:  # Проверяем, если данных по аниме нет
            continue

        # Извлекаем данные о названии, рейтинге и жанрах
        anime_name = anime_row.iloc[0]['Name']
        anime_rating = anime_row.iloc[0]['rating']  # Рейтинг аниме
        anime_genres = anime_row.iloc[0]['Genres']  # Жанры аниме
        recommendations.append({
            "anime_id": anime_id,
            "Name": anime_name,
            "score": score,
            "rating": anime_rating,  # Добавляем рейтинг
            "genres": anime_genres  # Добавляем жанры
        })

    return recommendations

# Функция для получения рекомендаций с использованием session-based модели
def recommend_session_based(user_id, model=model_session, sparse_matrix=sparse_matrix, top_n=10):
    user_idx = user_id_mapping.get(user_id)
    if user_idx is None:
        return []

    anime_indices, scores = model.recommend(user_idx, sparse_matrix[user_idx], N=top_n)
    recommendations = []
    for i, score in zip(anime_indices, scores):
        anime_id = anime_ids[i]
        anime_name = anime.loc[anime[anime['anime_id'] == anime_id].index[0], 'Name']
        recommendations.append({
            "anime_id": anime_id,
            "Name": anime_name,
            "score": score
        })
    
    return recommendations

# Функция для получения контентных рекомендаций
# def recommend_content_based(anime_id, top_n=10):
#     idx = anime[anime['anime_id'] == anime_id].index[0]
#     sim_scores = list(enumerate(model_content_based[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     anime_indices = [i[0] for i in sim_scores[1:top_n + 1]]
#     return anime.iloc[anime_indices][['anime_id', 'Name']].to_dict(orient='records')

def recommend_content_based_with_json(anime_id, model=content_based_model, top_n=10):
    anime_id_str = str(anime_id)
    # Проверяем, существует ли аниме в модели
    print(type(anime_id_str))
    print(model)

    if anime_id_str not in model:
        return []

    recommended_anime_ids = model[anime_id_str][:top_n]
    recommendations = []
    for recommended_id in recommended_anime_ids:
        anime_name = anime.loc[anime[anime['anime_id'] == recommended_id].index[0], 'Name']
        recommendations.append({
            "anime_id": recommended_id,
            "Name": anime_name
        })

    return recommendations


# Маршрут для коллаборативной фильтрации
@app.post("/recommend_collaborative")
async def recommend_collaborative_endpoint(request: RecommendationRequest):
    recommendations = recommend_collaborative(request.user_id, top_n=request.top_n)
    serialized_recommendations = [
        {
            "anime_id": int(rec["anime_id"]),
            "Name": rec["Name"],
            "score": float(rec["score"]),
            "rating": float(rec["rating"]),
            "genres": rec["genres"]
        }
        for rec in recommendations
    ]
    return {"recommendations": serialized_recommendations}

# Маршрут для session-based модели
@app.post("/recommend_session_based")
async def recommend_session_based_endpoint(request: RecommendationRequest):
    recommendations = recommend_session_based(request.user_id, top_n=request.top_n)
    serialized_recommendations = [
        {
            "anime_id": int(rec["anime_id"]),
            "Name": rec["Name"],
            "score": float(rec["score"])
        }
        for rec in recommendations
    ]
    return {"recommendations": serialized_recommendations}

# Маршрут для контентных рекомендаций
@app.post("/recommend_content_based")
async def recommend_content_based_endpoint(anime_id: int, top_n: int = 10):
    recommendations = recommend_content_based_with_json(anime_id = anime_id,top_n=top_n)
    # Преобразуем типы внутри списка словарей
    serialized_recommendations = [
        {
            "anime_id": int(rec["anime_id"]),
            "Name": rec["Name"]
        }
        for rec in recommendations
    ]
    return {"recommendations": serialized_recommendations}

@app.get("/top")
async def baseline_top():
    return {"recommendations": baseline[:100]}

