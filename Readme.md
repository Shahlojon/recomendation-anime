### Задача проекта

Создать рекомендательную систему для аниме, которая будет предоставлять пользователям персонализированные рекомендации. Система должна учитывать предпочтения пользователей, их историю взаимодействия с контентом, а также обеспечивать возможность холодного старта для новых пользователей и контента.

---

### Бизнес-цель проекта

Увеличить вовлеченность пользователей платформы, повысить количество просмотров аниме, и улучшить пользовательский опыт, предоставляя персонализированные рекомендации. Это может привести к росту времени, проводимого на платформе, а также улучшить удержание пользователей.

---

### Описание данных

1. **`anime.csv`**:
   - Поля:
     - `anime_id` — уникальный идентификатор аниме.
     - `title` — название аниме.
     - `genre` — жанры (через запятую).
     - `type` — тип (TV, OVA, Movie и т.д.).
     - `episodes` — количество эпизодов.
     - `rating` — средняя оценка.
     - `members` — количество пользователей, добавивших аниме в свои списки.

2. **`animelist.csv`**:
   - Поля:
     - `user_id` — уникальный идентификатор пользователя.
     - `anime_id` — идентификатор аниме.
     - `rating` — оценка пользователя.
     - `status` — статус (Watching, Completed, On Hold и т.д.).
     - `watched_episodes` — количество просмотренных эпизодов.

---

### Схема данных в реальной жизни

В реальной жизни эти данные могли бы лежать в реляционной базе данных, например, PostgreSQL:

1. **Таблица `Anime`**:
   - `anime_id` (PK)
   - `title`
   - `genre`
   - `type`
   - `episodes`
   - `rating`
   - `members`

2. **Таблица `Users`**:
   - `user_id` (PK)
   - `name` 
   - `join_date` 

3. **Таблица `UserAnimeInteractions`**:
   - `interaction_id` (PK)
   - `user_id` (FK, `Users.user_id`)
   - `anime_id` (FK, `Anime.anime_id`)
   - `rating`
   - `status`
   - `watched_episodes`

---

### МЛ системный дизайн

1. **Сбор данных**:
   - История просмотров пользователей.
   - Оценки и статусы аниме.
   - Метаданные аниме.

2. **Предобработка данных**:
   - Заполнение пропущенных значений (например, средними значениями для оценок).
   - Преобразование текстовых данных (жанры) в векторы (One-Hot или TF-IDF).

3. **Модели**:
   - **Базовая модель (Baseline)**:
     - Рекомендации на основе самых популярных аниме (по количеству участников или рейтингу).
   - **Content-Based Filtering**:
     - Использование жанров, типа и других характеристик аниме для рекомендаций.
   - **Collaborative Filtering**:
     - ALS (Matrix Factorization) или модели на основе нейронных сетей. Обучение и выбор лучшей модели
   - **Session-Based**:
     - Рекуррентные нейронные сети (RNN) или модели на основе Transformers 
   - **Холодный старт**:
     - Рекомендации на основе популярных жанров, пользовательских данных или случайных выборок.

4. **Деплой и API**:
   - FastAPI для предоставления рекомендаций.
   - Модели загружаются и используются в API.

5. **Мониторинг**:
   - Логирование запросов.
   - Метрики качества рекомендаций.

6. **Деплой**
   - Написать Dockerfile
   - Написать yaml файл
   - Настроить ci/cd

7. **A/B тестирование:** 
    Мониторинг качества рекомендаций (поведение пользователей, метрики).

---

### Подготовка моделей

1. **Baseline (ТОП-100)**:
   - Выбрать 100 самых популярных аниме.
   - Сохранить список в JSON-файле.

2. **Content-Based Filtering**:
   - Создать векторное представление аниме на основе жанров, типов и эпизодов.
   - Для каждого пользователя находить ближайшие аниме (например, с помощью косинусного сходства).

3. **Collaborative Filtering**:
   - Реализовать Matrix Factorization (например, с использованием библиотеки `surprise`).
   - Предсказать оценки для пользователя и ранжировать аниме.

4. **Session-Based**:
   - Использовать RNN или SASRec для анализа последовательности взаимодействий пользователя.



# Baseline Top - 100 anime

GET http://localhost:8000/top 

# Content Based Model

curl -X POST http://localhost:8000/recommend_content_based \
-H "Content-Type: application/json" \
-d '{"anime_id": 1, "top_n" = 10}'

# Session Based Model

curl -X POST http://localhost:8000/recommend_session_based \
-H "Content-Type: application/json" \
-d '{"user_id": 1, "top_n" = 10}'

# Коллаборативная фильтрация

curl -X POST http://localhost:8000/recommend_collaborative \
-H "Content-Type: application/json" \
-d '{"user_id": 1, "top_n" = 10}'

# top_n not required

docker build -t recomendation-anime .

docker run -p 8000:8000 recomendation-anime


## Финальный результат
  1. База данных: PostgreSQL или SQLite для хранения данных.
  - Рекомендационные модели:
     1. Baseline (популярное),
     2. Content-based,
     3. Collaborative filtering,
     4. Session-based.
  2. FastAPI сервис для рекомендаций.
  3. Докеризация: развертывание API с помощью Docker.