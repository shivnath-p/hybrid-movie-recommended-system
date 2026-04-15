import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load CSV files
movies = pd.read_csv("movies.csv", encoding='latin1')
ratings = pd.read_csv("ratings.csv", encoding='latin1')

# Merge data
data = pd.merge(ratings, movies, on="movieId")

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def content_recommend(title):
    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]

    return list(movies['title'].iloc[movie_indices])


# Collaborative Filtering
movie_ratings = data.groupby('title')['rating'].mean().sort_values(ascending=False)

def collaborative_recommend():
    return list(movie_ratings.head(5).index)


# Hybrid Recommendation
def hybrid_recommend(title):
    content = content_recommend(title)
    collab = collaborative_recommend()

    # Combine both
    result = list(set(content + collab))

    return result[:5]