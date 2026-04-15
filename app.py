import streamlit as st
import pandas as pd
from model import hybrid_recommend

# Load movie titles (FIXED)
movies = pd.read_csv("movies.csv", encoding='latin1')

st.title("🎬 Movie Recommendation System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    recommendations = hybrid_recommend(selected_movie)
    
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(movie)