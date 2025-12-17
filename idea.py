import streamlit as st
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import  google.generativeai as genai
import os
from google import genai
from PIL import Image




st.title("📽️SceneFit recommendation system")
st.header("End the scroll. Start the show🍿🍿")

st.divider()

st.markdown("""
<style>
    .image-container img {
        transition: transform 0.3s ease; 
    }
    .image-container img:hover {
        transform: scale(1.21); 
        cursor: pointer; 
    }
</style>
""", unsafe_allow_html=True)
# ----------------------------------------

col1, col2, col3, = st.columns(3)

image_url1 = "https://pixieposters.co.uk/cdn/shop/files/Untitled1-gigapixel-art-scale-0_50x_233ff0ed-e019-4a58-9c29-eee41aff7a7f_2048x.jpg?v=1714070542"
image_url2 = "https://sgimage.netmarble.com/images/netmarble/sololv/20240105/je2f1704430223298.jpg"
image_url3 = "https://m.media-amazon.com/images/I/51CteVnYinL._AC_UF894,1000_QL80_.jpg"
with col1:
    st.markdown(f'<div class="image-container"><img src="{image_url1}" style="width: 100%; height: auto;"></div>', unsafe_allow_html=True)



with col2:
    st.markdown(f'<div class="image-container"><img src="{image_url2}" style="width: 100%; height: auto;"></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="image-container"><img src="{image_url3}" style="width: 100%; height: auto;"></div>', unsafe_allow_html=True)



image_url4 = "https://m.media-amazon.com/images/M/MV5BOGNjNGQ3MmItYTM5NS00NjBiLWI0ZTItZDE5ZjQyNjg3ODBjXkEyXkFqcGc@._V1_.jpg"
image_url5 = "https://m.media-amazon.com/images/I/91Cs3476iTL._AC_UF894,1000_QL80_.jpg"
image_url6 = "https://m.media-amazon.com/images/M/MV5BYWU5MmRjZjAtYjA3ZC00YjBkLWFiNGMtYTNhNmZlYzhlM2NkXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg"

with col1:
    st.markdown(f'<div class="image-container"><img src="{image_url4}" style="width: 100%; height: auto;"></div>', unsafe_allow_html=True)



with col2:
    st.markdown(f'<div class="image-container"><img src="{image_url5}" style="width: 100%; height: auto;"></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="image-container"><img src="{image_url6}" style="width: 100%; height: auto;"></div>', unsafe_allow_html=True)

st.divider()

st.write("""
Have you ever been bored? why am I even asking, You being here is solid evidence of boredom 😦.
But not to worry I can suggest to you any movie 🎥 just based on what you have loved before🤩. Sooo shall we!

""")

movie_titles_file_path = Path("finalMovieListTWO.txt")

movie_titles = []
try:

    raw_movie_titles_content = movie_titles_file_path.read_text(encoding='latin-1')

    movie_titles = [title.strip() for title in raw_movie_titles_content.split('\n') if title.strip()]
    st.spinner("Loading movie titles",)
except FileNotFoundError:
    st.write(f"Error: The file '{movie_titles_file_path}' was not found.")
except UnicodeDecodeError:
    st.write(
        f"Error: Could not decode the file '{movie_titles_file_path}' with the specified encoding. Try a different encoding.")
except Exception as e:
    st.write(f"An error occurred while loading movie titles: {e}")



client = genai.Client(api_key="AIzaSyCM2c3ZlfI7hqJ49yJ0AH378jSAMfV_RPM")

user_input = st.text_input("What movie do you think is a five star?")

user_embedding_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[user_input],
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
user_embedding = np.array(user_embedding_result.embeddings[0].values).reshape(1, -1)

recommendations = []

if movie_titles:

    st.write("You have good taste, let me see what I got")
    movie_embeddings_results = client.models.embed_content(
        model="gemini-embedding-001",
        contents=movie_titles,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    movie_embeddings = np.array([e.values for e in movie_embeddings_results.embeddings])

    similarity_scores = cosine_similarity(user_embedding, movie_embeddings).flatten()

    for i, movie in enumerate(movie_titles):
        recommendations.append((similarity_scores[i], movie))

    recommendations.sort(key=lambda x: x[0], reverse=True)

    user_input_lower = user_input.lower()
    filtered_recommendations = []
    for score, movie in recommendations:
        if movie.lower() != user_input_lower:
            filtered_recommendations.append((score, movie))

    st.write("\nYou should definately try these: ")
    for i in range(min(5, len(filtered_recommendations))):
        score, movie = filtered_recommendations[i]
        st.write(f"{i + 1}. '{movie}' (Similarity: {score * 100:.2f}%)")
else:
    st.write("Cannot generate recommendations: No movie titles were loaded.")