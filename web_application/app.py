# import std libraries
import numpy as np
import pandas as pd

from IPython.display import HTML
import pickle
import json

from utils import (
    recommend_random,
    recommend_popular,
    recommend_nmf,
    recommend_cosine,
    link_finder,
)

import streamlit as st
from st_aggrid import AgGrid

BEST_MOVIES = pd.read_csv("./Streamlit_app/best_movies.csv")
BEST_MOVIES.rename(index=lambda x: x + 1, inplace=True)
TITLES = ["---"] + list(BEST_MOVIES["title"].sort_values())

ratings = pd.read_csv("./data/ml-latest-small/ratings.csv")
movies = pd.read_csv("./data/ml-latest-small/movie_links.csv")

# with open('distance_recommender.pkl', 'rb') as file:
#    DISTANCE_MODEL = pickle.load(file)

# with open('nmf_recommender.pkl', 'rb') as file:
#    NMF_MODEL = pickle.load(file)

# sidebar
with st.sidebar:
    # title
    st.title("It's movie time!")
    # image
    st.image("./Streamlit_app/movie_time.jpg")
    # blank space
    st.write("")
    # selectbox
    page = st.selectbox(
        "what would you like?",
        ["welcome page", "popular movies", "rate some movies", "recommended movies"],
    )

##########################################################
# Welcome Page
##########################################################

if page == "welcome page":
    # slogan
    st.write(
        """
    *Movies are like magic tricks (Jeff Bridges)*
    """
    )
    # blank space
    st.write("")
    # image
    st.image("./Streamlit_app/movie_pics.png")

##########################################################
# Popular Movies
##########################################################

elif page == "popular movies":
    # title
    st.title("Popular Movies")
    col1, col2, col3, col4 = st.columns(
        [10, 1, 5, 5]
    )  # relative size; col2 -> space filler
    with col1:
        n = st.slider(label="how many movies?", min_value=1, max_value=10, value=5)
    with col3:
        st.markdown("####")  # introduces empty space
        genre = st.checkbox("include genres")
    with col4:
        st.markdown("###")
        show_button = st.button(label="show movies")

    if genre:
        popular_movies = BEST_MOVIES[["movie_title", "genres"]]
    else:
        popular_movies = BEST_MOVIES[["movie_title"]]

    st.markdown("###")
    if show_button:
        st.write(HTML(popular_movies.head(n).to_html(escape=False)))

##########################################################
# Rate Movies
##########################################################

elif page == "rate some movies":
    # title
    st.title("Rate Movies")
    #
    col1, col2, col3 = st.columns([10, 1, 5])
    with col1:
        m1 = st.selectbox("movie 1", TITLES)
        st.write("")
        m2 = st.selectbox("movie 2", TITLES)
        st.write("")
        m3 = st.selectbox("movie 3", TITLES)
        st.write("")
        m4 = st.selectbox("movie 4", TITLES)
        st.write("")
        m5 = st.selectbox("movie 5", TITLES)

    with col3:
        r1 = st.slider(label="rating 1", min_value=1, max_value=5, value=3)
        r2 = st.slider(label="rating 2", min_value=1, max_value=5, value=3)
        r3 = st.slider(label="rating 3", min_value=1, max_value=5, value=3)
        r4 = st.slider(label="rating 4", min_value=1, max_value=5, value=3)
        r5 = st.slider(label="rating 5", min_value=1, max_value=5, value=3)

    query_movies = [m1, m2, m3, m4, m5]
    query_ratings = [r1, r2, r3, r4, r5]

    user_query = dict(zip(query_movies, query_ratings))

    # get user query
    st.markdown("###")
    user_query_button = st.button(label="save user query")
    if user_query_button:
        json.dump(user_query, open("user_query.json", "w"))
        st.write("")
        st.write("user query saved successfully")

##########################################################
# Movie Recommendations
##########################################################
else:
    # title
    # st.title("Your **Personalized** Movie Recommendations")
    title_text = "<h2 style='text-align: center;'>Your <span style='color: red;'>Personalized</span> Movie Recommendations</h2>"
    st.markdown(title_text, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([5, 1, 7, 5])

    with col1:
        n_recommendations = st.slider(
            label="Recommend how many movies?", min_value=1, max_value=20, value=10
        )

    with col3:
        recommender = st.radio(
            "recommender type",
            [
                "Random Recommender",
                "Popular Recommender",
                "NMF Recommender",
                "Distance Recommender",
            ],
        )
    with col4:
        st.write("###")
        recommend_button = st.button(label="recommend movies")

    if recommend_button:
        # load user query
        user_query = json.load(open("user_query.json"))

        # translate movie titles to movieId in query (as this was used in model training / recommender functions defined this way)
        query_with_ids = {
            movies.loc[movies["title"] == k, "movieId"].values[0]: v
            for k, v in user_query.items()
        }

        if recommender == "Random Recommender":
            recommendations = recommend_random(
                query_with_ids, ratings, k=n_recommendations
            )

        elif recommender == "Popular Recommender":
            recommendations = recommend_popular(
                query_with_ids, ratings, k=n_recommendations
            )

        elif recommender == "NMF Recommender":
            with open("./models/nmf_recommender.pkl", "rb") as file:
                model_nmf = pickle.load(file)

            recommendations = recommend_nmf(
                query_with_ids, model_nmf, ratings, k=n_recommendations
            )

        elif recommender == "Distance Recommender":
            with open("./models/nn_cosine_recommender.pkl", "rb") as file:
                cosine_nmf = pickle.load(file)

            recommendations = recommend_cosine(
                query_with_ids, cosine_nmf, ratings, k=n_recommendations
            )

        recommendations_title = link_finder(recommendations, movies)

        # st.title("List with Rankings")
        st.markdown(
            f"<h4>Your Top {n_recommendations} Moviiie Recommendations:</h4>",
            unsafe_allow_html=True,
        )

        # Iterate through the list using enumerate
        for rank, item in enumerate(recommendations_title, start=1):
            # Display each item with its corresponding ranking in bullet points
            st.markdown(f"{rank}. {item}", unsafe_allow_html=True)
