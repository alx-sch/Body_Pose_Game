from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors

example_query = {
    # movieId, rating
    4470: 5,
    48: 5,
    594: 5,
    27619: 5,
    152081: 5,
    595: 5,
    616: 5,
    1029: 5,
}


##############
##############


def title_finder(recommended_movieIDs, movies_df):
    """
    Takes recommendations from recommender functions (index objects type with movie IDs.
    Returns a DF showing recommended movie titles w/ genres.
    """
    recommended_titles_df = movies_df.set_index("movieId").loc[recommended_movieIDs]
    recommended_titles_list = recommended_titles_df["title"].tolist()

    return recommended_titles_list


##############
##############


def link_finder(recommended_movieIDs, movies_df):
    """
    Takes recommendations from recommender functions (index objects type with movie IDs.
    Returns a DF showing recommended movie titles w/ genres.
    """
    recommended_titles_df = movies_df.set_index("movieId").loc[recommended_movieIDs]
    recommended_titles_list = recommended_titles_df["movie_link"].tolist()

    return recommended_titles_list


##############
##############


def filter_popular(ratings_df, k=20):
    """
    Filters out movies rated by less than k users.
    Returns filtered ratings DF.

    :ratings_df: DF containing ratings
    :k: filter - min number of users that rated a movie (int)
    """

    # calculates the number of ratings per movie
    ratings_per_movie = ratings_df.groupby("movieId")["userId"].count()

    # filtesr for movies with more than k ratings and extract the index
    popular_movies = ratings_per_movie.loc[ratings_per_movie > k].index

    # filters the ratings matrix and only keep the popular movies
    ratings_filtered_df = ratings_df.loc[ratings_df["movieId"].isin(popular_movies)]

    return ratings_filtered_df


##############
##############


def filter_rating(ratings_df, k=2):
    """
    Filters out movies with an average rating of less than k.
    Returns filtered ratings DF.

    :ratings_df: DF containing ratings
    :k: filter - min average rating (float)
    """

    # Calculatea the average rating for each movie title
    mean_rating_score_per_movie = ratings_df.groupby("movieId")["rating"].mean()

    # filtesr for movies with more than k ratings and extract the index
    high_rating_movies = mean_rating_score_per_movie.loc[
        mean_rating_score_per_movie > k
    ].index

    # filters the ratings matrix and only keep the popular movies
    ratings_filtered_df = ratings_df.loc[ratings_df["movieId"].isin(high_rating_movies)]

    return ratings_filtered_df


##############
##############


def get_ratings_matrix(ratings):
    """
    Returns a sparse user-item rating matrix shape -> (row: user_id, column: movie_id, value: rating)
    """
    R = csr_matrix((ratings["rating"], (ratings["userId"], ratings["movieId"])))

    return R


##############
##############


def make_user_vector(query, length):
    # new user vector: needs to have the same format as the training data
    # pre fill it with zeros
    user_vec = np.repeat(0, length)

    # fill in ratings from the query
    user_vec[list(query.keys())] = list(query.values())

    return user_vec


##############
##############


def recommend_random(query, ratings_df, k=6):
    """
    Filters and recommends k random movies for any given input query.
    Returns a list of k movie ids
    """

    # filter out movies that have already been rated
    not_rated_movies = ratings_df.set_index("movieId").drop(query.keys())

    # randomize recommendations
    recommendations = not_rated_movies.sample(k).index
    recommendations = recommendations.tolist()

    return recommendations


##############
##############


def recommend_popular(query, ratings_df, k=6):
    """
    Filters and recommends the k most popular movies (most ratings).
    Returns a list of k movie ids
    """

    # sort movies by no. of ratings
    ratings_per_movie = (
        ratings_df.groupby("movieId")["userId"].count().sort_values(ascending=False)
    )

    # filter out movies that had already been rated
    movies_to_drop = list(query.keys())
    ratings_per_movie_filtered = ratings_per_movie.drop(index=movies_to_drop)

    # recommend top k movieIds
    recommendations = ratings_per_movie_filtered.head(k).index
    recommendations = recommendations.tolist()

    return recommendations


##############
##############


def recommend_nmf(query, model, ratings_df, k=6):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model.
    Returns recommendations as a list of k movie ids.
    """
    # 1. candidate generation

    ratings = filter_popular(ratings_df)
    ratings = filter_rating(ratings)

    # (data, (row_ind, col_ind))
    R = csr_matrix((ratings["rating"], (ratings["userId"], ratings["movieId"])))

    # construct a user vector
    user_vec = make_user_vector(query, R.shape[1])
    user_vec_sparse = csr_matrix(user_vec)

    # 2. scoring

    # calculate the score with the NMF model

    # user_vec -> encoding -> p_user_vec -> decoding -> user_vec_hat
    scores = model.inverse_transform(
        model.transform(user_vec_sparse)
    )  # model.transform(user_vec) -> user matrix then score for rest of movies

    # convert to a pandas series
    scores = pd.Series(scores[0])

    # 3. ranking

    # filter out movies allready seen by the user
    # give a zero score to movies the user has already seen
    scores[query.keys()] = 0

    # sort the scores from high to low
    scores = scores.sort_values(ascending=False)

    # return the top-k highst rated movie ids or titles
    # get the movieIds of the top k entries
    recommendations = scores.head(k).index
    recommendations = recommendations.tolist()

    return recommendations


##############
##############


def recommend_cosine(query, model, ratings_df, k=6):
    """
    Filters and recommends the top k movies for any given input query based on a
    trained Neighborhood-based model using Cosine similarity (collaborative  filtering).
    Returns recommendations as a list of k movie ids.
    """
    # 1. candidate generation

    ratings = filter_popular(ratings_df)
    ratings = filter_rating(ratings)

    # (data, (row_ind, col_ind))
    R = csr_matrix((ratings["rating"], (ratings["userId"], ratings["movieId"])))

    # construct a user vector
    user_vec = make_user_vector(query, R.shape[1])

    # 2. scoring

    # calculates the distances to all other users in the data!
    distances, userIds = model.kneighbors(
        [user_vec], n_neighbors=20, return_distance=True
    )

    # sklearn returns a list of predictions - extract the first and only value of the list
    distances = distances[0]
    userIds = userIds[0]

    # only look at ratings for users that are similar!
    neighborhood = ratings.set_index("userId").loc[userIds]

    # calculate the summed up rating for each movie
    # summing up introduces a bias for popular movies
    # averaging introduces bias for movies only seen by few users in the neighboorhood
    scores = neighborhood.groupby("movieId")["rating"].sum()

    # 3. ranking

    # give a zero score to movies the user has allready seen
    allready_seen = scores.index.isin(query.keys())
    scores.loc[allready_seen] = 0

    # sort the scores from high to low
    scores = scores.sort_values(ascending=False)

    # return the top-k highst rated movie ids or titles
    # get the movieIds of the top k entries
    recommendations = scores.head(k).index
    recommendations = recommendations.tolist()

    return recommendations
