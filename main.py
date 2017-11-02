# 1. k-NN for movie prediction. prediction=0/1 "Rating", k=1, Nearest=most alike ratings
# 2. Same, but k=5. prediction=avg of yes/no (most common)
# 3. Same, but k votes are weighted. k = all
# 4. Same, but rating is now between 0-5.

# How should missing values be calculated in the average?  Added in or left out?
import math

import multiprocessing

DATA = {}
TOTAL_MOVIES = 20
MOVIE_IDS = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
AVERAGES = {}
AVERAGES_ON_RATED = {}


def main():
    global DATA
    DATA = import_data('PracticeRatings.txt')
    mean_absolute_error, root_mean_squared_error = get_test_results('PracticeTestRatings.txt')
    print(mean_absolute_error, root_mean_squared_error)



def predict_rating(user_id: str, movie_id: str) -> float:
    """
    Predicts the rating a user will give to a movie based on ratings of similar users. Similarity is measured
    by the Pearson Correlation Coefficient.

    :param user_id: String ID of user for which to predict
    :param movie_id: String ID of movie for which to predict
    :returns: Float value between 0.0 and 5.0
    """
    alpha = 1 / len(DATA.keys())
    return avg_rating_on_rated(user_id) + alpha * sum(get_all_weighted_votes(user_id, movie_id))


# def predict_rating_wrapper(user_movie_tuple: tuple) -> float:
#     """
#     Wraps predict_rating() so it can called by map().
#
#     :param user_movie_tuple: Two-item tuple of user_id and movie_id to be passed into predict_rating.
#     :returns: Float value between 0.0 and 5.0 that is a prediction of what the user will rate the movie.
#     """
#     return predict_rating(user_movie_tuple[0], user_movie_tuple[1])


def avg_rating(user_id):
    global AVERAGES
    if user_id not in AVERAGES:
        # Atomic operation so threadsafe
        AVERAGES[user_id] = sum(DATA[user_id].values()) / TOTAL_MOVIES
    return AVERAGES[user_id]


def avg_rating_on_rated(user_id):
    global AVERAGES_ON_RATED
    if user_id not in AVERAGES_ON_RATED:
        # Atomic operation so threadsafe
        AVERAGES_ON_RATED[user_id] = sum(DATA[user_id].values()) / max(len(DATA[user_id]), 1)
    return AVERAGES_ON_RATED[user_id]


def get_all_weighted_votes(user_id, movie_id):
    return [pearson_coefficient(user_id, user) * (DATA[user].get(movie_id, 0) - avg_rating_on_rated(user)) for user in DATA.keys()]


def pearson_coefficient(user_a, user_b):
    sum_of_devations = sum([(DATA[user_a].get(movie, 0) - avg_rating(user_a)) * (DATA[user_b].get(movie, 0) - avg_rating(user_b)) for movie in MOVIE_IDS])
    sum_of_squares_a = sum([(DATA[user_a].get(movie, 0) - avg_rating(user_a)) ** 2 for movie in MOVIE_IDS])
    sum_of_squares_b = sum([(DATA[user_b].get(movie, 0) - avg_rating(user_b)) ** 2 for movie in MOVIE_IDS])
    denomenator = math.sqrt(sum_of_squares_a * sum_of_squares_b)
    if denomenator == 0:
        return 0
    return sum_of_devations / denomenator


def get_error_from_prediction(movie_user_rating: tuple) -> float:
    """
    Used to map predict_rating() over list of test instances and return the error from the actual rating value.
    :param movie_user_rating: Tuple of Str movie_id, Str user_id, int rating
    :return: Value between 0.0 and 5.0
    """
    return movie_user_rating[2] - predict_rating(movie_user_rating[1], movie_user_rating[0])


def get_test_results(file_name):
    errors = []
    movie_user_rating_tuples = []
    with open(file_name) as f:
        for line in f:
            movie_id, user_id, rating = line.strip().split(',')
            movie_user_rating_tuples.append((movie_id, user_id, int(rating)))
            # prediction = predict_rating(user_id, movie_id)
            # errors.append(int(rating) - prediction)

    thread_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    errors = thread_pool.map(get_error_from_prediction, movie_user_rating_tuples)

    mean_absolute_error = sum([abs(x) for x in errors]) / max(len(errors), 1)
    root_mean_squared_error = math.sqrt(sum([x**2 for x in errors]) / max(len(errors), 1))
    return mean_absolute_error, root_mean_squared_error


def import_data(file_name):
    d = {}
    with open(file_name) as f:
        for line in f:
            movie_id, user_id, rating = line.strip().split(',')
            if user_id not in d:
                d[user_id] = {movie_id: int(rating)}
            else:
                d[user_id][movie_id] = int(rating)
    return d


if __name__ == '__main__':
    main()