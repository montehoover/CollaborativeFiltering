
import math

import multiprocessing
import shelve
import time

g_data = {}
g_movie_ids = set()
g_total_movies = 0
g_averages = {}
g_averages_on_rated = {}
g_print_messages = True
g_total_tested = 0


def main():
    start = time.time()

    if g_print_messages:
        print("Starting to import data...")
    import_data('netflix_data/TrainingRatings.txt')
    if g_print_messages:
        print("Importing data completed in {:.2f} seconds".format(time.time() - start))
        print("Starting tests...")
    mean_absolute_error, root_mean_squared_error = get_test_results('PracticeTestingRatings.txt')
    print(mean_absolute_error, root_mean_squared_error)

    end = time.time()
    print("elapsed time:", end - start)


def predict_rating(user_id: str, movie_id: str) -> float:
    """
    Predicts the rating a user will give to a movie based on ratings of similar users. Similarity is measured
    by the Pearson Correlation Coefficient.
    :param user_id: String ID of user for which to predict
    :param movie_id: String ID of movie for which to predict
    :returns: Float value between 0.0 and 5.0
    """
    votes = get_all_weighted_votes(user_id, movie_id)
    alpha = 1 / sum([abs(x) for x in votes])
    return avg_rating_on_rated(user_id) + alpha * sum(votes)


def get_all_weighted_votes(user_id: str, movie_id: str) -> list:
    """
    Returns a list of the rating votes, where each vote is the rating that a user in the database has assigned to the
    given movie_id. Each vote normalized by subtracting the user's average rating and then weighted by how similar that
    user is to the target user for which we are predicting. Similarity is measured by the Pearson Correlation
    Coefficient. Note that users who have not rated the movie are counted as an implicit '0.0' vote, but their
    normalization factor is their average only over movies they have rated (see writeup for justification for this
    choice).
    :param user_id:
    :param movie_id:
    :return: list of weighted rating votes
    """
    counter = 0
    for user in g_data.keys():
        counter += 1
        if counter >= 5:
            break
        p = pearson_coefficient(user_id, user)
        actual = g_data[user].get(movie_id, 0.0)
        avg = avg_rating_on_rated(user)
        print("pearson:", p, "actual:", actual, "avg:", avg)

    return [pearson_coefficient(user_id, user) * (g_data[user].get(movie_id, 0.0) - avg_rating_on_rated(user))
            for user in g_data.keys()]


def pearson_coefficient(user_a: str, user_b: str) -> float:
    """
    :param user_a:
    :param user_b:
    :return: float representing measurement of similarity between two vectors of user ratings.
    """
    for movie in g_movie_ids:
        if movie in g_data[user_a] and movie in g_data[user_b]:

    sum_of_devations = sum([(g_data[user_a].get(movie, 0.0) - avg_rating(user_a)) *
                            (g_data[user_b].get(movie, 0.0) - avg_rating(user_b))
                            for movie in g_movie_ids])
    sum_of_squares_a = sum([(g_data[user_a].get(movie, 0.0) - avg_rating(user_a)) ** 2
                            for movie in g_movie_ids])
    sum_of_squares_b = sum([(g_data[user_b].get(movie, 0.0) - avg_rating(user_b)) ** 2
                            for movie in g_movie_ids])
    denominator = math.sqrt(sum_of_squares_a * sum_of_squares_b)
    if denominator == 0:
        return 0
    return sum_of_devations / denominator


def avg_rating(user_id: str) -> float:
    """
    :param user_id:
    :return: Average rating a given user has assigned to movies, treating unrated movies as having an implicit rating of
    zero.
    """
    global g_averages
    if user_id not in g_averages:
        # Atomic operation so threadsafe
        g_averages[user_id] = sum(g_data[user_id].values()) / g_total_movies
    return g_averages[user_id]


def avg_rating_on_rated(user_id: str) -> float:
    """
    :param user_id:
    :return: average rating a given user has assigned to movies, only counting movies that the user has actually rated
    """
    global g_averages_on_rated
    if user_id not in g_averages_on_rated:
        # Atomic operation so threadsafe
        g_averages_on_rated[user_id] = sum(g_data[user_id].values()) / max(len(g_data[user_id]), 1)
    return g_averages_on_rated[user_id]


def get_error_from_prediction(movie_user_rating: tuple) -> float:
    """
    Used to map predict_rating() over list of test instances and return the error from the actual rating value.
    :param movie_user_rating: Tuple of Str movie_id, Str user_id, int rating
    :return: Value between 0.0 and 5.0
    """
    p = predict_rating(movie_user_rating[1], movie_user_rating[0])
    print("actual: {:.2f}, predicted: {:.2f}, avg: {:.2f}".format(movie_user_rating[2],  p, avg_rating_on_rated(movie_user_rating[1])))
    global g_total_tested
    g_total_tested += 1
    return movie_user_rating[2] - p


def get_test_results(file_name: str) -> tuple:
    """
    Runs test predictions against movie rating instances from a test file.
    :param file_name:
    :return: tuple of Mean Absolute Error and Root Mean Squared Error
    """
    movie_user_rating_tuples = []
    with open(file_name) as f:
        for line in f:
            movie_id, user_id, rating = line.strip().split(',')
            movie_user_rating_tuples.append((movie_id, user_id, float(rating)))

    thread_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    errors = thread_pool.map(get_error_from_prediction, movie_user_rating_tuples)
    print(errors)

    mean_absolute_error = sum([abs(x) for x in errors]) / max(len(errors), 1)
    root_mean_squared_error = math.sqrt(sum([x**2 for x in errors]) / max(len(errors), 1))
    return mean_absolute_error, root_mean_squared_error


def import_data(file_name: str) -> None:
    """
    Imports movie ratings from an input file and loads them into a dict where the keys are user_id strings and the
    values are dicts. The nested dicts have key, value pairs of the form: str movie_id, float rating
    :param d:
    :param file_name:
    """
    start = time.time()
    with open(file_name) as f:
        global g_data
        global g_movie_ids
        global g_total_movies
        counter = 0
        for line in f:
            movie_id, user_id, rating = line.strip().split(',')
            if user_id not in g_data:
                g_data[user_id] = {movie_id: float(rating)}
            else:
                g_data[user_id][movie_id] = float(rating)
            # MOVIE_IDS is a set, so multiple instances of the same movie_id will not added
            g_movie_ids.add(movie_id)
            counter +=1
            if counter % 500000 == 0:
                checkpoint = time.time()
                print("Finished importing {} instances in {} seconds".format(counter, checkpoint - start))
        g_total_movies = len(g_movie_ids)

if __name__ == '__main__':
    main()