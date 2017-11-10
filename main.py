import heapq
import math
import multiprocessing
import time

import pickle

g_data = {}
g_movie_ids = set()
g_total_movies = 0
g_averages = {}
g_averages_on_rated = {}
g_print_messages = True
g_total_tested = 0
g_ball_tree = None


def main():
    start = time.time()

    if g_print_messages:
        print("Starting to import data...")

    import_data('netflix_data/TrainingRatings.txt')
    original_data = g_data
    global g_data
    g_data = remove_all_features_from_data()

    selected_features = set({'6336', '8260', '7505', '6917', '851', '15331', '2116', '5970', '1845', '11380', '8121', '11442', '7503', '11245', '145', '3538', '12623', '1444', '10743', '3740', '12780', '11011', '13638', '7145', '15687', '1104', '12778', '3904', '2130', '11376', '638', '12336', '4640', '12189', '9410', '4299', '15475', '3928', '11159', '13858', '1140', '6911', '4546', '12795', '14930', '4264', '6748', '1893', '5656', '3057', '5604', '12875', '17536', '1048', '15152', '1825', '5623', '14086', '10757', '17344', '9689', '1832', '14947', '8596', '6567', '9550', '14904', '6541', '8488', '8994', '2504', '2939', '797', '17310', '13587', '17255', '3054', '15582', '12243', '14625', '14505', '15820', '4002', '5699', '7188', '4750', '11923', '13055', '11887', '9697', '8718', '2955', '8493', '1192', '8207', '5941', '156', '5562', '5069', '11235', '6543', '12482', '398', '5294', '5342', '16670', '3274', '2658', '1482', '5098', '15670', '5267', '15998', '1167', '2355', '8023', '9716', '6228', '634', '2251', '12503', '16162', '16122', '1924', '3733', '140', '4385', '14209', '16741', '8005', '1500', '417', '7776', '11237', '2749', '12639', '7544', '15992', '6979', '16157', '14537', '4268', '12755', '2110', '318'})
    for feature in selected_features:
        add_feature_to_data(original_data, feature)

    #########################
    # Create Ball Tree
    #
    # global g_ball_tree
    # g_ball_tree = BallTree(g_data.keys())
    # with open('ball_tree.pickle', 'wb') as f:
    #     pickle.dump(g_ball_tree, f)

    if g_print_messages:
        print("Importing data completed in {:.2f} seconds".format(time.time() - start))
        print("Starting tests...")

    #########################
    # Get test results
    #
    mean_absolute_error, root_mean_squared_error = get_test_results('netflix_data/TestingRatings.txt', 100)
    print(mean_absolute_error, root_mean_squared_error)

    end = time.time()
    print("elapsed time:", end - start)

    print("Predictions for Monte:")
    predictions = []
    for movie in g_movie_ids:
        predictions.append((predict_rating('monte', movie), movie))
    print(reversed(sorted(predictions)))


def predict_rating(user_id: str, movie_id: str, k: int) -> float:
    """
    Predicts the rating a user will give to a movie based on ratings of similar users. Similarity is measured
    by the Pearson Correlation Coefficient.
    :param user_id: String ID of user for which to predict
    :param movie_id: String ID of movie for which to predict
    :returns: Float value between 0.0 and 5.0
    """
    if k == None:
        votes = get_all_weighted_votes(user_id, movie_id)
    else:
        votes = get_knn_weighted_votes(user_id, movie_id, k)
    abs_sum = sum([abs(x) for x in votes])
    if abs_sum != 0:
        alpha = 1 / abs_sum
    else:
        alpha = 0
    s = sum(votes)
    return avg_rating_on_rated(user_id) + alpha * sum(votes)


def get_knn_weighted_votes(user_id, movie_id, k):
    knn = g_ball_tree.get_knn(user_id, k)
    return [pearson_coefficient(user_id, user) * (g_data[user].get(movie_id) - avg_rating_on_rated(user))
            for user in knn
            if movie_id in g_data[user]]


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

    v = [pearson_coefficient(user_id, user) * (g_data[user].get(movie_id) - avg_rating_on_rated(user))
            for user in g_data.keys()
            if movie_id in g_data[user]]

    return v


def pearson_coefficient(user_a: str, user_b: str) -> float:
    """
    :param user_a:
    :param user_b:
    :return: float representing measurement of similarity between two vectors of user ratings.
    """
    sum_of_devations = sum([(g_data[user_a].get(movie) - avg_rating_on_rated(user_a)) *
                            (g_data[user_b].get(movie) - avg_rating_on_rated(user_b))
                            for movie in g_movie_ids
                            if movie in g_data[user_a] and movie in g_data[user_b]])
    sum_of_squares_a = sum([(g_data[user_a].get(movie) - avg_rating_on_rated(user_a)) ** 2
                            for movie in g_movie_ids
                            if movie in g_data[user_a] and movie in g_data[user_b]])
    sum_of_squares_b = sum([(g_data[user_b].get(movie) - avg_rating_on_rated(user_b)) ** 2
                            for movie in g_movie_ids
                            if movie in g_data[user_a] and movie in g_data[user_b]])

    # sum_of_devations = sum([(g_data[user_a].get(movie, 0.0) - avg_rating(user_a)) *
    #                         (g_data[user_b].get(movie, 0.0) - avg_rating(user_b))
    #                         for movie in g_movie_ids])
    # sum_of_squares_a = sum([(g_data[user_a].get(movie, 0.0) - avg_rating(user_a)) ** 2
    #                         for movie in g_movie_ids])
    # sum_of_squares_b = sum([(g_data[user_b].get(movie, 0.0) - avg_rating(user_b)) ** 2
    #                         for movie in g_movie_ids])
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


def remove_all_features_from_data() -> dict:
    new_data = {}
    for user in g_data:
        new_data[user] = {}
    return new_data


def add_feature_to_data(original_data: dict, movie_id: str) -> None:
    for user in original_data:
        if movie_id in original_data[user]:
            global g_data
            g_data[user][movie_id] = original_data[user][movie_id]


def remove_feature_from_data(removed: str) -> None:
    for user in g_data:
        global g_data
        g_data[user].pop(removed, None)


def get_error_from_prediction(movie_user_rating_k: tuple) -> float:
    """
    Used to map predict_rating() over list of test instances and return the error from the actual rating value.
    :param movie_user_rating_k: Tuple of Str movie_id, Str user_id, int rating
    :return: Value between 0.0 and 5.0
    """
    p = predict_rating(movie_user_rating_k[1], movie_user_rating_k[0], movie_user_rating_k[3])
    # print("actual: {:.2f}, predicted: {:.2f}, avg: {:.2f}".format(movie_user_rating[2],  p, avg_rating_on_rated(movie_user_rating[1])))
    global g_total_tested
    g_total_tested += 1
    return movie_user_rating_k[2] - p


def get_test_results(file_name: str, k=None) -> tuple:
    """
    Runs test predictions against movie rating instances from a test file.
    :param file_name:
    :return: tuple of Mean Absolute Error and Root Mean Squared Error
    """
    movie_user_rating_tuples = []
    with open(file_name) as f:
        for line in f:
            movie_id, user_id, rating = line.strip().split(',')
            movie_user_rating_tuples.append((movie_id, user_id, float(rating), k))

    thread_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    errors = thread_pool.map(get_error_from_prediction, movie_user_rating_tuples)

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

        monte = {'13': 5.0, '30': 4.0, '48': 5.0, '138': 5.0, '191': 4.0, '209': 5.0, '589': 5.0, '613': 4.0, '758': 4.0, '1008': 5.0, '1180': 5.0, '1208': 4.0, '1291': 4.0}
        g_data['monte'] = monte
        g_total_movies = len(g_movie_ids)


def feature_selection():
    selected_features = set()
    original_data = g_data
    global g_data
    g_data = remove_all_features_from_data()
    mae, baseline_rmse = get_test_results('ValidationTrainingRatings.txt')
    num_features = 0
    for movie in g_movie_ids:

        if movie not in selected_features:
            add_feature_to_data(original_data, movie)
            new_mae, new_rmse = get_test_results('ValidationTrainingRatings.txt')

            if new_rmse < baseline_rmse:
                selected_features.add(movie)
                baseline_rmse = new_rmse
                print("added new feature")

            if num_features >= 145:
                break

    return selected_features


class BallTree():
    def __init__(self, users: iter):
        self.leaf_value = None
        self.pivot = None
        self.radius = None
        self.left_child = None
        self.right_child = None

        if len(users) == 1:
            self.leaf_value = list(users)[0]
            return

        movie = self.find_movie_to_split_on(users)
        self.pivot = self.find_pivot_user(users, movie)
        pivot_val = g_data[self.pivot][movie]
        self.radius = max([self.distance(self.pivot, user) for user in users])

        left_child_users = [x for x in users if g_data[x].get(movie, 0.0) > pivot_val]
        if len(left_child_users) != 0:
            self.left_child = BallTree(left_child_users)
        else:
            self.left_child = None

        right_child_users = [x for x in users if g_data[x].get(movie, 0.0) <= pivot_val]
        if len(right_child_users) != 0:
            self.right_child = BallTree(right_child_users)
        else:
            self.right_child_users = None

    def get_knn(self, target, k):
        q = []
        self.get_knn_with_queue(target, k, q)
        return [x[1] for x in q]

    def get_knn_with_queue(self, target, k, q) -> None:
        if self.distance(target, self.pivot) >= q[0][0]:
            return

        if self.leaf_value != None:
            d = self.distance((target, self.leaf_value))
            if len(q) > 0:
                if d < q[0][0]:
                    heapq.heappush(q, (d, self.leaf_value))
            else:
                heapq.heappush(q, (d, self.leaf_value))
            if len(q) > k:
                self.remove_furthest(q)


        if self.left_child.pivot != None and self.right_child.pivot != None:
            d_left = self.distance(target, self.left_child.pivot)
            d_right = self.distance(target, self.right_child.pivot)
        else:
            # Arbitrarily declare one closer since they are both leaf nodes
            d_left = 1
            d_right = 2

        if d_left < d_right:
            self.left_child.get_knn_with_queue(target, k, q)
            self.right_child.get_knn_with_queue(target, k, q)
        else:
            self.right_child.get_knn_with_queue(target, k, q)
            self.left_child.get_knn_with_queue(target, k, q)


    def distance(self, a, b):
        return 1 - pearson_coefficient(a, b)

    def remove_furthest(self, q):
        newq = []
        while len(q) > 1:
            heapq.heappush(newq, heapq.heappop(q))
        # newq is all but the last item in q, thus removed the furthest item in q
        q = newq

    def find_movie_to_split_on(self, users):
        """
        Here we are choosing which attribute (or movie) axis on which to split the ball tree. We are choosing the most
        frequent movie, but a better choice would be to evaluate on information gain.
        :param users:
        :return:
        """
        tallys = {}
        for movie in g_movie_ids:
            for user in users:
                if movie in g_data[user]:
                    tallys[movie] = 1 + tallys.setdefault(movie, 0)
        return max(tallys.items(), key=lambda x: x[1])[0]

    def find_pivot_user(self, users, movie):
        #TODO: Fix this to correctly return median accounting for urated movies in feature vector
        rating_user_tuples = [(g_data[user][movie], user) for user in users if movie in g_data[user]]
        median_index = len(rating_user_tuples) // 2
        user_with_median_rating = sorted(rating_user_tuples)[median_index][1]

        return user_with_median_rating




if __name__ == '__main__':
    main()