# 1. k-NN for movie prediction. prediction=0/1 "Rating", k=1, Nearest=most alike ratings
# 2. Same, but k=5. prediction=avg of yes/no (most common)
# 3. Same, but k votes are weighted. k = all
# 4. Same, but rating is now between 0-5.

# How should missing values be calculated in the average?  Added in or left out?
import math

DATA = {}
# TOTAL_MOVIES = 20
# MOVIE_IDS = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
TOTAL_MOVIES = 4
MOVIE_IDS = ['8', '28', '9', '10']
# TOTAL_MOVIES = 2
# MOVIE_IDS = ['8', '28']


def main():
    # import_data()
    global DATA
    DATA = {'1': {'8': 3, '28': 4, '9': 2},
            '2': {'8': 3, '28': 4, '9': 1, '10': 1},
            '3': {'8': 3, '28': 4, '9': 2, '10': 3}
            }
    # DATA = {'1': {'8': 2},
    #         '2': {'8': 3, '28': 4},
    #         '3': {'8': 2, '28': 4}
    #         }

    print(DATA['1'])
    # print(DATA['2057'])
    # print(DATA['529'])
    # print(DATA['1002'])
    # print(DATA['828'])
    # print(DATA['102'])
    print(len(DATA))
    # 529, 1002, 828
    # predict_rating('529', '3')
    # print(avg_rating('529'))
    print(pearson_coefficient('1', '2'))
    print()
    print(pearson_coefficient('1', '3'))


def predict_rating(user_id, movie_id):
    alpha = 1
    return avg_rating(user_id) + alpha * sum(get_all_weighted_votes(user_id, movie_id))


def avg_rating(user_id):
    return sum(DATA[user_id].values()) / TOTAL_MOVIES #max(len(DATA[user_id]), 1)


def get_all_weighted_votes(user_id, movie_id):
    return [pearson_coefficient(user_id, user) * (DATA[user].get(movie_id, 0) - avg_rating(user)) for user in DATA.keys()]


def pearson_coefficient(user_a, user_b):
    for movie in MOVIE_IDS:
        print('a:', DATA[user_a].get(movie, 0), 'b:', DATA[user_b].get(movie, 0))
    sum_of_devations = sum([(DATA[user_a].get(movie, 0) - avg_rating(user_a)) * (DATA[user_b].get(movie, 0) - avg_rating(user_b)) for movie in MOVIE_IDS])
    print(sum_of_devations)
    sum_of_squares_a = sum([(DATA[user_a].get(movie, 0) - avg_rating(user_a)) ** 2 for movie in MOVIE_IDS])
    print(sum_of_squares_a)
    sum_of_squares_b = sum([(DATA[user_b].get(movie, 0) - avg_rating(user_b)) ** 2 for movie in MOVIE_IDS])
    print(sum_of_squares_b)
    denomenator = math.sqrt(sum_of_squares_a * sum_of_squares_b)
    print(denomenator)
    if denomenator == 0:
        return 0
    return sum_of_devations / denomenator



def import_data():
    with open('PracticeRatings.txt') as f:
        for line in f:
            movie_id, user_id, rating = line.strip().split(',')
            global DATA
            if user_id not in DATA:
                DATA[user_id] = {movie_id: int(rating)}
            else:
                DATA[user_id][movie_id] = int(rating)


if __name__ == '__main__':
    main()