# 1. k-NN for movie prediction. prediction=0/1 "Rating", k=1, Nearest=most alike ratings
# 2. Same, but k=5. prediction=avg of yes/no (most common)
# 3. Same, but k votes are weighted. k = all
# 4. Same, but rating is now between 0-5.

# judging ratings from unwatched/unrated movies will be tricky, but maybe the most useful.
# think about two peole that have watched totally different sets of movies - they are probably
# not very similar. Maybe try a way to make them rate lower even than if they have opposite
# ratings on the same movies. but be careful of people who have seen a lot of movies. in that
# case don't penalize.

import collections
import random


def main():
    people = []
    people.append({'Finding Nemo': 1, 'Robin Hood': 1, 'Jungle Book': 1, 'Mulan': 0, 'Little Mermaid': 0, 'Cinderella': 0, 'Snow White': 0})
    people.append({'Finding Nemo': 0, 'Robin Hood': 0, 'Jungle Book': 0, 'Mulan': 1, 'Little Mermaid': 1, 'Cinderella': 1, 'Snow White': 1})
    people.append({'Finding Nemo': 1, 'Robin Hood': 1, 'Jungle Book': 1, 'Mulan': 1, 'Little Mermaid': 1, 'Cinderella': 1, 'Snow White': 1})
    people.append({'Finding Nemo': 0, 'Robin Hood': 1, 'Jungle Book': 1, 'Mulan': 0, 'Little Mermaid': 0, 'Cinderella': 1, 'Snow White': 1})
    people.append({'Finding Nemo': 1, 'Robin Hood': 0, 'Jungle Book': 0, 'Mulan': 1, 'Little Mermaid': 1, 'Cinderella': 1, 'Snow White': 1})
    people.append({'Finding Nemo': 1, 'Robin Hood': 0, 'Jungle Book': 0, 'Mulan': 1, 'Little Mermaid': 1, 'Cinderella': 0, 'Snow White': 0})
    people.append({'Finding Nemo': 0, 'Robin Hood': 1, 'Jungle Book': 1, 'Mulan': 1, 'Little Mermaid': 0, 'Cinderella': 0, 'Snow White': 0})

    test = {'Robin Hood': 0, 'Jungle Book': 0, 'Mulan': 0, 'Little Mermaid': 0, 'Cinderella': 0, 'Snow White': 0}

    NN = find_nearest(people, test)
    # print(NN)
    # print(predict(NN, 'Finding Nemo'))

    l = []
    # with open('netflix_data/TrainingRatings.txt') as f:
    #     for i in range(10):
    #         line = f.readline()
    #         parts = line.split()
    #         print(parts)

    # print(l)

    with open('PracticeRatings.txt', 'w') as f:
        for movieId in range(1, 21):
            numRatings = random.randint(1, 3000)
            users = set()
            for i in range(numRatings):
                userId = 1
                while userId in users:
                    userId = random.randint(1, 3000)
                users.add(userId)
                rating = random.randint(1, 5)
                print(movieId, userId, rating, sep=',', file=f)

def find_nearest(examples, case):
    scores = []
    for e in examples:
        score = 0
        for movie, rating in e.items():
            if movie in case:
                if case[movie] == rating:
                    score += 1
                else:
                    score += -1
        # print(e, score)
        scores.append((e, score))
    nearest = max(scores, key = lambda x: x[1])[0]
    return nearest

def predict(nn, target):
    return nn[target]

if __name__ == '__main__':
    main()