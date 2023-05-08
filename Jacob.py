import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.model_selection import train_test_split

def clean_data():
    print("Reading Books Data...")
    books_data = pd.read_csv("Data/BX-Books.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Readting Users Data...")
    users_data = pd.read_csv("Data/BX-Users.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Reading Ratings Data...")
    book_ratings = pd.read_csv("Data/BX-Book-Ratings.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Done!")
    # In Book-Author, there are some values that are not strings. Drop their rows from the dataset
    books_data = books_data.loc[books_data['Book-Author'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Year-Of-Publication'].apply(lambda x: isinstance(x, int)), :]
    books_data = books_data.loc[books_data['Year-Of-Publication'].apply(lambda x: x != 0), :]
    books_data = books_data.loc[books_data['Publisher'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-S'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-M'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-L'].apply(lambda x: isinstance(x, str)), :]
    books_data.dropna(inplace=True)
    books_data.reset_index(drop=True, inplace=True)

    # Grab all the ISBNs that been rated over 20 times in books_data
    books_data = books_data.loc[books_data['ISBN'].isin(book_ratings['ISBN'].value_counts()[book_ratings['ISBN'].value_counts() >= 20].index), :]
    print("Books that have been rated more than 20 times: ", len(books_data))

    # filter out ratings that are not in the books_data
    book_ratings = book_ratings.loc[book_ratings['ISBN'].isin(books_data['ISBN']), :]
    print("Total number of ratings in the dataset is: ", len(book_ratings))

    # filter out users who have not rated more than 4 books from book_ratings
    book_ratings = book_ratings.loc[book_ratings['User-ID'].isin(book_ratings['User-ID'].value_counts()[book_ratings['User-ID'].value_counts() > 4].index), :]
    print("Number of ratings for books rated by users who have rated >=5 books: ", len(book_ratings))


    # keep users that are in book_ratings
    users_data = users_data.loc[users_data['User-ID'].isin(book_ratings['User-ID']), :]
    print("Total number of users in the dataset is: ", len(users_data))

    return users_data, books_data, book_ratings

def plot_frequency(book_ratings):
    book_ratings_counts = book_ratings['ISBN'].value_counts()
    book_ratings_counts.sort_values(ascending=False, inplace=True)

    x_values = np.arange(len(book_ratings_counts)) + 1
    y_values = book_ratings_counts.values
    # calculate the slope and y-intercept of the regression line
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x_values), np.log(y_values))
    # plot the data points and regression line
    plt.plot(x_values, y_values, 'o')
    plt.plot(x_values, np.exp(intercept + slope*np.log(x_values)))
    plt.title('Number of Ratings per Book (Log-Log Scale)')
    plt.xlabel('Books')
    plt.ylabel('Number of Ratings')
    plt.xscale('log')
    plt.yscale('log')

    # add the R^2 value to the plot
    plt.text(0.1, 0.9, 'R^2 = ' + str(round(r_value**2, 3)), transform=plt.gca().transAxes)
    plt.show()


def create_matrix(book_ratings):
    # Create a pivot table where the rows are the users and the columns are the ISBNs
    print("Creating a pivot table where the rows are the users and the columns are the ISBNs...")
    matrix = pd.pivot_table(book_ratings, values='Book-Rating', index='User-ID', columns='ISBN', fill_value=0)
    return matrix


def cosine_similarity_(matrix):
    cosine_sim = cosine_similarity(matrix, matrix)
    # Turn the cosine_sim into a dataframe
    cosine_sim = pd.DataFrame(cosine_sim, index=matrix.index, columns=matrix.index)
    # Fill the diagonal with 0s instead of 1s
    np.fill_diagonal(cosine_sim.values, 0)
    return cosine_sim
    
def save_clean_data(user_data, book_data, book_ratings):
    # save the cleaned data to csv files
    user_data.to_pickle("data/user_data.pkl.gz")
    book_data.to_pickle("data/book_data.pkl.gz")
    book_ratings.to_pickle("data/book_ratings.pkl.gz")

def load_clean_data():
    # load the cleaned data
    user_data = pd.read_pickle("data/user_data.pkl.gz")
    book_data = pd.read_pickle("data/book_data.pkl.gz")
    book_ratings = pd.read_pickle("data/book_ratings.pkl.gz")
    return user_data, book_data, book_ratings


def get_mean_abs_error(predictions, test_data_matrix):
    errors = []
    for user_id, book_ratings in predictions.items():
        if user_id not in test_data_matrix.index:
            continue
        for book in book_ratings:
            if book not in test_data_matrix.columns:
                continue
            # get the actual ratings for the user
            actual_rating = test_data_matrix.loc[user_id, book]
            # calculate the absolute difference between the predicted and actual ratings
            absolute_error = np.abs(book_ratings.loc[book] - actual_rating)
            errors.append(absolute_error)
    return np.mean(errors)

def get_recommendations(user_id, user_rating_matrix, cosine_sim):
    if user_id not in user_rating_matrix.index or user_id not in cosine_sim.index:
        print(user_id, " is not in the rating matrix or is not in the cosine similarity matrix")
        return None
    similar_users = cosine_sim[user_id].sort_values(ascending=False)
    similar_users = similar_users[similar_users > 0]
    # get the user ids of the similar users
    similar_users = similar_users.index
    book_data = {}
    for user in similar_users:
        similarity = cosine_sim.loc[user_id, user]
        if user not in user_rating_matrix.index:
            continue
        ratings = user_rating_matrix.loc[user, :]
        books = ratings[ratings > 0].index
        ratings = ratings[ratings > 0].values
        for book, rating in zip(books, ratings):
            if book not in book_data:
                book_data[book] = [[similarity, rating]]
            else:
                book_data[book].append([similarity, rating])
    book_scores = {}
    for k, v in book_data.items():
        total_sim = np.sum([sim for sim, rating in v])
        weighted_rating = np.sum([sim * rating for sim, rating in v]) / total_sim
        average_rating = np.mean([rating for sim, rating in v])
        book_scores[k] = [average_rating, weighted_rating]

    book_recommendations = pd.DataFrame.from_dict(book_scores, orient='index', columns=['average_rating', 'weighted_rating'])
    book_recommendations.sort_values(by='weighted_rating', ascending=False, inplace=True)
    return book_recommendations


def predict_ratings(user_id, book_ids, matrix, cosine_sim):
    recommendations = get_recommendations(user_id, matrix, cosine_sim)
    if recommendations is None:
        print("no recommendations found for user_id: ", user_id)
        return None
    # get the ratings for the books in book_ids

    book_ratings = []
    for book_id in book_ids:
        if book_id not in recommendations.index:
            pass
        else:
            book_ratings.append((book_id, recommendations.loc[book_id, 'weighted_rating']))
    return book_ratings


if __name__ == "__main__":
    new_data = False
    if new_data:
        user_data, book_data, book_ratings = clean_data()
        save_clean_data(user_data, book_data, book_ratings)

        user_rating_matrix = create_matrix(book_ratings)
        user_rating_matrix.to_pickle("data/user_rating_matrix.pkl.gz")
        print("user_rating_matrix created")

        cosine_sim = cosine_similarity_(user_rating_matrix)
        cosine_sim.to_pickle("data/cosine_sim_matrix.pkl.gz")
        print("cosine_sim_matrix created")
    else:
        user_data, book_data, book_ratings = load_clean_data()
        user_rating_matrix = pd.read_pickle("data/user_rating_matrix.pkl.gz")
        cosine_sim = pd.read_pickle("data/cosine_sim_matrix.pkl.gz")
        print("Loading data fininshed")

    train_data, test_data = train_test_split(book_ratings, test_size=0.5, random_state=42)
    print("Training data size: ", train_data.shape)
    print("Test data size: ", test_data.shape)

    n_users = train_data['User-ID'].nunique()
    n_books = train_data['ISBN'].nunique()
    print("Number of unique users: ", n_users)
    print("Number of unique books: ", n_books)

    # count number of unique users in the test data
    n_users_test = test_data['User-ID'].nunique()
    # count number of unique books in the test data
    n_books_test = test_data['ISBN'].nunique()
    print("Number of unique users in test data: ", n_users_test)
    print("Number of unique books in test data: ", n_books_test)
    

    # create the user-item matrix for the training data
    train_data_matrix = create_matrix(train_data)
    print(train_data_matrix.shape)
    train_sim = cosine_similarity_(train_data_matrix)
    test_data_matrix = create_matrix(test_data)
    print(test_data_matrix.shape)

    # we will try to predict the user ratings for the test data using the training data
    # create a dictionary where the keys are the user ids and the values are the book ids
    user_item_dict = {}
    for user_id in test_data_matrix.index:
        user_books = test_data_matrix.loc[user_id, :]
        user_item_dict[user_id] = user_books.index.tolist()

    # get the predictions for the test data
    predictions = {}
    for user_id, book_ids in user_item_dict.items():
        book_rating_predictions = predict_ratings(user_id, book_ids, test_data_matrix, train_sim)
        if book_rating_predictions is None:
            continue
        for book_id, rating in book_rating_predictions:
            if user_id not in predictions:
                predictions[user_id] = [(book_id, rating)]
            else:
                predictions[user_id].append((book_id, rating))

    
    total_error = 0
    rows_processed = 0
    fake_error = 0
    fake_error1 = 0
    fake_error2 = 0
    fake_error3 = 0
    fake_error4 = 0
    fake_error5 = 0
    fake_error6 = 0
    fake_error7 = 0
    
    for user_id, book_ratings in predictions.items():
        # get the actual ratings for the user
        actual_ratings = test_data_matrix.loc[user_id]
        # get the books that the user has rated
        common_books = actual_ratings[actual_ratings > 0].index 
        for book in common_books:
            actual_rating = actual_ratings[book]
            if actual_rating == 0:
                continue 
            predicted_rating = [rating for book_id, rating in book_ratings if book_id == book]
            if len(predicted_rating) == 0:
                continue
            predicted_rating = predicted_rating[0]
            print("Actual rating: ", actual_rating, "Predicted rating: ", predicted_rating)
            error  = np.abs(actual_rating - predicted_rating)
            total_error += error
            fake_error +=  np.abs(actual_rating - 3)
            fake_error1 += np.abs(actual_rating - 4)
            fake_error2 += np.abs(actual_rating - 5)
            fake_error3 += np.abs(actual_rating - 6)
            fake_error4 += np.abs(actual_rating - 7)
            fake_error5 += np.abs(actual_rating - 8)
            fake_error6 += np.abs(actual_rating - 9)
            fake_error7 += np.abs(actual_rating - 10)
            rows_processed += 1

    print("Rows processed: ", rows_processed)
    print("Mean absolute error: ", total_error / rows_processed)
    print("Fake error: ", fake_error/ rows_processed)
    print("Fake error1: ", fake_error1/ rows_processed)
    print("Fake error2: ", fake_error2/ rows_processed)
    print("Fake error3: ", fake_error3/ rows_processed)
    print("Fake error4: ", fake_error4/ rows_processed)
    print("Fake error5: ", fake_error5/ rows_processed)
    print("Fake error6: ", fake_error6/ rows_processed)
    print("Fake error7: ", fake_error7/ rows_processed)
