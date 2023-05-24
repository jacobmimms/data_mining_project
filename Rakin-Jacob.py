import json
import time
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from sklearn.model_selection import train_test_split

def get_clean_rating_data():
    print("Reading Books Data...")
    books_data = pd.read_csv("Data/BX-Books.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Readting Users Data...")
    users_data = pd.read_csv("Data/BX-Users.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Reading Ratings Data...")
    book_ratings = pd.read_csv("Data/BX-Book-Ratings.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Done!")
    # In Book-Author, there are some values that are not strings. Drop their rows from the dataset
    books_data = pd.read_csv("Data/BX-Books.csv", sep=';', on_bad_lines='skip', encoding="latin")
    books_data = books_data.loc[books_data['Book-Author'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Year-Of-Publication'].apply(lambda x: isinstance(x, int)), :]
    books_data = books_data.loc[books_data['Year-Of-Publication'].apply(lambda x: x != 0), :]
    books_data = books_data.loc[books_data['Publisher'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-S'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-M'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-L'].apply(lambda x: isinstance(x, str)), :]
    books_data.dropna(inplace=True)
    books_data.reset_index(drop=True, inplace=True)
    # return book ratings wehre the ISBN is in the books_data
    return book_ratings.loc[book_ratings['ISBN'].isin(books_data['ISBN']), :]
    


def clean_data():
    print("Reading Books Data...")
    books_data = pd.read_csv("Data/BX-Books.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Readting Users Data...")
    users_data = pd.read_csv("Data/BX-Users.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Reading Ratings Data...")
    book_ratings = pd.read_csv("Data/BX-Book-Ratings.csv", sep=';', on_bad_lines='skip', encoding="latin")
    print("Done!")
    # In Book-Author, there are some values that are not strings. Drop their rows from the dataset
    books_data = pd.read_csv("Data/BX-Books.csv", sep=';', on_bad_lines='skip', encoding="latin")
    books_data = books_data.loc[books_data['Book-Author'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Year-Of-Publication'].apply(lambda x: isinstance(x, int)), :]
    books_data = books_data.loc[books_data['Year-Of-Publication'].apply(lambda x: x != 0), :]
    books_data = books_data.loc[books_data['Publisher'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-S'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-M'].apply(lambda x: isinstance(x, str)), :]
    books_data = books_data.loc[books_data['Image-URL-L'].apply(lambda x: isinstance(x, str)), :]
    books_data.dropna(inplace=True)
    books_data.reset_index(drop=True, inplace=True)

    ############## old code ##############
    #     # filter out zero ratings from book_ratings
    # book_ratings = book_ratings.loc[book_ratings['Book-Rating'] != 0, :]
    # print("Total number of ratings in the dataset is after removing zeros: ", len(book_ratings))

    # books_data = books_data.loc[books_data['ISBN'].isin(book_ratings['ISBN'].value_counts()[book_ratings['ISBN'].value_counts() >= 5].index), :]
    # print("Books that have been rated at least 5 times: ", len(books_data))

    # # books_data = books_data.loc[books_data['ISBN'].isin(book_ratings['ISBN'].value_counts()[book_ratings['ISBN'].value_counts() >= 10].index), :]
    # # print("Books that have been rated at least 10 times: ", len(books_data))

    # # books_data = books_data.loc[books_data['ISBN'].isin(book_ratings['ISBN'].value_counts()[book_ratings['ISBN'].value_counts() >= 20].index), :]
    # # print("Books that have been rated at least 20 times: ", len(books_data))

    # # filter out ratings that are not in the books_data
    # book_ratings = book_ratings.loc[book_ratings['ISBN'].isin(books_data['ISBN']), :]
    # print("Total number of ratings in the dataset is: ", len(book_ratings))

    # # filter out users who have not rated more than 4 books from book_ratings
    # book_ratings = book_ratings.loc[book_ratings['User-ID'].isin(book_ratings['User-ID'].value_counts()[book_ratings['User-ID'].value_counts() > 4].index), :]
    # print("Number of ratings for books rated by users who have rated >=5 books: ", len(book_ratings))

    # # keep users that are in book_ratings
    # users_data = users_data.loc[users_data['User-ID'].isin(book_ratings['User-ID']), :]
    # print("Total number of users in the dataset is: ", len(users_data))


    # return users_data, books_data, book_ratings
    ############## end old code ##############



    # filter out zero ratings from book_ratings
    book_ratings = book_ratings.loc[book_ratings['Book-Rating'] != 0, :]

    # filter out users who have not rated more than 4 books from book_ratings
    users_data = users_data.loc[users_data['User-ID'].isin(book_ratings['User-ID'].value_counts()[book_ratings['User-ID'].value_counts() > 4].index), :]

    # remove book ratings by users who are not in users_data
    book_ratings = book_ratings.loc[book_ratings['User-ID'].isin(users_data['User-ID']), :]

    # filter out books that have not been rated at least 5 times
    book_ratings = book_ratings.loc[book_ratings['ISBN'].isin(book_ratings['ISBN'].value_counts()[book_ratings['ISBN'].value_counts() >= 5].index), :]

    # filter books_data to only include books in book_ratings
    books_data = books_data.loc[books_data['ISBN'].isin(book_ratings['ISBN']), :]
    book_ratings = book_ratings.loc[book_ratings['ISBN'].isin(books_data['ISBN']), :]

    # remove users not in book_ratings
    users_data = users_data.loc[users_data['User-ID'].isin(book_ratings['User-ID']), :]

    # remove book data that are not in book_ratings
    books_data = books_data.loc[books_data['ISBN'].isin(book_ratings['ISBN']), :]
    return users_data, books_data, book_ratings

def plot_frequency(book_ratings, title= 'Number of Ratings per Book'):
    book_ratings_counts = book_ratings['ISBN'].value_counts()
    book_ratings_counts.sort_values(ascending=False, inplace=True)

    x_values = np.arange(len(book_ratings_counts)) + 1
    y_values = book_ratings_counts.values

    print(book_ratings_counts.tail(10))
    print(min(y_values))
    # calculate the slope and y-intercept of the regression line
    slope, intercept, r_value, p_value, std_err = linregress(np.log(x_values), np.log(y_values))
    # plot the data points and regression line
    plt.plot(x_values, y_values, 'o')
    plt.plot(x_values, np.exp(intercept + slope*np.log(x_values)))
    plt.title(title)
    plt.xlabel('Books')
    plt.ylabel('Number of Ratings')
    plt.xscale('log')
    plt.yscale('log')
    
    # make figure taller
    plt.gcf().set_size_inches(6, 6)

    # add the R^2 value to the plot
    plt.text(0.1, 0.9, 'R^2 = ' + str(round(r_value**2, 3)), transform=plt.gca().transAxes)
    plt.show()


def create_matrix(book_ratings):
    # Create a pivot table where the rows are the users and the columns are the ISBNs
    print("Creating a user-rating dataframe where the rows are the users and the columns are the ISBNs...")
    matrix = pd.pivot_table(book_ratings, values='Book-Rating', index='User-ID', columns='ISBN', fill_value=0)
    return matrix

def basic_cosine_similarity(matrix):
    cosine_sim = cosine_similarity(matrix, matrix)
    # Turn the cosine_sim into a dataframes
    cosine_sim = pd.DataFrame(cosine_sim, index=matrix.index, columns=matrix.index)
    # Fill the diagonal with 0s instead of 1s
    np.fill_diagonal(cosine_sim.values, 0)
    return cosine_sim

def adjusted_cosine_similarity(rating_matrix):
    results = np.zeros((rating_matrix.shape[0], rating_matrix.shape[0]))
    # create binary matrix where 1 indicates that the user has rated the book
    overlap = cosine_similarity(rating_matrix)
    np.fill_diagonal(overlap, 0)
    overlap[overlap > 0] = 1

    # iterate over the indices of overlap where the value is 1
    count = 0
    start_time = time.time()
    for i, j in zip(*np.where(overlap == 1)):
        user_i = rating_matrix.iloc[i].to_numpy()
        user_j = rating_matrix.iloc[j].to_numpy()
        # get the indices where both users have rated the same book
        common_indices = np.intersect1d(user_i.nonzero(), user_j.nonzero())
        # if there are no common indices, then the similarity is 0
        if len(common_indices) == 0:
            results[i][j] = 0
            continue
        # get the ratings for the common indices
        user_i_ratings = user_i[common_indices]
        user_j_ratings = user_j[common_indices]
        # calculate the adjusted cosine similarity
        results[i][j] = cosine_similarity(user_i_ratings.reshape(1, -1), user_j_ratings.reshape(1, -1))[0][0]
        count += 1
        if count % 10000 == 0:
            print(f'count {count} took: {time.time() - start_time}')
            start_time = time.time()
        
    return results
    
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


def get_recommendations(id, user_rating_matrix, cosine_sim, mean_normalization=False):
    # if id not in user_rating_matrix.index or id not in cosine_sim.index:
    #     print(f"{id} is not in the rating matrix or is not in the cosine similarity matrix")
    #     return None
    
    similar_users = cosine_sim.loc[id].sort_values(ascending=False)
    similar_users = similar_users[similar_users > 0].index
    
    book_data = {}
    
    for user in similar_users:
        # if user not in user_rating_matrix.index:
        #     print(f"{user} is not in the rating matrix")
        #     continue
        similarity = cosine_sim.loc[id, user]
        ratings = user_rating_matrix.loc[user, :]
        if mean_normalization:
            mean_rating = ratings.mean() 
            ratings = (ratings - mean_rating).replace(0, np.nan) 
            ratings = ratings.dropna() # remove the NaNs
        ratings = ratings[ratings > 0]
        for book, rating in ratings.items():
            if book not in book_data:
                book_data[book] = []
            book_data[book].append((similarity, rating))
    
    book_scores = {}
    for k, v in book_data.items():
        total_sim = np.sum([sim for sim, rating in v])
        weighted_rating = np.sum([sim * rating for sim, rating in v]) / total_sim

        average_rating = np.mean([rating for sim, rating in v])
        book_scores[k] = {'average_rating': average_rating, 'weighted_rating': weighted_rating}
    
    book_recommendations = pd.DataFrame.from_dict(book_scores, orient='index')
    book_recommendations.sort_values(by='weighted_rating', ascending=False, inplace=True)
    
    return book_recommendations[['average_rating', 'weighted_rating']]


def predict_ratings(user_id, book_ids, matrix, cosine_sim):
    recommendations = get_recommendations(user_id, matrix, cosine_sim, mean_normalization=True)
    # if recommendations is None:
    #     print("no recommendations found for user_id: ", user_id)
    #     return None
    # get the ratings for the books in book_ids

    book_ratings = []
    for book_id in book_ids:
        if book_id not in recommendations.index:
            print(f"{book_id} is not in the recommendations dataframe")
            pass
        else:
            book_ratings.append((book_id, recommendations.loc[book_id, 'weighted_rating']))
    return book_ratings

def predict_all_ratings(user_ratings, cosine_sim):
    """
    This function creates a matrix (user_ids x book_ids) of predicted ratings 
    """ 
    weighted_ratings = np.matmul(cosine_sim, user_ratings)
    cosine_sim_sum = cosine_sim.sum(axis=1)
    print(cosine_sim_sum)
    # divide every row by the sum of the cosine similarities
    weighted_ratings = weighted_ratings / cosine_sim_sum[:, None]
    print(weighted_ratings)
    # weighted_ratings = pd.DataFrame(weighted_ratings, index=user_ratings.index, columns=user_ratings.columns)
    return weighted_ratings

def test(book_ratings):
    # group the data by user ID
    grouped_data = book_ratings.groupby('User-ID')
    # create two dataframes for each user
    train_data = pd.DataFrame(columns=book_ratings.columns)
    test_data = pd.DataFrame(columns=book_ratings.columns)

    for user, group in grouped_data:
        # split the group into training and test data
        user_train_data, user_test_data = train_test_split(group, test_size=0.5, random_state=42)
        train_data = pd.concat([train_data, user_train_data])
        test_data = pd.concat([test_data, user_test_data])
    # check that every user in the test data is in the training data
    missing_users = test_data[~test_data['User-ID'].isin(train_data['User-ID'].unique())]['User-ID'].unique()
    if len(missing_users) > 0:
        print("The following users are in the test data but not in the training data:")
        print(missing_users)
    else:
        print("All users in the test data are also in the training data.")


    # create the user-item matrix for the training data
    train_data_matrix = create_matrix(train_data)
    test_data_matrix = create_matrix(test_data)

    train_sim = cosine_similarity_(train_data_matrix)
    test_sim = cosine_similarity_(test_data_matrix)

    # we will try to predict the user ratings for the test data using the training data
    # create a dictionary where the keys are the user ids and the values are the book ids
    user_item_dict = {}
    for user_id in test_data_matrix.index:
        user_books = test_data_matrix.loc[user_id, :]
        user_item_dict[user_id] = user_books.index.tolist()

    # get the predictions for the test data
    predictions = {}
    for user_id, book_ids in user_item_dict.items():
        # if user_id > 10000:
        #     break
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
    
    actual_rating_mean = test_data_matrix[test_data_matrix > 0].mean().mean()
    print(f"Mean of actual ratings: {actual_rating_mean}")

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
            error  = np.abs(actual_rating - predicted_rating)
            total_error += error
            fake_error +=  np.abs(actual_rating - actual_rating_mean)
            rows_processed += 1

    print("Rows processed: ", rows_processed)
    print("Mean absolute error: ", total_error / rows_processed)
    print("Fake error: ", fake_error/ rows_processed)

def check_book_frequency():
    ratings = get_clean_rating_data()
    ratings = ratings.loc[ratings['User-ID'].isin(ratings['User-ID'].value_counts()[ratings['User-ID'].value_counts() > 4].index), :]
    print(len(ratings))
    print(f"Number of unique books: {len(ratings['ISBN'].unique())}")
    print(f"Number of unique users: {len(ratings['User-ID'].unique())}")
    plot_frequency(ratings, f'Number of Ratings per Book \n All Ratings \n Total Ratings: {len(ratings)}')
    ratings = ratings[ratings['Book-Rating'] != 0]
    print(len(ratings))
    print(f"Number of unique books: {len(ratings['ISBN'].unique())}")
    print(f"Number of unique users: {len(ratings['User-ID'].unique())}")

    plot_frequency(ratings, f'Number of Ratings per Book \n (Ratings > 0) \n Total Ratings: {len(ratings)}')
    # get books that are rated at least 5 times
    
    ratings = ratings.groupby('ISBN').filter(lambda x: len(x) >= 5)
    print(len(ratings))
    print(f"Number of unique books: {len(ratings['ISBN'].unique())}")
    print(f"Number of unique users: {len(ratings['User-ID'].unique())}")
    plot_frequency(ratings, f'Number of Ratings per Book \n (#ratings > 5) \n Total Ratings: {len(ratings)}')

if __name__ == "__main__":
    new_data = True
    adjusted_cosine = False
   
    if new_data:
        user_data, book_data, book_ratings = clean_data()
        save_clean_data(user_data, book_data, book_ratings)

        user_rating_matrix = create_matrix(book_ratings)
        user_rating_matrix.to_pickle("data/user_rating_matrix.pkl.gz")
        print("user_rating_matrix created")
        if adjusted_cosine:
            cosine_sim = adjusted_cosine_similarity(user_rating_matrix) 
            cosine_sim.to_pickle("data/adjusted_cosine_sim_matrix.pkl.gz")

        else:
            cosine_sim = basic_cosine_similarity(user_rating_matrix)
            cosine_sim.to_pickle("data/basic_cosine_sim_matrix.pkl.gz")
        print("cosine_sim_matrix created")

    else:
        user_data, book_data, book_ratings = load_clean_data()
        user_rating_matrix = pd.read_pickle("data/user_rating_matrix.pkl.gz")
        if adjusted_cosine:
            cosine_sim = np.load("data/adjusted_cosine_sim_matrix.npy")
        else:
            cosine_sim = pd.read_pickle("data/basic_cosine_sim_matrix.pkl.gz")
        print("Loading data fininshed")

    predicted_ratings = predict_all_ratings(user_rating_matrix, cosine_sim.to_numpy())
    print("What would you like to call the file that stored your predicted ratings?")
    file_name = input()
    
    np.save("data/basic_predicted_ratings.npy", predicted_ratings)

    predicted_ratings = np.load("data/basic_predicted_ratings.npy")
    print(predicted_ratings[0:10])
    print("predicted_ratings shape: ", predicted_ratings.shape)
    # set all ratings that are not in the original matrix to 0
    predicted_ratings[user_rating_matrix == 0] = 0
    print(predicted_ratings[0:10])

    print("predicted_actual_ratings shape: ", predicted_ratings.shape)


    user_rating_matrix[predicted_ratings == 0] = 0
    print(user_rating_matrix[0:10])
    print("user_rating_matrix shape: ", user_rating_matrix.shape)

    print("Number of ratings that are not 0: ", np.count_nonzero(predicted_ratings))
    print("Number of actual ratings that are not 0: ", np.count_nonzero(user_rating_matrix))

    # mean absolute error
    print("Mean absolute error: ", np.sum(np.abs(predicted_ratings - user_rating_matrix)) / np.count_nonzero(user_rating_matrix))


