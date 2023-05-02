import pandas as pd 

def main():
    # load data 
    data = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding="latin-1")
    # drop ratings with 0
    data = data[data['Book-Rating'] != 0]
    # print size of data
    # print(data.head())
    # print(data.shape)

    # count number of ratings for each book
    book_rating_count = data.groupby('ISBN')['Book-Rating'].count().reset_index().rename(columns={'Book-Rating': 'RatingCount'})
    # print(book_rating_count.head())   
    # print(book_rating_count.shape)
    # print(book_rating_count['RatingCount'].describe())
    # filter  books with fewer than 10 ratings

    book_rating_count = book_rating_count[book_rating_count['RatingCount'] >= 4]
    print(book_rating_count.head())

    # print how many users have rated the remaining books
    # filter data to only contain books from book_rating_count
    data = data[data['ISBN'].isin(book_rating_count['ISBN'])]
    # print(data.head())
    print(data.shape)
    # count how many unque users there are
    print(data['User-ID'].nunique())


    # print(book_rating_count.shape)

if __name__ == "__main__":
    main()    