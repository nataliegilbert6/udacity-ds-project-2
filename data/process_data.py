# import libraries
import sys
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        load_data()
        Purpose: Loads messages and categories data into a single dataframe
        Inputs:
                - messages_filepath: a filepath to a csv with some messages in them
                - categories_filepath: a filepath to a csv with some categories in them
        Returns: A merged dataframe 
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(categories, messages, on='id')
    
    return df


def clean_data(df):
    """
        clean_data()
        Purpose: Cleans a dataframe
        Inputs:
                - df: an uncleaned dataframe
        Returns: A cleaned pandas dataframe
    """
    
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # related should be boolean, but it's not
    categories['related'].replace(2,1, inplace=True)
    
    cols_to_drop=['categories']
    df = df.drop(*cols_to_drop, axis=1)

    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
        save_data()
        Purpose: Saves dataframe to a given database
        Inputs:
                - df: a dataframe that needs to be saved
                - database_filename: the name of the database file
        Returns: Nothing
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disasterResponse', engine, if_exists='replace', index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()