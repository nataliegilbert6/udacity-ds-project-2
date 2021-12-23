# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
import pickle
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
        load_data()
        Purpose: Loads a database into a dataframe
        Inputs:
                - database_filepath: a path to the database that we'll load from
        Returns:
    """
    engine = create_engine('sqlite:///' + database_filepath)
    print(engine)
    df = pd.read_sql("SELECT * FROM disasterResponse", engine)
    print(df)

    X = df.message.values
    
    #remove id, genre, message, and original
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]

    return X, Y, Y.columns


def tokenize(text):
    """
        tokenize()
        Purpose: splits up a larger body of text into smaller words
        Inputs: a body of text
        Returns: tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        #reduce words to root form
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
        build_model()
        Purpose: Creates a model pipeline
        Inputs: None
        Returns: A model pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    parameters = {
    'clf__n_estimators': [50, 100, 200],
    'clf__min_samples_split': [2, 3, 4]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
        evaluate_model()
        Purpose: Prints some stats on your model's performance
        Inputs:
                - model
                - X_test
                - Y_test
                - category_names
        Output: An F1 Score, accuracy, 
    """
    Y_pred = model.predict(X_test)
    
    # run a classification report on each column
    for index, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,index]))

    print("Accuracy Score: " + str(accuracy_score(Y_test, Y_pred)))
    
def save_model(model, model_filepath):
    """
        save_model()
        Purpose: Serializes a model into a pickle file and saves it
        Inputs:
                - model: the model to save
                - model_filepath: the path where the model should be saved
        Returns: Nothing
    """
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print(model)
 
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()