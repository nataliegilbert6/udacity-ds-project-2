import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
        tokenize()
        Purpose: splits up a larger body of text into smaller words
        Inputs: a body of text
        Returns: tokenized text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disasterResponse.db')
df = pd.read_sql_table('disasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    """
        index()
        Purpose: index webpage displays cool visuals and receives user input text for model
    """
    # extract data needed for visuals
    
    #genre data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # category data
    category_counts = (df.iloc[:,4:] == 1).sum().values
    category_names = df.iloc[:,4:].columns
     
    # word cloud data
    message_list = df['message'].unique().tolist()
    len_message_list = [len(tokenize(message)) for message in message_list]
    
    # create visuals
    graphs = [
        {
            #genre
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            # category
            'data': [
                Bar(
                    x=message_list,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 45
                }
            }
        },
        {
            #messages
            'data': [
                Histogram(
                    x = len_message_list
                )
            ],
            'layout': {
                'title': 'Distribution of Word Counts Per Message',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of words",
                    'range': [0,100]
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()