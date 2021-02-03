from plotly.subplots import make_subplots
from wordcloud import WordCloud
from io import BytesIO

import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

df = pd.read_csv('data/df_tweets_clean.csv', index_col=0)
df_personal = pd.read_csv('data/df_tweets_personal.csv', index_col=0)

# Convert date columns
df['date'] = pd.to_datetime(df.date)
df_personal['date'] = pd.to_datetime(df_personal.date)

# Layout
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', 
    font_color="white",
    title_font_color="white",
    legend_title_font_color="white"
)

fig_counts = go.Figure(layout=layout)

# Count no. of tweets per day
no_tweets = df['date'].value_counts().sort_index()

# Visualization: No. of tweets per day
fig_counts.add_trace(go.Scatter(x=no_tweets.index, y=no_tweets.values, 
                                mode='lines', name='No. of tweets each day'))

# Add markers for three important dates 
dates = ['2020-11-06', '2020-11-26', '2020-12-15']
dates = list(map(pd.to_datetime, dates))
dates = no_tweets.get(dates)

hovertext = ['1st reading on 6 November: budget was released by the Ministry of Finance',
             '2nd reading on 26 November: budget was passed at the policy stage',
             '3rd reading on 15 December: budget was passed at the committee stage (the final stage)']

fig_counts.add_trace(go.Scatter(mode='markers',
                                marker_symbol='circle',
                                marker_size=10,
                                marker_color='lightblue',
                                x=dates.index, 
                                y=dates.values + 20,
                                name='Important dates',
                                hovertext=hovertext))

fig_counts.update_layout(
    title = 'Tweets count from 2020-11-06 to 2020-12-21',
    title_x=0.5, 
    yaxis_title='Tweets count',
    xaxis_showgrid=False,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)'
)

# Create a function to perform tokenizations on tweets 

def clean_text(text, unigrams=True, punctuation=True):
    '''Assume text is of type string. Convert text into lowercase.
    Remove stopwords and punctuations. Tokenize text into unigrams/bigrams. 
    
    Args:
        text (str): text to be tokenized.
        unigrams (bool): tokenize text into unigrams if True, bigrams if False.
        punctuation (bool): remove punctuation if True. 
        
    Returns:
        a list of tokens.
    '''
    from nltk.util import bigrams
    from nltk.corpus import stopwords
    import string
    import re
    
    # Covnvert text into lowercase
    text = text.lower()
    text = text.replace('&amp;', '&')
    
    # Remove url(s) which is present in another column 
    text = re.sub(' https:\/\/[^\s]+', '', text)
    
    # Remove punctuations
    if punctuation == True:
        punctuations = string.punctuation + '‘’“”...'
        text = ''.join([char for char in text if char not in punctuations])
                       
    # Remove stopwords  
    eng_stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in eng_stop_words])
    
    if unigrams == True:   
        # Tokenize texts into unigrams
        tokens = text.split()
    else:
        # Tokenize texts into bigrams
        tokens = list(bigrams(text.split()))
    
    return tokens 

# Create a line plot for daily sentiment scores 
colors = ['salmon', 'indianred', 'mediumseagreen', 'mediumpurple']
scores = ['neg', 'neu', 'pos', 'compound']
labels = ['Negative', 'Neutral', 'Positive', 'Compound']

fig_sent = go.Figure(layout=layout)
for index in range(4):
    # Calculate the mean score for each day 
    daily_sentiment = df_personal.groupby('date')['vader_' + scores[index]].mean()
    fig_sent.add_trace(go.Scatter(x=daily_sentiment.index, 
                                  y=daily_sentiment.values, 
                                  mode='lines', 
                                  name=labels[index],
                                  line_color=colors[index]))
    
fig_sent.update_layout(title='Sentiment of Malaysians towards Budget 2021 over time',
                       title_x=0.5,
                       yaxis_title='VADER Sentiment Score',
                       xaxis_showgrid=False,
                       plot_bgcolor='rgba(0,0,0,0)',
                       showlegend=True)


external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/superhero/bootstrap.min.css']
#['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

description = dcc.Markdown('''
    ## What did Malaysians think about Belanjawan 2021?
    
    In November 2020,  the government of Malaysia has released its official budget for 2021. 
    Whether Malaysians agreed with this budget is salient 
    because it represents the largest public expenditure ever in Malaysia's history (RM322.5 billion) 
    as well as the hope of economic recovery from the ongoing COVID-19 pandemic. 
    
    This project analyzed tweets concerning Belanjawan 2021 for the period between 2020-11-06 and 2020-12-21 
    in order to capture public opinion on this Budget.
    
    ''', style={'padding':'2%'})

sidenote = dcc.Markdown('''
    - Notes: 
        1. All plots except the plot of Tweets count are based on tweets in English only.
        2. The sentiment analysis of Tweets is based on VADER dictionary method.
        3. Below is a date slider, which applies to the 4 plots on the right:
    ''', style={'padding':'1%'})

# Create server variable with Flask server object for use with gunicorn
server = app.server

app.layout = html.Div(children=[
    html.Div(children=[
        
    html.Div(description, 
            style={'display':'inline-block', 'width':'60%'}),
          
    html.Div(children=[
        sidenote,
        # Add a date range picker
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df['date'].min(),
            max_date_allowed=df['date'].max(),
            start_date=df['date'].min(),
            end_date=df['date'].max(),
            style={'width':'80%', 'padding-left':'30%', 'fontSize':'small', 'height':'1vh'}),
    ], style={'display': 'inline-block', 'width':'40%'}
            )
            
    ]), 
                                             
    html.Div(children=[
        html.Div(
            # Figure 1: Tweets count
            children=[dcc.Graph(figure=fig_counts, style={'height':'100%'}),
                     dcc.Markdown('''*\*The blue markers correspond to important dates in the budget passage. Hover over for details.*''',
                                 style={'fontSize':'small', 'paddingLeft':'5%', 'verticalAlign': 'top'})],
            style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'middle', 
                  'height': '45vh'}),

        html.Div(
            children=[dcc.Graph(id='freq_dist_of_words', style={'height':'100%', 'width':'100%'})],
            style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'middle',
                  'height': '45vh'}),
        
        html.Div(
            children=[dcc.Graph(id='sent_pie_chart', style={'height': '40vh', 'width':'100%'}),
                      
                      html.Div(children=[
                          dcc.Markdown('''The mean sentiment score for this period:''', 
                                  style={'textAlign':'center', 'width':'80%', 'display': 'inline-block'}),
                          dcc.Markdown(id='mean_sentiment', 
                                       style={'textAlign':'left', 'width':'20%', 'display': 'inline-block'}),
                          dcc.Markdown('''*\*Sentiment score ranges from -1 (most negative) to +1 (most positive).*''',
                                 style={'fontSize':'small', 'textAlign':'center', 'verticalAlign': 'top'})
                      ])]
                      ,
            style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle'})
    ]),
    
    html.Div(children=[
        html.Div(children=[
        # Figure 2: Interactive frequency distribution of words   
        dcc.Graph(figure=fig_sent, style={'height':'100%'}),
                 #dcc.Markdown('''*\*For brevity, compound score represents the average sentiment,  
                 #with -1 being most negative and +1 most positive.*''',
                 #                style={'fontSize':'small', 'textAlign':'center', 'verticalAlign': 'top'})
                ], style={'display': 'inline-block', 'width':'35%', 'height': '45vh',  'verticalAlign': 'top'}),

         # Figure: Positive Word Cloud
        html.Div(children=[
            dcc.Markdown('''Most Common Words in Positive Tweets''', style={'height': '5vh', 
                                                                            'textAlign': 'center', 'paddingTop':'5%'}),
            html.Img(id='wc_positive', 
                     src='', style={'width':'100%', 'height':'35vh', 'paddingTop':'5%', 'paddingRight':'5%'}),
        ], style={'display': 'inline-block', 'width':'35%', 'height': '45vh'}), 
        # Figure: Negative Word Cloud
        html.Div(children=[
            dcc.Markdown('''Most Common Words in Negative Tweets''', style={'height': '5vh', 'textAlign': 'center',  'paddingTop':'5%'}),
            html.Img(id='wc_negative', 
                     src='', style={'width':'100%', 'height':'35vh', 'paddingTop':'5%'}),
        ], style={'display': 'inline-block', 'width':'30%', 'height': '45vh'})

    ], style={'verticalAlign':'middle'})        
])

@app.callback(
    dash.dependencies.Output('freq_dist_of_words', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
     dash.dependencies.Input('date-picker-range', 'end_date')])
    
def update_graph(start_date, end_date):    
    # Convert start_date and end_date 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Subset data between start_date and end_date 
    bool_filter = [i and j for i, j in 
                       zip(start_date <= df['date'], df['date'] <= end_date)]
    data = df['tweet'][bool_filter].apply(clean_text)
    data = pd.Series([token for tokens in data for token in tokens])
    
     # Calculate the frequency distribution of words 
    data = data.value_counts()[:20]

    # Plot the frequency distribution for top 20 words 
    fig = go.Figure([go.Bar(x=data.index, y=data.values)],
                    layout=layout)
    fig.update_layout(title='Top 20 Common Words for Tweets <br> between '+
                              str(start_date.date()) + ' and ' + str(end_date.date()),
                      title_x=0.5, 
                      xaxis=dict(tickangle=45),
                      yaxis_title='Frequency', plot_bgcolor='rgba(0,0,0,0)')
    return fig

@app.callback(
    dash.dependencies.Output('wc_positive', 'src'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
     dash.dependencies.Input('date-picker-range', 'end_date')])

def update_wc_positive(start_date, end_date):    
    # Convert start_date and end_date 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Subset data between start_date and end_date 
    bool_filter = [i and j for i, j in 
                       zip(start_date <= df_personal['date'], df_personal['date'] <= end_date)]
    data = df_personal.loc[bool_filter,]
    
    # Plot a word cloud of positive tweets
    pos_tweets = data.loc[data['vader_compound']>0, 'tweet']
    wc_tweets = ' '.join(pos_tweets.apply(clean_text).str.join(sep=' '))
    wordcloud = WordCloud(width=1500, height=900, 
                          prefer_horizontal=1,
                          background_color='#2B3E50', 
                          collocations=False, 
                          colormap='GnBu',
                          min_font_size=30).generate(wc_tweets)
#     fig, ax = plt.subplots(figsize=(20, 10)) 
#     ax.imshow(wordcloud)
#     ax.set_title('Most Common Words in Positive Tweets', fontsize=40, color='white')
#     ax.axis("off") 
#     fig.tight_layout(pad=0) 
#     out_url = fig_to_uri(fig, transparent=True)
    return wordcloud.to_image()

@app.callback(
    dash.dependencies.Output('wc_negative', 'src'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
     dash.dependencies.Input('date-picker-range', 'end_date')])

def update_wc_negative(start_date, end_date):    
    # Convert start_date and end_date 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Subset data between start_date and end_date 
    bool_filter = [i and j for i, j in 
                       zip(start_date <= df_personal['date'], df_personal['date'] <= end_date)]
    data = df_personal.loc[bool_filter,]
    
    # Plot a word cloud of negative tweets
    pos_tweets = data.loc[data['vader_compound']<0, 'tweet']
    wc_tweets = ' '.join(pos_tweets.apply(clean_text).str.join(sep=' '))
    wordcloud = WordCloud(width=1500, height=900, 
                          prefer_horizontal=1,
                          background_color='#2B3E50', 
                          collocations=False, 
                          colormap='Reds',
                          min_font_size=30).generate(wc_tweets)
#     fig, ax = plt.subplots(figsize=(20, 10)) 
#     ax.imshow(wordcloud)
#     ax.set_title('Most Common Words in Negative Tweets', fontsize=40, color='white')
#     ax.axis("off") 
#     fig.tight_layout(pad=0) 
#    out_url = fig_to_uri(wordcloud, transparent=True)

    return wordcloud.to_image()

@app.callback(
    dash.dependencies.Output('sent_pie_chart', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
     dash.dependencies.Input('date-picker-range', 'end_date')])

def update_sent_pie_chart(start_date, end_date):    
    # Convert start_date and end_date 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Subset data between start_date and end_date 
    bool_filter = [i and j for i, j in 
                       zip(start_date <= df_personal['date'], df_personal['date'] <= end_date)]
    data = df_personal.loc[bool_filter,]
    
    # Calculate no of positive and negative tweets
    pos = sum(data['vader_compound'] > 0)
    neg = sum(data['vader_compound'] < 0)
    labels = ['Positive Tweets', 'Negative Tweets']

    fig = go.Figure(go.Pie(labels=labels, values=[pos, neg], hole=.3), 
                    layout=layout)
    fig.update_layout(title='Number of Positive and Negative Tweet <br> between '+
                              str(start_date.date()) + ' and ' + str(end_date.date()),
                      title_x=0.5,
                      showlegend=True)

    return fig

@app.callback(
    dash.dependencies.Output('mean_sentiment', 'children'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
     dash.dependencies.Input('date-picker-range', 'end_date')])

def update_mean_sentiment(start_date, end_date):    
    # Convert start_date and end_date 
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Subset data between start_date and end_date 
    bool_filter = [i and j for i, j in 
                       zip(start_date <= df_personal['date'], df_personal['date'] <= end_date)]
    data = df_personal.loc[bool_filter,]
    
    # Calculate mean sentiment
    mean = round(data['vader_compound'].mean(), 2)

    return str(mean)

if __name__ == '__main__':
    app.run_server(debug=True)