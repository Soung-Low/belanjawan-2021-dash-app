from plotly.subplots import make_subplots
from wordcloud import WordCloud

import dash
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import string
import re

# Import data 
df = pd.read_csv('data/df_tweets_clean.csv', index_col=0)
df_personal = pd.read_csv('data/df_tweets_personal.csv', index_col=0)

# Convert date columns
df['date'] = pd.to_datetime(df.date)
df_personal['date'] = pd.to_datetime(df_personal.date)

# Configure plot layout
layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', 
    font_color="white",
    title_font_color="white",
    legend_title_font_color="white"
)

# Plot 1: No. of tweets per day
fig_counts = go.Figure(layout=layout)
no_tweets = df['date'].value_counts().sort_index() # Count no. of tweets per day
fig_counts.add_trace(go.Scatter(x=no_tweets.index, y=no_tweets.values, 
                                mode='lines', name='No. of tweets each day'))
# Add markers for three important dates 
dates = ['2020-11-06', '2020-11-26', '2020-12-15'] 
dates = list(map(pd.to_datetime, dates)) 
dates = no_tweets.get(dates)
# Add hovertext 
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
    plot_bgcolor='rgba(0,0,0,0)')

# Plot 2: Daily sentiment scores 
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


# Import CSS template 
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/superhero/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define descriptions for plots 
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
    
    # Section 1
    html.Div(children=[

        # Add introduction of the project    
        html.Div(description, style={'display':'inline-block', 'width':'60%'}),
            
        html.Div(children=[
            # Add side note
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

    # Section 2                                 
    html.Div(children=[
        html.Div(children=[
            # Figure 1: Tweets count
            dcc.Graph(figure=fig_counts, style={'height':'100%'}),
            # Add notes for Figure 1 
            dcc.Markdown('''*\*The blue markers correspond to important dates in the budget passage.  
            Hover over for details.*''',
                style={'fontSize':'small', 'verticalAlign': 'top', 'textAlign':'center'})
            ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'middle', 'height': '45vh'}
        ),

        html.Div(
            # Figure 2: Top 20 Common Words 
            children=[dcc.Graph(id='freq_dist_of_words', style={'height':'100%', 'width':'100%'})],
            style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'middle', 'height': '45vh'}
        ),
        
        html.Div(children=[
            # Figure 3: Number of Positive and Negative Tweets 
            dcc.Graph(id='sent_pie_chart', style={'height': '40vh', 'width':'100%'}),
            # Add the mean sentiment for a given period 
            html.Div(children=[
                dcc.Markdown('''The mean sentiment score for this period:''',
                    style={'textAlign':'center', 'width':'80%', 'display': 'inline-block'}),
                dcc.Markdown(id='mean_sentiment', 
                    style={'textAlign':'left', 'width':'20%', 'display': 'inline-block'}),
                dcc.Markdown('''*\*Sentiment score ranges from -1 (most negative) to +1 (most positive).*''',
                    style={'fontSize':'small', 'textAlign':'center', 'verticalAlign': 'top'})
                ])
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'middle'}
        )
    ]),
    
    # Section 3 
    html.Div(children=[
        html.Div(children=[
            # Figure 4: Sentiment score   
            dcc.Graph(figure=fig_sent, style={'height':'100%'}),
            # Add note for Figure 4
            dcc.Markdown('''*\*For brevity, compound score represents the average sentiment,  
            with -1 being most negative and +1 most positive.*''',
                style={'fontSize':'small', 'textAlign':'center', 'verticalAlign': 'top'})
            ], style={'display': 'inline-block', 'width':'35%', 'height': '45vh',  'verticalAlign': 'top'}
        ),

        html.Div(children=[
            # Figure 5 : Positive Word Cloud
            dcc.Markdown('''Most Common Words in Positive Tweets''', 
                style={'height': '5vh', 'textAlign': 'center', 'paddingTop':'5%'}),
            html.Img(id='wc_positive', src='', 
                style={'width': '90%', 'height': '40vh', 'textAlign': 'center', 'paddingTop':'5%'}),
            ], style={'display': 'inline-block', 'width':'35%', 'height': '45vh'}
        ), 
        
        html.Div(children=[
            # Figure 6: Negative Word Cloud
            dcc.Markdown('''Most Common Words in Negative Tweets''', 
                style={'height': '5vh', 'textAlign': 'center',  'paddingTop':'5%'}),
            html.Img(id='wc_negative', src='', 
                style={'width': '90%', 'height': '40vh', 'textAlign': 'center', 'paddingTop':'5%'}),
            ], style={'display': 'inline-block', 'width':'30%', 'height': '45vh'}
        )], style={'verticalAlign':'middle'}
    )        
])

# Figure 2: Top 20 Common Words 
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
    data = df['tweet_clean'][bool_filter]
    data = pd.Series([token for tokens in data for token in tokens.split(' ')])
    
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

# Figure 3: Number of Positive and Negative Tweets 
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
    data = df_personal.loc[bool_filter,:]
    
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

# Figure 5: Positive Word Cloud
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
    data = df_personal.loc[bool_filter,:]
    
    # Plot a word cloud of positive tweets
    pos_tweets = data.loc[data['vader_compound']>0, 'tweet_clean']
    wc_tweets = ' '.join(pos_tweets)
    wordcloud = WordCloud(width=1500, height=900, 
                          prefer_horizontal=1,
                          background_color='#2B3E50', 
                          collocations=False, 
                          colormap='GnBu',
                          min_font_size=30).generate(wc_tweets)
    return wordcloud.to_image()

# Figure 6: Negative Word Cloud
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
    neg_tweets = data.loc[data['vader_compound']<0, 'tweet_clean']
    wc_tweets = ' '.join(neg_tweets)
    wordcloud = WordCloud(width=1500, height=900, 
                          prefer_horizontal=1,
                          background_color='#2B3E50', 
                          collocations=False, 
                          colormap='Reds',
                          min_font_size=30).generate(wc_tweets)
    return wordcloud.to_image()

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)