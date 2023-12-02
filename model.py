# Standard library imports
import os
import time
import json
import csv
import string
import importlib.util
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import emoji
import pickle
import praw
import requests
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from langdetect import detect, DetectorFactory, LangDetectException
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, timedelta
from transformers import pipeline
from requests.exceptions import HTTPError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# Download necessary NLTK data
import nltk
import ssl
import re
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load environment variables
_ = load_dotenv(find_dotenv())
stop_words = set(stopwords.words('english'))

#Importsand installs
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def install_package(package_name):
    # Check if the package is installed
    if importlib.util.find_spec(package_name) is None:
        print(f"{package_name} is not installed. Installing...")
        # Install the package
        subprocess.check_call(["pip", "install", package_name])
    else:
        print(f"{package_name} is already installed.")
def call_install():
    install_package('nltk')
    install_package('requests')
    install_package('langdetect')
    install_package('tqdm')
    install_package('bs4')
    install_package('python-dotenv')
    install_package('pandas')
    install_package('matplotlib')
    install_package('emoji')
    install_package('pickle')
    install_package('praw')

#function to perform google search and save results to a file
def perform_google_search(query, filename,nos):
    #for security purpose never write your api_key in the code it self , either save it in some other file or in envvar
    api_key = os.getenv('SERP_API_KEY')#required to access api
    params = {
        "q": query,#your search
        "location": "India",#select a location to get best results , here India
        "hl": "en",
        "gl": "in",
        "google_domain": "google.co.in",
        "num": nos,#number of links i want to extract text from
        "api_key": api_key,#my api key
    }

    search = GoogleSearch(params)
    #print(search)
    results = search.get_dict()#return results in dictonary format
    #print(results)

    if 'organic_results' in results: #search for key organic results , if exists save all the links in a file
        for result in tqdm(results['organic_results'], desc='Downloading google top results'):
            write_to_file(result['link'], filename)#call to visit each link
    else:
        return ("No organic results found. Please check your query or API key.")#if error then api must be wrong or any new updates with serpapi


# function to collect reddit comments using praw
def collect_reddit_comments(query, filename,nos):
    reddit = praw.Reddit( #main class that communicate with reddit
        client_id=os.getenv("REDDIT_CLIENT_ID"),#to identify your application , created in reddit dev
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),#to authenticate that its you
        user_agent="commentcollector"#just a name given by you to your bot _-_
    )

    results = reddit.subreddit('all').search(query, limit=nos*20) #the number of reddit posts you want to fetch
    #here praw method of reddit.subreddit('all') gets results that has all subreddit later search query searches for the qurey and limit satets the number of results you want
    #if not os.path.exists('reddit/comments'):
    #    os.makedirs('reddit/comments')
    # for i in results:
    #     print(i)    #gives id of the serch of top 20  eg : '16jbjeu'


    with open(filename, 'a', encoding='utf-8') as f: #append to file
        for i, post in enumerate(tqdm(results, total=20, desc="Processing posts")):
            

            post.comments.replace_more(limit=0)#to read all more comments keep it 0 --__
                                                                                  # ----__
                                                                                  # -----__
            #the waterfall like structure in reddit is expanded , i.e more comment or reply on a comment
            comments = post.comments.list()[:1000]
            #take at most 1000 comments if less are availabe , it returns without error


            #writing it in the data.txt file
            for comment in tqdm(comments, total=1000, desc=f"Processing comments for post {i + 1}"):
                comment_text = comment.body + '\n\n\n'#separation of comments for my feasibility
                f.write(comment_text)

            print(f"Fetching post {i + 1}: {post.title}")


#step two : function to get video comments from youtube
def get_video_comments(api_key, video_id,nos):
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100"
    comments = []
    #you will get 100 res * 10 = 1000 for every video

    for i in range(20):
        response = requests.get(url)

        if response.status_code == 200:
            data = json.loads(response.text)

            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = data.get('nextPageToken')
            #if more comments i.e to get comments on next page
            if next_page_token:
                url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100&pageToken={next_page_token}"
            else:
                break
        else:
            print(f"Failed to get comments for video ID {video_id}. HTTP status code: {response.status_code}")
            break

    return comments

def get_top_videos(api_key, query, filename,nos):
    one_month_ago = datetime.now() - timedelta(days=30)#only get the results of last 30 days

    #print(one_month_ago)  #res : 2023-08-26 19:13:22.746893

    published_after = one_month_ago.isoformat("T") + "Z"  #the youtube api requires the date format in iso 8601 so here its converted that is micro to milli
    #yyyy-mm-dd hh:mm:ss.ssssss to yyyy-mm-dd hh:mm:ss.sss

    #in url i have put limit to 20 results it can changed by  changing maxResults variable value
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&part=snippet&type=video&maxResults=30&q={query}&publishedAfter={published_after}"
    #url to fetch found on youtube api v3  search endpoint on console


    response = requests.get(url)

    if response.status_code == 200:#check html status code if ok continue
        data = json.loads(response.text)

        with open(filename, 'a', encoding='utf-8') as f:
            for item in tqdm(data['items'], desc="Processing videos"):
                video_id = item['id']['videoId']#api
                title = item['snippet']['title']#api

                print(f"Processing video '{title}' with ID {video_id}...")

                comments = get_video_comments(api_key, video_id,nos)

                print(f"Got {len(comments)} comments for video '{title}' with ID {video_id}.")
                f.write('\n\n\n' + title + '\n\n\n')

                for comment in tqdm(comments, desc=f"Writing comments for video '{title}'"):
                    f.write(comment + '\n')

def save_query(query, file_path='queries.txt'):
    with open(file_path, 'a') as file:
        file.write(str(query) + '\n')

def get_latest_query(file_path='queries.txt'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Get the last non-empty line
    latest_query = None
    for line in reversed(lines):
        if line.strip():
            latest_query = line.strip()
            break
    return latest_query


# Main function to collect all data and write it to data.txt

# if __name__ == "__main__":
#     main()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text by lines
    lines = text.split('\n')

    preprocessed_lines = []
    for line in tqdm(lines, desc='Processing Lines'):
        # Tokenize the line into words
        word_tokens = word_tokenize(line)

        # Ignore lines with fewer than 3 words
        if len(word_tokens) < 3:
            continue

        # Remove stopwords
        filtered_text = [word for word in word_tokens if word not in stop_words]

        english_text = []
        for word in filtered_text:
            try:
                if detect(word) == 'en':
                    english_text.append(word)
            except LangDetectException:
                continue

        # Add the preprocessed line to the list of lines
        preprocessed_lines.append(' '.join(english_text))

    return preprocessed_lines

def file_to_dict(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    chat_dict = {}
    for line in tqdm(lines, desc="Processing chat words"):
        parts = line.strip().split()
        if len(parts) >= 2:
            key = parts[0]
            value = ' '.join(parts[1:])
            chat_dict[key] = value
        else:
            print(f"Skipping line: {line.strip()}")

    return chat_dict

def save_dict(chat_dict, dict_path):
    with open(dict_path, 'wb') as file:
        pickle.dump(chat_dict, file)

def load_dict(dict_path):
    with open(dict_path, 'rb') as file:
        return pickle.load(file)


def convert_chat_words(input_file_path, output_file_path, chat_dict):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Replace chat words with standard English words in each line
    converted_lines = []
    for line in tqdm(lines, desc="Converting chat words"):
        words = line.split()
        converted_words = [chat_dict.get(word, word) for word in words]
        converted_line = ' '.join(converted_words)
        converted_lines.append(converted_line)

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(converted_lines))


def emoji_to_text(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    text = emoji.demojize(content)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(text)




def get_longest_sentences(file_path, output_file_path, num_sentences):
    with open(file_path, 'r',encoding='utf8') as file:
        text = file.read()

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Sort the sentences by their length
    sentences.sort(key=len, reverse=True)

    # Get the longest sentences
    longest_sentences = sentences[:num_sentences]

    # Write the longest sentences to a text file
    with open(output_file_path, 'w',encoding='utf8') as file:
        for sentence in longest_sentences:
            file.write(sentence + '\n')




def extract_sentiment_sentences(input_file_path, output_file_path, batch_size=100, threshold=0.99):
    # Initialize the sentiment analysis pipeline
    nlp = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Filter sentences that contain sentiment
    sentiment_sentences = []
    for i in tqdm(range(0, len(lines), batch_size), desc="Extracting sentiment sentences"):
        batch = lines[i:i+batch_size]
        try:
            batch_sentiments = nlp(batch)
            for j, sentiment in enumerate(batch_sentiments):
                # Split the sentence into words
                words = batch[j].split()
                # If the sentence is not empty and (has more than 2 words or is strongly sentimental)
                if words and (len(words) > 2 or (len(words) <= 2 and sentiment['score'] > threshold)):
                    sentiment_sentences.append((batch[j], sentiment['label'], sentiment['score']))
        except Exception as e:
            print(f"Skipping batch due to error: {e}")

    # Write sentences with sentiment and their scores to a CSV file
    with open(output_file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sentence", "Sentiment", "Score"])
        for sentence, sentiment, score in sentiment_sentences:
            writer.writerow([sentence, sentiment, score])




# Check if the file exists
def deletefiles(query):
    file_path = f'CollectedData/{query}data.txt'
    file_path1 = f'CollectedData/{query}data_pre1.txt'
    file_path2 = f'CollectedData/{query}data_pre2.txt'
    file_path3 = f'CollectedData/{query}data_pre3.txt'
    file_path4=f'Results/{query}LS.txt'
    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(file_path1):
        os.remove(file_path1)
    if os.path.exists(file_path2):
        os.remove(file_path2)
    if os.path.exists(file_path3):
        os.remove(file_path3)
    if os.path.exists(file_path4):
        os.remove(file_path4)



def showRes(query):
    df = pd.read_csv(f'Results/{query}sentimentdata.csv')
    # Count the number of each sentiment
    sentiment_counts = df['Sentiment'].value_counts()

    # Plot the sentiment counts
    plt.figure(figsize=(8, 6))
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
    plt.title(f'Sentiment Analysis Results for {query}')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Sentences')
    plt.savefig(f'static/Results/{query}_plot.png')


def main(query,nos):
    create_directory_if_not_exists('Results')
    create_directory_if_not_exists('CollectedData')
    create_directory_if_not_exists('static/Results')



    file_path = f'Results/{query}sentimentdata.csv'
    if(os.path.isfile(file_path)):
        return
    call_install()
    start_time = time.time()#time elapsed calc
    api_key_youtube = os.getenv('YT_API_KEY')
    nos=2
    #query="Xperia XZ"
    #Query
    save_query(query)

    # Create a new folder called 'CollectedData' in the current directory


    # Define the data file path with a relative path
    data_filename = f'CollectedData/{query}data.txt'

    # Perform google search and collect data
    #perform_google_search(query+'review', data_filename)

    # Collect reddit comments and append them to data file
    collect_reddit_comments(query, data_filename,nos)

    # Get top youtube video's comments
    get_top_videos(api_key_youtube, query, data_filename,nos)

    end_time = time.time()
    elapsed_time = end_time - start_time
    chat_dict = file_to_dict('chat_words.txt')
    save_dict(chat_dict, 'chat_dict.pkl')
    chat_dict = load_dict('chat_dict.pkl')
    convert_chat_words(f'CollectedData/{query}data.txt', f'CollectedData/{query}data_pre1.txt', chat_dict)
    
    
    query = get_latest_query()
    
    get_longest_sentences(f'CollectedData/{query}data.txt', f'Results/{query}LS.txt', 50)
    emoji_to_text(f'CollectedData/{query}data_pre1.txt', f'CollectedData/{query}data_pre2.txt')

    with open(f'CollectedData/{query}data_pre2.txt', 'r',encoding='utf-8') as file:
        raw_text = file.read()

# Convert list of lines back to string, maintaining newlines
    preprocessed_text_str = '\n'.join(preprocess_text(raw_text))

# Write the preprocessed text to a new file
    with open(f'CollectedData/{query}data_pre3.txt', 'w', encoding='utf-8') as file:
        file.write(preprocessed_text_str)

    extract_sentiment_sentences(f'CollectedData/{query}data_pre3.txt', f'Results/{query}sentimentdata.csv')

    #getting topics

    with open(f'Results/{query}LS.txt', 'r',encoding='utf8') as file:
        text = file.read()

    sentences = nltk.sent_tokenize(text)

    # Extract keywords
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Perform topic modeling
    lda = LatentDirichletAllocation(n_components=10)
    lda.fit(X)

    # Now 'lda.components_' contains the topics
    # Each row corresponds to a topic, and each column corresponds to a word
    # The value in each cell is the importance of the word in the topic

    # To get the top 10 words in each topic:
    top_words = lda.components_.argsort()[:, -10:]

    # 'top_words' is now a 2D array where each row contains the indices of the top 10 words in a topic
    # You can map these indices back to words using 'vectorizer.get_feature_names()'

    # Write the top 10 words for each topic to 'ans.txt'
    with open(f'Results/{query}Topics.txt', 'w',encoding="utf8") as file:
        for i, topic in enumerate(top_words, start=1):
            words = [vectorizer.get_feature_names_out()[index] for index in topic]
            file.write(f'{" ".join(words)}\n')

    deletefiles(query)

    showRes(query)




    #print(f"Time elapsed: {elapsed_time} seconds")

