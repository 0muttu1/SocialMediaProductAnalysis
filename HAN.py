import csv
import pandas as pd
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

def process_reviews(input_file, output_file, column_name):
    # Copy non-empty lines from input to output
    with open(input_file, 'r',encoding="utf8") as r, open(output_file, 'w',encoding="utf8") as o:
        for line in r:
            if line.strip():
                o.write(line)

    # Convert text to CSV
    def text_to_csv(input_file, output_file, column_name):
        with open(input_file, 'r',encoding="utf8") as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow([column_name])
            for line in infile:
                csv_writer.writerow([line.strip()])

    text_to_csv(output_file, 'Coutput.csv', column_name)

    df = pd.read_csv('Coutput.csv')
    df.dropna(inplace=True)

    # Remove HTML tags
    def remove_html_tags(txt):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', txt)

    df[column_name] = df[column_name].apply(remove_html_tags)

    # Remove URLs
    df[column_name] = df[column_name].replace(to_replace=r'^https?:\/\/.*[\r\n]*', value='', regex=True)

    # Remove punctuation
    exclude = string.punctuation
    def remove_punc(text):
        return text.translate(str.maketrans('', '', exclude))

    df[column_name] = df[column_name].apply(remove_punc)

    # Stem words
    stemmer = PorterStemmer()
    def stem_words(text):
        return " ".join([stemmer.stem(word) for word in text.split()])

    df[column_name] = df[column_name].apply(lambda text: stem_words(text))

    # Convert chat words
    chats = {}
    with open("chats.txt") as f:
        for line in f:
            (key, val) = line.split(None,1)
            chats[(key)] = val

    def chat_conversion(text):
        new_txt = []
        for w in text.split():
            if(w.upper() in chats):
                new_txt.append(chats[w.upper()])
            else:
                new_txt.append(w)
        return " ".join(new_txt)

    df[column_name] = df[column_name].apply(chat_conversion)

    # Remove stopwords
    STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    df[column_name] = df[column_name].apply(lambda text: remove_stopwords(text))

    # Demojize
    df[column_name] = df[column_name].apply(emoji.demojize)
    df[column_name] = df[column_name].str.lower()

    # Sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    def sentiment_score(review):
        tokens = tokenizer.encode(review, return_tensors='pt')
        result = model(tokens)
        return int(torch.argmax(result.logits)) + 1

    df['score'] = df[column_name].apply(lambda x: sentiment_score(x[:512]))

    def feeling(sentiment):
        if sentiment > 3:
            return 'positive'
        elif sentiment == 3:
            return 'neutral'
        else:
            return 'negative'

    df['sentiment'] = df['score'].apply(feeling)
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(sentiment_counts.index, sentiment_counts.values, color='lightblue')

    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Plot sentiment counts for keywords
    file_path = 'specifications.txt'
    with open(file_path, 'r') as file:
        file_content = file.read()
        keywords = file_content.split('\n')

    # for keyword in keywords:
    #     rows_with_keyword = df[df[column_name].str.lower().str.contains(keyword)].shape[0]
    #     if(rows_with_keyword < 1):
    #         continue

    #     keyword_counts = df[df[column_name].str.lower().str.contains(keyword, na=False)]['sentiment'].value_counts()
    #     keyword_counts.plot(kind='bar', color='lightgreen')
    #     plt.xlabel('Sentiment')
    #     plt.ylabel(f'Count of "{keyword}"')
    #     plt.title(f'Count of "{keyword}" in Each Sentiment')
    #     plt.show()

    return df


df = process_reviews('input.txt', 'output.txt', 'reviews')
