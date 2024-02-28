from django.shortcuts import render
from django.http import JsonResponse
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from django import forms

import nltk
from nltk.corpus import stopwords
import spacy
from pygooglenews import GoogleNews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load English language model for NER
nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
nltk.download('stopwords')

# Load Keras model
model = load_model("C:\\Users\\Admin\\Downloads\\yt_balanced_lstm1.h5")
model1 = load_model("E:\\models\\models\\model.h5")
model2 = load_model("E:\\models\\models\\modelright.h5")

# Initialize Tokenizer
tokenizer = Tokenizer()

def preprocess_text(text):
    if isinstance(text, list):  # Check if input is a list
        # Concatenate the list elements into a single string
        text = ' '.join(text)
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_input(user_input, maxlen):
    user_input = preprocess_text(user_input)
    tokenizer.fit_on_texts([user_input])
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    return pad_sequences(user_input_sequence, maxlen=maxlen)

def predict_fake_news(user_input):
    processed_input = preprocess_input(user_input, 1000)
    prediction = model.predict(processed_input)[0, 0]
    return prediction

def NewsDisplay(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        if user_input:
            prediction = predict_fake_news(user_input)
            result = 'This news is likely real.' if prediction > 0.8 else 'This news is likely fake.'
            return render(request, 'homePage/homeindex.html', {'result': result, 'prediction': prediction})
    return render(request, "homePage/homeindex.html")

##def calculate_similarity(claim, articles):
    processed_claim = preprocess_text(claim)
    processed_articles = [preprocess_text(article) for article in articles]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_claim] + processed_articles)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    return similarity_scores

def check_stances(claim_text, articles):
    processed_claim = preprocess_text(claim_text)
    processed_articles = [preprocess_text(article) for article in articles]
    headline_sequence = preprocess_input(processed_articles, 1000)
    body_sequence = preprocess_input(processed_claim, 2000)

    prediction1 = model1.predict([headline_sequence, body_sequence])
    prediction2 = model2.predict([headline_sequence, body_sequence])
    all_predictions = np.concatenate([prediction1, prediction2])
  

    predicted_stance = {'p1': prediction1, 'p2': prediction2}

    return predicted_stance

def verify_claim(claim_text):
    gn = GoogleNews(lang='en', country='US')
    search_results = gn.search(preprocess_text(claim_text))
    articles = [entry['title'] for entry in search_results['entries']]
    predicted_stance = check_stances(claim_text, articles)
    claim_results = []
    for article, p1, p2 in zip(articles, predicted_stance['p1'], predicted_stance['p2']):
        avg_stance = (p1 + p2) / 2  # Calculate the average stance
        claim_results.append((article, avg_stance))
    
    return claim_results

def ClaimCheck(request):
    if request.method == 'POST':
        claim_text = request.POST.get('user_input')
        if claim_text:
            claim_results = verify_claim(claim_text)
            # Limit the number of results to 10
            claim_results = claim_results[:10]
            context = {'claim_results': claim_results}
            return render(request, "homePage/claimchecking.html", context)
    return render(request, "homePage/claimchecking.html")
