from django.shortcuts import render
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from collections import Counter
import os
import nltk
from nltk.corpus import stopwords
import spacy
from pygooglenews import GoogleNews
import numpy as np
import requests
import spacy
from collections import Counter



#check for internet
def has_internet_connection():
    try:
        requests.get("https://google.com", timeout=5)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False
    
# Download the SpaCy model
if not os.path.exists(os.path.join(nltk.data.find('corpora'), 'stopwords')):
    nltk.download('punkt')
    nltk.download('stopwords')

# Download the SpaCy model if not already downloaded
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")

# Load the downloaded SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load Keras model
model = load_model("C:\\Users\\Admin\\Downloads\\newlstm.h5")
model1 = load_model("E:\models\models\modelrightepoch10.h5")


# Initialize Tokenizer
tokenizer = Tokenizer()

#preprocess the text to remove stopwords
def preprocess_text(text):
    if isinstance(text, list): 
        text = ' '.join(text)
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

#take user input to send for preprocessing 
def preprocess_input(user_input, maxlen):
    user_input = preprocess_text(user_input)
    tokenizer.fit_on_texts([user_input])
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    return pad_sequences(user_input_sequence, maxlen=maxlen)


#model1 prediciton
def predict_fake_news(user_input):
    processed_input = preprocess_input(user_input, 1000)
    prediction = model.predict(processed_input)[0, 0]
    return prediction

#
def NewsDisplay(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        if user_input:
            prediction = predict_fake_news(user_input)
            result = 'This news is likely real.' if prediction > 0.8 else 'This news is likely fake.'
            return render(request, 'homePage/homeindex.html', {'result': result, 'prediction': prediction})
    return render(request, "homePage/homeindex.html")

#determine the max stace based on the prediction
def determine_stance(prediction):  
    stances = ['Agree', 'Disagree', 'Discuss', 'Unrelated']
    max_index = np.argmax(prediction)
    return stances[max_index]


def check_stances(claim_text, articles, max_articles=5):
    processed_claim = preprocess_text(claim_text)
    processed_articles = [preprocess_text(article['title']) for article in articles[:max_articles]]
    headline_sequences = [preprocess_input(article, 1000) for article in processed_articles]
    body_sequence = preprocess_input(processed_claim, 2000)

    predictions1 = [model1.predict([headline_sequence, body_sequence]) for headline_sequence in headline_sequences]
   

    predicted_stances = []
    for article, p1 in zip(articles[:max_articles], predictions1):
        max_p1_value = determine_stance(p1)
       


         
        predicted_stances.append({'title': article['title'],'link':article['link'] , 'maxp1': max_p1_value})

    return predicted_stances



def verify_claim(claim_text, max_articles=5):
    gn = GoogleNews(lang='en',country='US')
    search_results = gn.search(preprocess_text(claim_text),limit=5)
    articles = search_results['entries'][:max_articles] 
    claim_results = check_stances(claim_text, articles)
    stance_counts = Counter(article['maxp1'] for article in claim_results) 
    most_common_stance, _ = stance_counts.most_common(1)[0]
 
    return claim_results
    

  

def ClaimCheck(request):
    if request.method == 'POST':
        claim_text = request.POST.get('user_input')
        prediction = predict_fake_news(claim_text)
        result = 'This news is likely real.' if prediction > 0.6 else 'This news is likely fake.'  
        if(has_internet_connection()):
            if claim_text:
                claim_results = verify_claim(claim_text)
                # Count occurrences of each stance
                stances = [article['maxp1'] for article in claim_results]
                stance_counts = Counter(stances)
                # Find the most frequent stance
                most_frequent_stance = stance_counts.most_common(1)[0][0]
                
                context = {'claim_results': claim_results,
                           'result': result,
                           'prediction': prediction,
                           'most_frequent_stance': most_frequent_stance}
                return render(request, "homePage/claimchecking.html", context)
        else:
            context = {'prediction': prediction,'result': result}
            return render(request, "homePage/result.html", context)
    return render(request, "homePage/claimchecking.html")


