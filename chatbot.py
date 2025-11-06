import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
data = pd.read_csv("university_faq.csv")

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['clean_question'] = data['Question'].apply(preprocess)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_question'])

# Chatbot function
def chatbot_response(user_query):
    user_query_clean = preprocess(user_query)
    user_vec = vectorizer.transform([user_query_clean])
    similarities = cosine_similarity(user_vec, X)
    index = similarities.argmax()
    return data['Answer'][index]

# Run loop
print("🎓 University FAQ Chatbot (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() == 'exit':
        print("Chatbot: Goodbye! 👋")
        break
    print("Chatbot:", chatbot_response(query))
