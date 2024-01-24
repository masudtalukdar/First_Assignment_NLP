import wikipediaapi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

### Function to get Wikipedia content
def get_wikipedia_content(title, lang='en'):
    user_agent = "New_Assignment_Nlp/1.1 (masudtalukdar.cn@email.com)"  ### Update with your own user agent
    wiki_wiki = wikipediaapi.Wikipedia(lang, headers={'User-Agent': user_agent})
    page_py = wiki_wiki.page(title)

    if not page_py.exists():
        print(f"Page '{title}' does not exist.")
        return None

    return page_py.text

### Function to create a dataset from Wikipedia
def create_wikipedia_dataset(geographic_title, non_geographic_title, num_samples=500):
    ### Fetch Wikipedia content
    geographic_content = get_wikipedia_content(geographic_title)
    non_geographic_content = get_wikipedia_content(non_geographic_title)

    ### Create a DataFrame with the data, using different column names
    data = pd.DataFrame({
        'Content': [geographic_content] * num_samples + [non_geographic_content] * num_samples,
        'Class': ['geographic'] * num_samples + ['non-geographic'] * num_samples
    })
    ### Remove rows with None values
    data = data.dropna()

    return data

### Sample geographic and non-geographic Wikipedia articles
geographic_title = "Geography_of_Asia"
non_geographic_title = "Computer_science"

### Create the dataset
dataset = create_wikipedia_dataset(geographic_title, non_geographic_title)

### Print the Dataset
print(dataset)

### Data Preprocessing
dataset['Content'] = dataset['Content'].apply(lambda x: x.lower() if x else x)
print("Number of samples per class:")
print(dataset['Class'].value_counts())

### Text Cleaning and Tokenization
stopwords_set = set(stopwords.words('english'))
snowball_stemmer = SnowballStemmer(language='english')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords_set]
    tokens = [snowball_stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

dataset['Content'] = dataset['Content'].apply(preprocess_text)

### Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset['Content'])
y = dataset['Class']

### Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

### Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

### Evaluate the model
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

### Example prediction on a new text
new_text = "This is a new text about geographic topics."
new_text_vectorized = vectorizer.transform([new_text])
prediction = model.predict(new_text_vectorized)
print(f'Prediction for the new text: {prediction}')

### Save the trained model
model_filename = 'classifier_model.joblib'
joblib.dump(model, model_filename)

### Save the TF-IDF vectorizer
vectorizer_filename = 'vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)

print(f"Model and vectorizer saved as {model_filename} and {vectorizer_filename}")
