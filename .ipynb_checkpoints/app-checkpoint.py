from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Load your DataFrame 'df_sel'
# Replace this with your actual data loading code
df_sel = pd.read_csv('Food Ingredients and Recipe Dataset with Image Name Mapping.csv')  # Adjust file path accordingly

# Preprocess the ingredients column
def preprocess(text):
    # Remove punctuation and convert to lowercase
    return text.lower().replace('[^\w\s]', '')

# Apply preprocessing to ingredients
df_sel['Ingredients'] = df_sel['Ingredients'].apply(preprocess)

# Vectorize ingredients using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_sel['Ingredients'])
# Perform LSI
lsa_model = TruncatedSVD(n_components=100, random_state=42)
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

def get_top_matching_recipes(input_ingredients, df_sel, top_n=5):
    # Preprocess input ingredients
    input_text = ' '.join(input_ingredients)
    input_tfidf = vectorizer.transform([input_text])
    input_lsa = lsa_model.transform(input_tfidf)

    # Calculate cosine similarity between input and recipes
    similarity_scores = cosine_similarity(input_lsa, lsa_matrix)
    
    # Calculate Euclidean distance between input and recipes 
    euclidean_scores = euclidean_distances(input_lsa, lsa_matrix)

    # Combine the scores of both cosine and euclidean distance 
    combine_score = 0.5 * similarity_scores + 0.5 * (1 / (1 + euclidean_scores))
    
    # Get indices of top matching recipes
    top_indices = combine_score.argsort(axis=1).flatten()[-top_n:][::-1]

    # Get top matching recipes
    top_recipes = [(df_sel.iloc[i]['Title'], combine_score[0, i]) for i in top_indices]

    return top_recipes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    ingredients = request.form['ingredients'].split(',')
    top_recipes = get_top_matching_recipes(ingredients, df_sel)
    return render_template('result.html', top_recipes=top_recipes)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port number to your desired port
