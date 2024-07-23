
# Recipe Search Engine

The Recipe Search Engine is a fully functional web application built in Python that allows users to search for recipes based on a list of ingredients. By utilizing a combination of  cosine similarity scores and euclidean distance scores , the application returns the top 5 recipes that best match the provided ingredients. This project demonstrates the use of machine learning techniques in a practical application.




## Features
 
 - **Ingredient-Based Search**: Users can input a list of ingredients, and the application will return relevant recipes.
- **Cosine Similarity**: The application uses cosine similarity to rank recipes based on ingredient matches.
- **Euclidean distance**: The application combines the scores of these metrics to rank the recipes based on the given ingredients list.
- **User-Friendly Interface**: A simple and intuitive web interface for easy interaction.
- **Recipe Details**: Each recipe includes details such as preparation time, cooking time, and instructions.

## Tech Stack

- Python
- Flask (Web Framework)
- Pandas (Data Manipulation)
- Scikit-learn (Cosine Similarity , Euclidean distance)

## Installation

Clone the git repository 

```bash
    git clone https://github.com/yourusername/recipe-search-engine.git

```
Navigate to the project directory:

```bash
    cd recipe-search-engine
```
Create a virtual environment if mac user and install the necessary libraries:
```bash
    python -m venv venv
```
Run the application:
```bash
    python app.py
```
Open your web browser and navigate to http://127.0.0.1:5000
## Usage/Examples

- Enter a list of ingredients in the input field, separated by commas.
- Click the "Search" button.
- The application will display the top 5 recipes that match your ingredients, along with their details.

