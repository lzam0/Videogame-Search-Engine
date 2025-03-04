# IR Course Work

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tabulate import tabulate
import matplotlib.pyplot as plt
import regex as re
import time

# For Stemming
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm") 

import os
from bs4 import BeautifulSoup


def read_html(directory):
    """
    Reads all HTML files in videogames folder to extract game information,
    apply weighting to the content, and retrieve metadata such as titles and file names.

    Test how long it takes to take in all documents and process them

    Args:
        directory (str): Path to the directory containing HTML files.

    Returns:
        tuple: Contains the following lists:
            - weighted_documents (list of str): Weighted content of HTML files.
            - game_info_list (list of dict): Extracted game information for each file.
            - titles (list of str): Titles of the HTML files.
            - file_names (list of str): File names of the HTML files.
    """
    
    weighted_documents = []
    game_info_list = []
    titles = []
    file_names = []

    # Check if the folder exists
    if not os.path.exists(directory):
        print(f"The folder '{directory}' does not exist.")
        return
    
    start_time = time.time()

    # iterate through all files in the folder
    for filename in os.listdir(directory):
        if filename.endswith('.html'):
            file_path = os.path.join(directory, filename)

            # Extract content from the "Game Info" section
            game_info = extract_content(file_path)
            game_info_list.append(game_info)

            # Process the document and apply extra weight to Publisher and Genre
            weighted_document = process_content_with_weighting(file_path, game_info)
            weighted_documents.append(weighted_document)

            # Open the file and parse the title
            with open(file_path, 'r') as file:
                soup = BeautifulSoup(file, 'html.parser')
                titles.append(soup.title.string if soup.title else 'No Title')

            # Add the filename for the table
            file_names.append(filename)
            print(f"{filename} imported")

    # testing the efficiencey
    time_taken = time.time() - start_time
    #return time_taken

    return weighted_documents, game_info_list, titles, file_names

def extract_content(html_file):
    """
    Extracts content from an HTML file based on section names like 'DEVELOPER', 'PUBLISHER', etc.
    
    Parameters:
    html_file (str): The path to the HTML file.

    Returns:
    dict: A dictionary containing the extracted game information, with keys:
          - 'developer': Name of the game's developer.
          - 'publisher': Name of the game's publisher.
          - 'genre': The genre of the game.
          - 'esrb': The ESRB rating of the game
    """

    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Extract relevant data from the "Game Info" section
    game_info = {}

    # Developer
    developer = soup.find('td', text='Developer')
    if developer:
        game_info['developer'] = developer.find_next('td').get_text(strip=True)

    # Publisher
    publisher = soup.find('td', text='Publisher')
    if publisher:
        game_info['publisher'] = publisher.find_next('td').get_text(strip=True)

    # Genre
    genre = soup.find('td', text='Genre')
    if genre:
        game_info['genre'] = genre.find_next('td').get_text(strip=True)


    # ESRB
    esrb = soup.find('td', text='ESRB')
    if esrb:
        esrb_link = esrb.find_next('td').find('a')  # Find the <a> tag inside the <td>
        game_info['esrb'] = esrb_link.get_text(strip=True) if esrb_link else None


    return game_info

def process_content_with_weighting(html_file, game_info):
    """
    Process the HTML content and give extra weight to important sections.

    Parameters:
    html_file (str): The path to the HTML file.
    game_info (dict): The extracted game information.

    Returns:
    str: A weighted version of the document text with preprocessed content.
    """
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Extract and preprocess the document
    document = preprocess_text(soup.get_text())[0]

     # Extract named entities and apply extra weight
    named_entities = extract_named_entities(soup.get_text())
    weighted_named_entities = ' '.join([f"{entity[0]} " * 3 for entity in named_entities])  # Weight by repeating entities

    # Preprocess weighted sections
    developer = preprocess_text(game_info.get('developer', ''))[0]
    publisher = preprocess_text(game_info.get('publisher', ''))[0]
    genre = preprocess_text(game_info.get('genre', ''))[0]
    esrb = preprocess_text(game_info.get('esrb', ''))[0]

    # Add more weight to specific content
    weighted_document = (document + " " + 
                         (developer * 3) + " " + 
                         (publisher * 3) + " " + 
                         (genre * 3) + " " + 
                         (esrb * 3) +
                         weighted_named_entities)

    return weighted_document


def preprocess_text(text):
    """
    Preprocesses the input text by performing the following:
    - Convert text to lowercase.
    - Remove HTML tags and special characters.
    - Remove extra whitespace.
    - Lemmatize tokens.
    - Remove stopwords.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize text using spaCy
    doc = nlp(text)

    # Lemmatize tokens and remove stopwords
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    # Stemming tokens and remove stopwords
    # tokens = [
    #     stemmer.stem(token.text) for token in doc
    #     if not token.is_stop and not token.is_punct and not token.is_space
    # ]

    # Extract named entities
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text, named_entities

def extract_named_entities(text):
    """
    Extract named entities from text using spaCy's NER model.

    Parameters:
    text (str): The input text to extract named entities from.

    Returns:
    list: A list of named entities identified in the text.
    """
    doc = nlp(text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    return named_entities

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def search_query(query, weighted_documents):
    """
    Searches for the relevance of a given query in a set of weighted documents using cosine similarity.

    Parameters:
        query (str): The search query string.
        weighted_documents (list of str): A list of documents, each represented as a string, to be searched through.

    Returns:
        numpy.ndarray: A 2D array containing the cosine similarity scores between the query and each document.
    """
    # Vectorize the documents with the custom tokenizer
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer)

    # Premade Tokenizer
    #vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(weighted_documents)

    # Process the query and then Vectorize the query
    processed_query = preprocess_text(query)[0]
    query_vector = vectorizer.transform([processed_query])

    # Compute similarity (e.g., cosine similarity)
    similarity = cosine_similarity(query_vector, tfidf_matrix)

    return similarity

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def custom_tokenizer(text):
    """
    Custom Tokenizer that tokenizesthe provided text, utilising spaCy, comparing lemmatization and stemming

    Parameters:
    text (str): Input text to be tokenized

    Returns:
    list: A list of tokens after lemmatization and stemming.

    """
    # List to store processed tokens
    doc = nlp(text)

    tokens = []

    for token in doc:
        # Normalize whitespace and remove punctuation
        clean_token = re.sub(r'\s+', ' ', token.text).strip()  # Normalize whitespace
        clean_token = re.sub(r'[^\w\s]', '', clean_token)  # Remove punctuation

        # Skip punctuation and space-only tokens
        if not token.is_punct and not token.is_space: #and clean_token.lower() not in stop_words:
            # Lemmatize token and convert it to lowercase
            lemmatized_token = token.lemma_.lower()

            # Stemming the token and converting it to lowercase
            stemmed_token = stemmer.stem(token.text.lower())

            # Test between the two
            tokens.append(lemmatized_token)
            #tokens.append(stemmed_token)
    return tokens

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def efficiency_testing():
    """
    Efficiency Testing between lemmatization and stemming
    """
    directory = 'videogames'
    
    counter = 1
    time_taken = []
    while counter < 11:
        print("Run Counter Value", counter)

        # Start timing the test run
        start_time = time.time()
        test_run = read_html(directory)
        end_time = time.time()
        
        # Calculate the time taken for this test run
        run_time = end_time - start_time
        print(f"READ HTML Function took {run_time:.4f} seconds to run")
        
        # Append the time taken for this run
        time_taken.append(run_time)
        
        counter += 1
    
    # Calculate the average time taken
    avg_time = sum(time_taken) / len(time_taken)
    print(f"Average time taken: {avg_time:.4f} seconds")
    
    # Create the box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(time_taken)
    plt.title("Box Plot of Time Taken for Efficiency Testing (Stemming)")
    plt.ylabel("Time (seconds)")
    plt.xticks([1], ['Test Run Time'])  # Label x-axis
    plt.grid(True)
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    directory = 'videogames'
    
    # Process HTML files
    weighted_documents, game_info_list, titles, file_names = read_html(directory)

    print(f"{len(game_info_list)} files imported")
    
    search_terms = ["ICO", "Okami", "Devil Kings",
                    "Dynasty Warriors", "Sports Genre Games",
                    "Hunting Genre Games", "Game Developed By Eurocom",
                    "Game Published by Activision",
                      "Game Published by Sony Computer Entertainment",
                      "Teen PS2 Games"]
    # Create a loop for input query
    while True:
        # User input query
        query = input("Enter a search query (or type 'stop' to exit): ")
        
        # break loop if "stop entered"
        if query.lower() == "stop":
            break

        # Search the query in the weighted documents
        similarity = search_query(query, weighted_documents)

        # Prepare the results with their similarity scores
        results = []
        for idx, score in enumerate(similarity[0]):
            game_info = game_info_list[idx]
            results.append([
                titles[idx],
                file_names[idx],
                game_info.get('developer', 'N/A'),
                game_info.get('publisher', 'N/A'),
                game_info.get('genre', 'N/A'),
                game_info.get('esrb', 'N/A'),
                score
            ])

        # Sort the results by similarity score in descending order
        sorted_results = sorted(results, key=lambda x: x[6], reverse=True)

        # Display the top 10 similarity scores
        top_10_results = sorted_results[:10]

        # Define table headers
        headers = ["Title", "File Name", "Developer", "Publisher", "Genre", "ESRB", "Similarity"]

        # Print table using tabulate
        print(f"\nResults for query '{query}':")
        print(tabulate(top_10_results, headers=headers, tablefmt="grid"))

        # Display the data to the user as a graph
        top_titles = [result[0] for result in top_10_results]
        top_similarities = [result[6] for result in top_10_results]

        # Create a bar chart for the results
        plt.figure(figsize=(12, 6))
        bars = plt.barh(top_titles, top_similarities, color='skyblue')
        plt.xlabel('Similarity Score')
        plt.ylabel('Document Title')
        plt.title(f"Search Results for Query: '{query}'")
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
        plt.tight_layout()
        
        # Include the similarity score next to each bar
        for bar, similarity in zip(bars, top_similarities):
            plt.text(
                bar.get_width() + 0.01,  # Offset to the right of the bar
                bar.get_y() + bar.get_height() / 2,
                f"{float(similarity):.4f}",
                va='center',
                ha='left',
                color='black',
                fontweight='bold'
            )
        
        # Display the plot
        plt.show()

        
if __name__ == "__main__":
    #efficiency_testing()
    #test_stopword_effect()
    main()