import os
import requests
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import openai

# OpenAI API kulcs közvetlen beállítása
os.environ['OPENAI_API_KEY'] = 'sk-PH7IzQpiAMGRPyKqz728dmXEXHa1fUMAQXinQTBr52T3BlbkFJ2dDGhagm4lpzGwlKnG3Wn2E0j12eEZS3LGBYfC5B0A'

# Setup Flask app
app = Flask(__name__)

# Create OpenAI client using the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize vector database
persist_directory = '/Users/binobenjamin/Documents/Vector-Langchain/chroma_data'
raw_documents = TextLoader('/Users/binobenjamin/Documents/Vector-Langchain/Próba2.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Store documents in vector database
db = Chroma.from_documents(
    documents,
    OpenAIEmbeddings(),
    persist_directory=persist_directory
)

# Function to scrape data from a given URL
def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            return ' '.join(paragraphs)[:1000]
        else:
            return "Error: Unable to access the website."
    except Exception as e:
        return f"Error scraping the website: {str(e)}"

# Serve the HTML frontend
@app.route('/')
def home():
    return render_template('index.html')

# Chat route to process user messages and call the GPT-3.5-turbo API
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    url_to_scrape = request.json.get('url')

    if not user_input:
        return jsonify({'response': "Error: No input provided"}), 400

    scraped_data = ''
    if url_to_scrape:
        scraped_data = scrape_website(url_to_scrape)

    try:
        # Search for matching document in vector database
        docs = db.similarity_search(user_input)
        if docs:
            document_match = docs[0].page_content
        else:
            document_match = "No matching document found."

        # Add the most relevant document directly to the system message for context
        system_message = (
            "You are a helpful assistant. Below is information scraped from a website and a related document that might be helpful:\n\n"
            f"Web scraped data: {scraped_data}\n\n"
            f"Relevant document: {document_match}\n\n"
            "Now, respond to the user's question in detail."
        )

        # Make an API call to the model with the initial system message
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
            ]
        )
        bot_response = response['choices'][0]['message']['content']

        return jsonify({'response': bot_response})

    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
