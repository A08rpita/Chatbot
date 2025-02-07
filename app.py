import os
import requests
import logging
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS 
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Flask app setup
app = Flask(__name__)
api = Api(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Try to use GPU
    res = faiss.StandardGpuResources()
    print("FAISS GPU is available.")
except Exception:
    print("FAISS GPU is not available. Using CPU mode.")
    faiss.omp_set_num_threads(4)

# Function to extract data from the URL
def extract_data_from_url(url):
    """Fetches and extracts text from the provided URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises error for HTTP failures (4xx, 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Improved extraction: Include more content (e.g., course titles and descriptions)
        course_titles = [h2.get_text() for h2 in soup.find_all('h2')]
        course_descriptions = [p.get_text() for p in soup.find_all('p')]  # Extract descriptions
        return "\n".join(course_titles + course_descriptions)

    except requests.RequestException as e:
        logging.error(f"Error fetching data from URL: {e}")
        return ""

# Function to create embeddings and store in vector store
def create_vector_store(text):
    """Splits text into chunks and stores embeddings in FAISS vector database."""
    if not text.strip():
        logging.warning("No valid text found to create a vector store.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Increased overlap
    text_chunks = text_splitter.split_text(text)

    logging.info("Text Chunks: %s", text_chunks)  # Debugging the chunks

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get conversational chain (Fixed: Use `load_qa_chain` instead of `MapReduceChain`)
def get_conversational_chain():
    """Returns a conversational question-answering chain using LangChain's `load_qa_chain`."""
    prompt_template = """
    Answer the question based on the provided context. If the answer is not available in the context, 
    respond with: 'Answer is not available in the context.' Do NOT provide incorrect answers.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Use `load_qa_chain` with the "stuff" chain_type
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)  # You can also try "refine" or "map_reduce"
    return chain

# API Resource for handling user questions
class Chat(Resource):
    def post(self):
        """Handles chat requests and returns a response from the vector database."""
        user_question = request.json.get('question')
        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        try:
            # Load embeddings and the vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

            # Search for relevant documents
            docs = new_db.similarity_search(user_question)

            if not docs:
                return jsonify({"error": "No relevant documents found."}), 404  # Handle case of no relevant docs

            # Get the conversational chain
            chain = get_conversational_chain()

            # Call the chain to generate the response (Fix: Use `.invoke()`)
            response = chain.invoke({"input_documents": docs, "question": user_question})

            # Ensure we extract only the output_text from the response
            output_text = response.get("output_text", "No response generated.")

            # Return the extracted output text in the JSON response
            return jsonify({"reply": output_text})

        except Exception as e:
            logging.error(f"Error processing question: {e}")
            return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500

# API endpoint
api.add_resource(Chat, '/chat')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Main function to run the extraction and vector store creation
def main():
    """Extracts data from a URL and creates a vector store."""
    url = "https://brainlox.com/courses/category/technical"
    extracted_text = extract_data_from_url(url)
    if extracted_text:
        logging.info("Extracted text: %s", extracted_text)  # Debugging extracted text
        create_vector_store(extracted_text)
    else:
        logging.warning("No text extracted from the URL.")

if __name__ == "__main__":
    main()
    app.run(debug=True)
