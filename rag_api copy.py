from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM

app = Flask(__name__)

# Function to load documents and summarize
def summarize_article(url):
    # Step 1: Load the document
    if url.endswith('.pdf'):
        loader = PyPDFLoader(url)
        raw_documents = loader.load()
    else:
        loader = TextLoader(url)
        raw_documents = loader.load()

    # Step 2: Initialize embeddings and vector store
    embed = OllamaEmbeddings(base_url="http://localhost:11434", model="monic-embed-text")
    vector_store = FAISS.from_documents(raw_documents, embed)

    # Step 3: Initialize LLM
    llm = OllamaLLM(model="monic-chat")

    # Step 4: Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use appropriate chain type
        vectorstore=vector_store
    )

    # Step 5: Query for summarization (you can replace this with more dynamic logic)
    query = "Summarize the article."
    response = qa_chain.run(query)

    return response

# Define a route for summarizing articles
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Call the summarize_article function with the provided URL
    summary = summarize_article(url)

    # Return the summary in JSON format
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
