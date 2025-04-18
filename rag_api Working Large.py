import requests
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM

def summarize_article(url):
    # Add headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # If the URL is a PDF
    if url.endswith('.pdf'):
        loader = PyPDFLoader(url)
        raw_documents = loader.load()
    else:
        # If it's a regular URL, download the content
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch URL: {url} with status code {response.status_code}")
        
        # Assuming the content is text-based (you can adjust this if needed)
        text_content = response.text
        
        # Use TextLoader to process the text content
        loader = TextLoader(text_content)  # You can customize this part as needed
        raw_documents = loader.load()

    # Initialize embeddings and vector store
    embed = OllamaEmbeddings(base_url="http://localhost:11434", model="monic-embed-text")
    vector_store = FAISS.from_documents(raw_documents, embed)

    # Initialize LLM
    llm = OllamaLLM(model="monic-chat")

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use appropriate chain type
        vectorstore=vector_store
    )

    # Query for summarization (you can replace this with more dynamic logic)
    query = "Summarize the article."
    response = qa_chain.run(query)

    return response


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Call the summarize_article function with the provided URL
    try:
        summary = summarize_article(url)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
