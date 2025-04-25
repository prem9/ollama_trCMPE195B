import requests
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain_chroma import Chroma

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
        #soup = BeautifulSoup(response.content, 'lxml')
        soup = BeautifulSoup(response.content, 'html.parser')


        paragraphs = soup.find_all('p')
        if not paragraphs:
            raise ValueError("Could not extract content from XML format.")

        text_content = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
        
        # Process the text content through TextLoader
        loader = TextLoader(text_content.strip())  # Ensure the content is loaded correctly
        raw_documents = loader.load()

    return raw_documents
def extract_pmc_article_xml(pmc_url):
    pmc_id = pmc_url.strip('/').split('/')[-1].replace('PMC', '')
    xml_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/?report=xml&format=text"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(xml_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'lxml')

    paragraphs = soup.find_all('p')
    if not paragraphs:
        raise ValueError("Could not extract content from XML format.")

    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
    return text.strip()
# Flask API Setup
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Call summarize_article to get the document content
    try:
        #raw_documents = summarize_article(url)
        raw_documents = TextLoader("./context.txt").load()
        # Split the document into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        documents = text_splitter.split_documents(raw_documents)

        # Initialize embeddings and vector store
        oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
        #db = FAISS.from_documents(documents, embedding=oembed)
        db = Chroma.from_documents(documents, embedding=oembed)

        # Set up the retrieval system and LLM
        query = "Summarize the article."
        docs = db.similarity_search(query)

        template = """Role and Context:
        You are an AI assistant with combined expertise equivalent to a Ph.D. in medical sciences and an MD specializing in oncology. Your goal is to assist cancer patients by summarizing complex articles in clear, concise language.
        
        {article_text}

        Thank you."""
        
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOllama(model="llama3.1:8b", temperature=0)
        # Use RunnableLambda to integrate your Python function
        url_to_text = RunnableLambda(lambda url: {"article_text": extract_pmc_article_xml(url)})
        # Create the chain to process the article and generate the summary
        chain = (
            RunnablePassthrough() 
            | url_to_text
            | prompt
            | model
            | StrOutputParser()
        )

        # Generate the summary
        summary = chain.invoke(url)

        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

