import requests
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

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
        soup = BeautifulSoup(response.content, 'lxml')

        paragraphs = soup.find_all('p')
        if not paragraphs:
            raise ValueError("Could not extract content from XML format.")

        text_content = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
        # Use TextLoader to process the text content
        loader = TextLoader(text_content.strip())  # You can customize this part as needed
        raw_documents = loader.load()

    
    return raw_documents 


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL is required'}), 400
    raw_documents_c = TextLoader("./context.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    documents = text_splitter.split_documents(raw_documents_c)
    # Initialize embeddings and vector store
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    db = Chroma.from_documents(documents, embedding=oembed)
    query = "Summarize the article about Carboplatin usage in cancer treatment."
    docs = db.similarity_search(query)
    template = """Role and Context
    You are an AI assistant with combined expertise equivalent to a Ph.D. in medical sciences and an MD specializing in oncology. Your primary goal is to assist cancer patients by summarizing complex medical and scientific articles from sources like PubMed or other online medical journals into clear, concise, and easy-to-understand language. Patients will provide you with URLs to articles or directly paste article text for summarization.
    Summarization Instructions
    When summarizing:
    Convert complex medical jargon into plain language suitable for laypersons.
    Clearly outline key findings, conclusions, and implications relevant to cancer patients.
    Ensure the summary is accurate, concise, and informative.
    Ethical and Professional Considerations
    Clearly indicate where information provided is general guidance versus content directly extracted from the specific article.
    Provide balanced insights to support informed decision-making by patients but explicitly advise consultation with healthcare professionals before acting upon recommendations.
    Overall Goal
    Empower cancer patients by translating scientific literature into actionable knowledge, supporting informed decisions about treatment options, supplementary therapies, and overall health management.
    For further explaining complex medical terms like P-value, Confidence Interval, Correlation Coefficient, Hazard ratio etc., 


    {article_text}

    Thank you."""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOllama(
        model="llama3.1:latest",
        temperature=0
    )
    retriever = db.as_retriever()
  #  vector_store = FAISS.from_documents(raw_documents, embed)
    # Use RunnableLambda to integrate your Python function
    url_to_text = RunnableLambda(lambda url: {"article_text": summarize_article(url)})
    # Initialize LLM

    #llm = OllamaLLM(model="monic-chat")

    # Create the RetrievalQA chain

    #qa_chain = RetrievalQA.from_chain_type(
    #    llm=llm,
    #    chain_type="stuff",  # Use appropriate chain type
    #    vectorstore=vector_store
    #)

    # Query for summarization (you can replace this with more dynamic logic)
    #query = "Summarize the article."
    #response = qa_chain.run(query)
    chain = (
    RunnablePassthrough()
    | url_to_text
    | prompt
    | model
    | StrOutputParser()
    )
    #summary = chain.invoke("https://pmc.ncbi.nlm.nih.gov/articles/PMC10046228/")
    # Call the summarize_article function with the provided URL
    try:
        summary = chain.invoke(url)
        #summary = summarize_article(url)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
