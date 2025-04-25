import requests
# The URL to the API
api_url = "http://localhost:5000/summarize"
#api_url = "http://192.168.1.54:5000/summarize"

# Data to send in the POST request (URL of the document to summarize)
data = {
    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10046228/"  # Replace with the actual file path or URL
}
#https://pubmed.ncbi.nlm.nih.gov/30085525/
#https://pmc.ncbi.nlm.nih.gov/articles/PMC11012626/
# Sending a POST request to the Flask API
response = requests.post(api_url, json=data)

# Check if the request was successful
if response.status_code == 200:
    print(response.json())  # Should print the summary of the document
else:
    print(f"Error: {response.status_code}")
    print(response.text)  # Print the error message or response body
