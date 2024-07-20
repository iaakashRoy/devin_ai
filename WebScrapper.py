import os
import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

# Todo - Although we're filtering the top chunks, but we need to filter them based on a threshold value with certain conditions
# Condition 1: while adding chunks, will go through the chunks and see if it's over the threshold value.
# If it is, then we'll add it to the list of top chunks of the website. If not, we skip that website.
# If the total number of chunks<5, then we search for more 2 websites and repeat the process.
# At the end we can rank the chunks and choose the top 5 or 10 of them.
    
class WebScrapper:
    def __init__(self, chunk_size=200, chunk_overlap=40, top_k=5):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def google_search(self, query, num_results=5):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': os.getenv("GOOGLE_API_KEY"),
            'cx': os.getenv("GOOGLE_CSE_ID"),
            'num': num_results
        }
        response = requests.get(url, params=params)
        results = response.json()
        return results

    @staticmethod
    def custom_extractor(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    def web_scrape(self, url):
        loader = RecursiveUrlLoader(url=url, extractor=self.custom_extractor)
        docs = loader.load()
        for doc in docs:
            content = doc.page_content
        if isinstance(content, str):
            return content
        else:
            return "Skipped a document due to non-string content."

    @staticmethod
    def replace_(text, replace_list):
        for item in replace_list:
            text = text.replace(item, '')
        text = text.replace('\n', ' ')
        return text

    def top_scraped_results(self, query: str):
            results = self.google_search(query)
            url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            result_dict = {item['title']: item['link'] for item in results.get('items', [])}
            urls = []
            for key, value in result_dict.items():
                chunks = self.text_splitter.split_text(self.web_scrape(value))

                # Embed the chunks and the query
                if not chunks:
                    continue

                chunk_embeddings = self.model.encode(chunks)
                query_embedding = self.model.encode([query])

                # Ensure embeddings are 2D arrays
                if chunk_embeddings.ndim == 1:
                    chunk_embeddings = chunk_embeddings.reshape(1, -1)
                if query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)

                # Compute cosine similarity between the query embedding and the chunk embeddings
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

                # Get the top k chunks with the highest similarity scores
                top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
                top_chunks = []

                for i in top_k_indices:
                    urls_ = re.findall(url_pattern, chunks[i])
                    top_chunks.append(self.replace_(chunks[i], urls_))
                    urls += urls_
                result_dict[key] = top_chunks

            result_dict["urls"] = urls

            return result_dict