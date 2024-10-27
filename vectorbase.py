import faiss
import numpy as np
import google.generativeai as genai
import pandas as pd


class vectorbase:
    def __init__(self):
        self.index=faiss.read_index("Assets/faiss_index.bin")
        self.data=pd.read_csv("Assets/nasdaq_with_summaries.csv")
        self.data['Summary'] = self.data['Summary'].fillna(" ")
        self.data['Industry'] = self.data['Industry'].fillna(" ")
        self.data['Sector'] = self.data['Sector'].fillna(" ")

        # Select relevant columns for embedding
        self.data['Description'] = self.data['Summary'] + " " + self.data['Industry'] + " " + self.data['Sector']

    # Function to perform search based on the FAISS index
    def find_similar_stocks(self,query, top_k=6):
        # Generate an embedding for the query text
        query_embedding = genai.embed_content(model="models/text-embedding-004", content=[query])['embedding'][0]
        query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

        # Search the FAISS index with the query embedding
        distances, indices = self.index.search(query_vector, top_k)

        # Retrieve and display the top matching results, sorted by Market Cap
        results = self.data.iloc[indices[0]].assign(distance=distances[0]).sort_values(by='Market Cap', ascending=False)  # Change to ascending=True for ascending sort

        return results[['Symbol', 'Name', 'Industry', 'Sector', 'Summary', 'distance', 'Market Cap']]