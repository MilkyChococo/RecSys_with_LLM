from preprocess.preprocess import QueryCleaner
from Data.qadataset import ProductQADataset
from search.retriever import SemanticSearchQA
from sentence_transformers import SentenceTransformer
import torch
import json
import os
def main(user_query):
    # Load the dataset
    dataset = ProductQADataset("Data/product_recommendation.jsonl")
    # Initialize the cleaner and model
    cleaner = QueryCleaner("preprocess/english_stopwords.json")

    # Initialize the model
    model_path = os.path.abspath(r"C:\Users\GIGABYTE\OneDrive\Desktop\ReSys LLm\model\fine-tuned-miniLM-model")
    model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the retriever
    retriever = SemanticSearchQA(model, cleaner, dataset.get_all())

    # Test the retriever with a sample query
    result = retriever.search(user_query)
    
    print(result)