from Data.qadataset import ProductQADataset
from preprocess.preprocess import QueryCleaner
def main():
    dataset = ProductQADataset("Data\product_recommendation.jsonl")
    queries = dataset.get_queries()
    
    answers = dataset.get_answers()
    cleaner = QueryCleaner("/mnt/data/english_stopwords.json")
    corpus_queries = [cleaner.clean(q) for q in corpus_queries]
    print(queries[0])
    print(answers[0])
    

