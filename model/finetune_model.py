from sentence_transformers import SentenceTransformer, InputExample, losses
from Data.qadataset import ProductQADataset
from preprocess.preprocess import QueryCleaner
from torch.utils.data import DataLoader
import torch
import wandb
wandb.login(key="your_wandb_api_key")  # Replace with your actual WandB API key
#from huggingface_hub import login
#login("your_huggingface_token")
def main():
    # Load the dataset
    dataset = ProductQADataset("Data\product_recommendation.jsonl")
    queries = dataset.get_queries()
    answers = dataset.get_answers()

    # Initialize the cleaner and model
    cleaner = QueryCleaner("preprocess\english_stopwords.json")
    corpus_queries = [cleaner.clean(q) for q in queries]

    # Create training examples
    train_examples = [
        InputExample(texts=[q, a], label=1.0) for q, a in zip(corpus_queries, corpus_answers)
    ]

    # Initialize the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    wandb.init(
    project="miniLM-finetune",
    name="qa-semantic-search",
    config={
        "epochs": 30,
        "batch_size": 32,
        "model": "all-MiniLM-L6-v2",
        "loss": "MultipleNegativesRankingLoss"
    }
    )
    #fine-tune the model
    model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=30,
    warmup_steps=int(0.1 * len(train_dataloader)),
    show_progress_bar=True
    )

    model.save("./fine-tuned-miniLM-model")
    wandb.finish()
    
if __name__ == "__main__":
    main()
