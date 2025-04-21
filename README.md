# Product Recommendation System Using LLM 

**Product Recommendation System Using LLM** is a Python-based recommendation engine that leverages advanced Machine Learning and Large Language Model (LLM) techniques to deliver accurate and context-aware product suggestions. Its main goal is to understand and respond to customer requests with high precision, while gracefully handling even invalid or off-topic questions. 

This project is designed to be both powerful and user-friendly, making it easy to integrate intelligent recommendations into your applications. Users can customize the system with their own data by providing a .jsonl dataset of product information and sample queries. This flexible data-driven approach means you can update the product catalog at any time without needing to fully retrain or fine-tune the model. The result is a robust recommendation system that stays up-to-date and efficient, tailored to your specific products and queries.

## ⚙️ FrameWork

- **Python**
- **Hugging Face Transformers** – for LLM inference
- **FAISS** – for fast vector-based similarity search
- **Hugging Face Model** – for fine-tuning or using pre-trained models
- **JSONL format** – for simple, extensible dataset customization
  <pre><code>{"qid":1,"query":"I want a clothing item that is suitable for a formal event.","ansid":1,"answer":"We have some products fixed for you:1. Pure Pineapple from Calvin Klein with the price of $629.99 and a discount of $495.2. Guangzhou sweater from Diesel with the price of $35 and a discount of $13.3. Guangzhou sweater from Polo with the price of $35 and a discount of $34.4. Converse Shoes from Calvin Klein with the price of $35 and a discount of $34.5. 2 Layer Windbreaker from Calvin Klein with the price of $44 and a discount of $35."}
  {"qid":2,"query":"I want accessories that is suitable for a formal event.","ansid":2,"answer":"We have some products fixed for you:1. Men's Painted Hat from Calvin Klein with the price of $44 and a discount of $35.2. Microfiber Wool Scarf sweater from Tommy Hilfiger with the price of $64 and a discount of $35."}
  {"qid":3,"query":"I want a handbag that is suitable for a casual event.","ansid":3,"answer":"We have some products fixed for you:1. Pure Pineapple from Calvin Klein with the price of $64 and a discount of $35."}</code></pre>
## 🔑 Key Strengths

- **Robust Query Handling**: Understands a wide range of user queries (even noisy or irrelevant ones) and responds gracefully without breaking.  
- **Easy Maintenance**: New products and queries can be added via the dataset file, **no full model fine-tuning or retraining needed** for updates.  
- **High Performance**: Delivers **faster** and **more reliable** responses compared to traditional text-generation models.  
- **Accuracy without Hallucinations**: Avoids the common *hallucinations* seen in generative models – the system stays **grounded in your data**, so recommendations remain **factual and relevant**.
