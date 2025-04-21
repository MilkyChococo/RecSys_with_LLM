import faiss
from Data.preprocess_query import QueryCleaner

class SemanticSearchQA:
    def __init__(self, model, cleaner: QueryCleaner, corpus_data: list):
        self.model = model
        self.cleaner = cleaner
        self.corpus_queries = [cleaner.clean(item["query"]) for item in corpus_data]
        self.corpus_answers = [item["answer"] for item in corpus_data]
        self.index = self._build_index(self.corpus_queries)

    def _build_index(self, queries):
        query_embs = self.model.encode(queries, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embs)
        index = faiss.IndexFlatIP(query_embs.shape[1])
        index.add(query_embs)
        return index

    def _format_answer(self, products):
        if isinstance(products, str):
            return products

        lines = ["Here are some products we recommend for you:\n"]
        for idx, prod in enumerate(products, 1):
            name = prod.get("name", "Unnamed")
            brand = prod.get("brand", "Unknown")
            price = prod.get("price", "N/A")
            discount = prod.get("discount", "N/A")

            lines.append(
                f"{idx}. **{name}**\n"
                f"   - Brand: {brand}\n"
                f"   - Price: ${price}\n"
                f"   - Discount: ${discount}\n"
            )
        return "\n".join(lines)

    def search(self, user_query: str, threshold=0.5) -> str:
        query = self.cleaner.clean(user_query)
        q_emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)

        D, I = self.index.search(q_emb, k=1)
        top_idx = I[0][0]
        score = D[0][0]

        if score < threshold or top_idx == -1 or user_query == "":
            return "Sorry, we couldn’t find a relevant answer for your question."

        matched_answer = self.corpus_answers[top_idx]
        if isinstance(matched_answer, list):
            return self._format_answer(matched_answer)
        return matched_answer