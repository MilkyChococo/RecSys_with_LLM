import json

class ProductQADataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load_jsonl()
        print(f"Loaded {len(self.data)} Q&A pairs from {self.filepath}")

    def _load_jsonl(self):
        data = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def get_all(self):
        return self.data

    def get_queries(self):
        return [item["query"] for item in self.data]

    def get_answers(self):
        return [item["answer"] for item in self.data]

    def get_pair(self, index):
        return self.data[index]["query"], self.data[index]["answer"]

    def __len__(self):
        return len(self.data)
