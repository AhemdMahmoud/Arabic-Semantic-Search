# Arabic Semantic Search Implementation

This repository contains an implementation of semantic search for Arabic text using vector databases. The project compares the performance of two popular vector databases: ChromaDB and FAISS.

![image](https://github.com/user-attachments/assets/0ecd91d0-7eef-45db-8e1e-18422e4ede6e)


## Overview

Semantic search uses the meaning of text rather than keyword matching, enabling more accurate and context-aware searches. This implementation:

1. Loads Arabic question-answering and news datasets
2. Converts text to vector embeddings using multilingual models
3. Stores these embeddings in vector databases (ChromaDB and FAISS)
4. Performs similarity searches to retrieve relevant documents
5. Compares the performance of both databases in terms of speed and accuracy

## Dependencies

```
pip install chromadb datasets faiss-cpu sentence-transformers
```

## Datasets Used

- **Arabic Q&A Dataset**: `sadeem-ai/arabic-qna`
- **Arabic News Dataset**: `arbml/SANAD`

## Implementation Steps

### 1. Data Preparation

- Loaded the Arabic Q&A dataset and filtered for questions with answers
- Loaded the SANAD Arabic news dataset and filtered for articles with sufficient length
- Combined these datasets to create a corpus of documents
- Created metadata for each document (source, title)
- Assigned unique IDs to each document

### 2. Text Vectorization

- Used the SentenceTransformer model: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- Encoded all documents and questions into 512-dimensional vectors

### 3. Vector Database Setup

#### ChromaDB Implementation
```python
chroma_client = chromadb.PersistentClient(path="./chromadb-ar-docs")
collections = chroma_client.get_or_create_collection(
    name="Friska",
    metadata={"hnsw:space": "cosine"}
)

collections.add(
    documents=doc_texts,
    ids=doc_ids,
    metadatas=metadata,
    embeddings=encoded_docs
)
```

#### FAISS Implementation
```python
norm_encoded_docs = deepcopy(encoded_docs)
faiss.normalize_L2(norm_encoded_docs)

faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
faiss_index.add_with_ids(norm_encoded_docs, doc_ids)
```

### 4. Search Example

```python
question = "ما السبب في صغر الأسنان بالمقارنة مع حجم الفكين؟"
encoded_question = model.encode(question).reshape(1, -1)
faiss.normalize_L2(encoded_question)
results = faiss_index.search(encoded_question, 3)
```

### 5. Saving and Loading Models

```python
# Save FAISS index
with open("./faiss-ar-docs/index.pickle", "wb") as handle:
    pickle.dump(faiss_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save associated data
with open("./faiss-ar-docs/data.pickle", "wb") as handle:
    pickle.dump({
        "data": doc_texts,
        "docs_ids": doc_ids,
        "metadata": metadata
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## Performance Comparison

### Retrieval Speed

The code measures the time it takes for each database to process all queries in the dataset:

| Database | Before Adding More Data | After Adding More Data |
|----------|-------------------------|------------------------|
| ChromaDB | 17.18 seconds           | 24.21 seconds          |
| FAISS    | 3.61 seconds            | 7.05 seconds           |

FAISS consistently outperforms ChromaDB in retrieval speed, being approximately 3-4 times faster.

### Retrieval Accuracy

The accuracy metrics are divided into three categories:
- **Valid**: Exact matches (same document ID)
- **Similar**: Different document but same source
- **Invalid**: Different document and different source

#### Before Adding More Data:

| Database | Valid     | Similar   | Invalid   |
|----------|-----------|-----------|-----------|
| ChromaDB | 33.61%    | 23.78%    | 42.61%    |
| FAISS    | 33.89%    | 23.61%    | 42.51%    |

#### After Adding More Data:

| Database | Valid     | Similar   | Invalid   |
|----------|-----------|-----------|-----------|
| ChromaDB | 28.59%    | 19.62%    | 51.80%    |
| FAISS    | 28.14%    | 19.79%    | 52.07%    |

Both databases show comparable accuracy, with a slight degradation when adding more data (which is expected as the search space grows).

## Conclusion

- **Speed**: FAISS significantly outperforms ChromaDB for retrieval speed
- **Accuracy**: Both databases perform similarly in terms of retrieval accuracy
- **Use Case**: FAISS may be more suitable for applications requiring fast retrieval over large datasets, while ChromaDB offers comparable accuracy with a potentially simpler API

## Future Work

- Experiment with different embedding models
- Implement query expansion techniques
- Add support for hybrid search (semantic + keyword)
- Optimize vector database parameters for better performance
