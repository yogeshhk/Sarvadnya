# Datasets for In-depth Analysis of Graph-based RAG

## Dataset information

We use the following 11 datasets:

|Dataset | \# of Tokens | \# of Questions | \# of Chunks | QA Type|
| --- |--- |--- |--- | :---: | 
|MultihopQA | 1,434,889 | 2,556 | 609  | Specific QA |
|Quality | 1,522,566 | 4,609 |265 | Specific QA |
|PopQA  | 2,630,554 | 1,172  | 33,595 | Specific QA|
|MusiqueQA | 3,280,174 | 3,000 | 29,898 | Specific QA | 
|HotpotQA   |8,495,056 | 3,702 |66,581 | Specific QA|
|ALCE | 13,490,670 | 948 | 89,562| Specific QA |
|Mix | 611,602 | 125 | 61| Abstract QA  |
|MultihopSum | 1,434,889 | 125 | 609| Abstract QA  |
|Agriculture |1,949,584 | 125 |12 | Abstract QA  |
|CS | 2,047,923| 125 | 10| Abstract QA |
|Legal |4,774,255 | 125 |94 | Abstract QA   |

## Data format

Corpus

```json
{
"title": "FIRST TITLE",
"context": "FIRST TEXT",
"id": 0
}
{
"title": "SECOND TITLE",
"context": "SECOND TEXT",
"id": 1
}
```

Question

```json
{
"question": "QUESTION 1",
"options": "DICT-style options for multiple-choice questions (Optional)",
"answer": "ANSWER",
"answer_idx":"Answer options for multiple-choice questions (Optional)",
"id": 0
}
```
