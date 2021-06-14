# text-mining
Classes to explore various text embeddings and learning models

## Vectorizer:
- custom / from-scratch TF-IDF matrix encoding
- custom weights to improve matching

## Text Classifier:
- logistic regression on TFIDF with custom tokenization
- comparison of training/testing on two different datasets -- seeing how well it generalizes
- testing model on custom vocabulary created from intersection of the two datasets

## Continuous Embeddings
- comparison of logistic regression on basic TFIDF embeddings to doc2vec embeddings
- seeing how well model generalizes over different test-train splits
