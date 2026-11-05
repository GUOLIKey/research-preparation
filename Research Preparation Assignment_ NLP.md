# Research Preparation Assignment  
## NLP, Representation Learning, Clustering, Topic Modeling, and Deep Generative Models

### 1. Purpose

This assignment is designed to prepare you for future research projects in natural language processing (NLP).

The goal is to learn the basic concepts while writing code and running experiments.

This is not a reading-only assignment. For each topic, you should write code, run experiments, inspect the results, and explain what you observe.

**Duration:** 5 weeks  
**Expected workload:** 10–15 hours per week  
**Total workload:** approximately 50–75 hours

At the end of the assignment, you should be able to:

Understand a standard machine learning workflow.

Use Python, NumPy, pandas, scikit-learn, and PyTorch for research experiments.

Understand tensors, neural networks, loss functions, gradient descent, backpropagation, and model training.

Understand common text representations such as bag-of-words, TF-IDF, and dense text embeddings.

Apply clustering methods such as K-means.

Understand the basic idea of representation learning.

Implement a simple autoencoder (AE).

Understand the basic structure of a variational autoencoder (VAE).

Understand the basic idea of topic modeling.

Run experiments comparing different representations.

Organize code and experimental results in a research-style project.

---

# 2. General Requirements

Create one GitHub repository for this assignment.

A suggested structure is:

```text
research-preparation/
│
├── week1_python/
├── week2_machine_learning/
├── week3_pytorch/
├── week4_nlp/
├── final_project/
│
├── README.md
└── requirements.txt
```

Jupyter notebooks are acceptable during the learning stage. For the final project, try to separate data processing, model definition, training, evaluation, and visualization into clear functions or Python files.

For every experiment, record:

```text
Experiment name
Model
Input representation
Main hyperparameters
Random seed
Training loss
Evaluation results
Short observations
```

Do not simply copy code from tutorials.

You should be able to explain what every main part of your code is doing.

---

# Week 1: Python for Machine Learning

**Time: 10–12 hours**

## Learning Goals

The goal of Week 1 is to become comfortable using Python for data processing and numerical computing.

You should review the following Python concepts:

Variables and data types

Lists, tuples, dictionaries, and sets

Conditions and loops

Functions

List comprehensions

Classes and basic object-oriented programming

File reading and writing

Python modules and packages

Exception handling

Basic debugging

You should then become familiar with NumPy and pandas.

For NumPy, focus on:

Arrays

Shape and dimensions

Indexing and slicing

Reshaping

Matrix operations

Mean, variance, sum, and maximum

Broadcasting

Random number generation

Vectorized operations

For pandas, focus on:

DataFrame

Loading CSV files

Selecting rows and columns

Filtering

Missing values

Basic statistics

Grouping

Sorting

---

## Hands-on Task 1: Python and NumPy

Create a notebook named:

```text
week1_python_numpy.ipynb
```

Generate a random matrix with:

```text
1000 samples
20 features
```

Perform the following tasks without using unnecessary Python loops:

Calculate the mean of every feature.

Calculate the standard deviation of every feature.

Normalize every feature.

Find the five samples with the largest Euclidean norm.

Calculate the pairwise cosine similarity between the first 20 samples.

Implement your own function:

```python
cosine_similarity(x, y)
```

Do not use the scikit-learn cosine similarity function for this part.

Verify your implementation using an existing library.

---

## Hands-on Task 2: Small Data Analysis

Choose a small CSV dataset from Kaggle or another open dataset.

Use pandas to:

Load the data.

Check the number of samples and features.

Check missing values.

Calculate summary statistics.

Select several columns.

Filter samples according to conditions.

Create at least three simple plots.

Write five short observations about the dataset.

The purpose is not to perform machine learning yet. The goal is to become comfortable manipulating data.

---

## Recommended Resources

Kaggle provides a short interactive Python course covering functions, conditions, lists, loops, strings, dictionaries, and external libraries.

[Kaggle Python Course](https://www.kaggle.com/learn/python?utm_source=chatgpt.com)

[Official Python Tutorial](https://docs.python.org/3/tutorial/?utm_source=chatgpt.com)

[NumPy User Guide](https://numpy.org/doc/stable/user/?utm_source=chatgpt.com)

[pandas Getting Started](https://pandas.pydata.org/docs/getting_started/?utm_source=chatgpt.com)

### Week 1 Deliverables

Submit:

```text
week1_python_numpy.ipynb
week1_data_analysis.ipynb
```

Your notebooks should contain both code and short explanations.

---

# Week 2: Machine Learning Foundations and Text Representation

**Time: 10–15 hours**

## Learning Goals

You should understand the following concepts before moving to deep learning:

Samples and features

Labels

Supervised learning

Unsupervised learning

Training set

Validation set

Test set

Feature representation

Model parameters

Hyperparameters

Loss function

Optimization

Overfitting

Underfitting

Generalization

Classification

Clustering

Dimensionality reduction

You do not need to study every mathematical derivation at this stage, but you should understand what each concept means and why it is used.

---

# Part A: A Simple Classification Experiment

Use the Iris dataset, Wine dataset, or another small dataset available in scikit-learn.

Split your data into training and test sets.

Train:

```text
Logistic Regression
k-Nearest Neighbors
Decision Tree
```

Report:

```text
Training accuracy
Test accuracy
Confusion matrix
```

Then answer:

Why should we not evaluate a model only using training data?

What is overfitting?

What happens if a decision tree becomes too deep?

Why do we need a test set?

---

# Part B: Introduction to NLP Data

Now start working with text.

Use the **20 Newsgroups dataset** available in scikit-learn.

The full dataset contains around 18,000 documents from 20 categories. The official scikit-learn clustering example also provides a smaller four-category version with roughly 3,400 documents, which is a good size for this assignment.

For the first experiment, use only these four categories:

```text
comp.graphics
sci.space
alt.atheism
talk.religion.misc
```

Load the dataset and remove:

```text
headers
footers
quotes
```

---

## Text Preprocessing

Learn the basic meaning of:

Token

Vocabulary

Stop words

Bag-of-Words

Term frequency

Inverse document frequency

TF-IDF

Sparse matrix

Then use:

```python
TfidfVectorizer
```

to convert the text into numerical vectors.

Inspect:

```text
Number of documents
Vocabulary size
TF-IDF matrix shape
Several example words
Several example TF-IDF vectors
```

Do not treat the vectorizer as a black box. Explain what the input and output represent.

---

# Part C: Your First Text Clustering Experiment

Apply K-means to the TF-IDF vectors.

Try:

```text
K = 2
K = 4
K = 6
K = 8
```

Since the dataset provides true category labels, you may use these labels for evaluation after clustering.

Report:

```text
Adjusted Rand Index (ARI)
Normalized Mutual Information (NMI)
Silhouette score
```

The labels must not be used to train K-means.

For each cluster, report the 10 words with the largest average TF-IDF values.

Try to interpret what each cluster represents.

The official scikit-learn example for document clustering is a useful reference.

[Scikit-learn Text Clustering Example](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html?utm_source=chatgpt.com)

### Week 2 Deliverables

Submit:

```text
week2_basic_ml.ipynb
week2_text_clustering.ipynb
```

Also write approximately one page explaining:

```text
Bag-of-Words
TF-IDF
K-means
ARI
NMI
```

Use your own words.

---

# Week 3: PyTorch and Neural Networks

**Time: 12–15 hours**

## Learning Goals

This week focuses on PyTorch and neural network training.

You should understand:

Tensor

Tensor shape

CPU and GPU

Dataset

DataLoader

Batch

Epoch

Neural network layer

Linear layer

Activation function

ReLU

Forward pass

Loss function

Gradient

Backpropagation

Optimizer

Learning rate

Model training mode

Model evaluation mode

---

## PyTorch Learning Tasks

Complete the main sections of the official PyTorch beginner tutorial:

Tensors

Datasets and DataLoaders

Building a neural network

Automatic differentiation

Optimization

Saving and loading models

The official PyTorch beginner material walks through these parts as one machine learning workflow.

[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html?utm_source=chatgpt.com)

---

# Hands-on Task 1: Build a Neural Network Yourself

Use MNIST or FashionMNIST.

Create a neural network such as:

```text
Input
↓
Linear
↓
ReLU
↓
Linear
↓
ReLU
↓
Linear
↓
10 classes
```

Do not copy a complete implementation without modification.

Write your own:

```text
model class
training loop
evaluation loop
```

Plot:

```text
Training loss versus epoch
Training accuracy versus epoch
Test accuracy versus epoch
```

Run at least three experiments with different learning rates.

For example:

```text
0.1
0.01
0.001
```

Explain how the learning rate changes training.

---

# Hands-on Task 2: Autoencoder

Now move from classification to representation learning.

Build a simple AE using MNIST.

A possible structure is:

```text
784
↓
256
↓
64
↓
16
↓
64
↓
256
↓
784
```

The 16-dimensional vector is the **latent representation**.

Train the AE to reconstruct the input image.

After training:

Display original images.

Display reconstructed images.

Extract the 16-dimensional latent representations.

Apply K-means to the latent representations.

Use K = 10.

Calculate ARI and NMI using the MNIST labels.

Compare clustering using:

```text
Raw pixels
AE latent representations
```

Answer:

Does representation learning improve clustering?

Why might a compressed representation make clustering easier?

---

### Week 3 Deliverables

Submit:

```text
week3_neural_network.ipynb
week3_autoencoder.ipynb
```

Your AE notebook should clearly show:

```text
Encoder
Latent representation
Decoder
Reconstruction loss
Training process
Latent-space clustering
```

---

# Week 4: NLP Representation Learning and Topic Modeling

**Time: 10–15 hours**

Now return to the 20 Newsgroups dataset.

This week connects classical NLP with modern representation learning.

---

# Part A: Three Types of Text Representation

You have already used:

```text
TF-IDF
```

Now compare it with a dense text embedding.

Use a small pretrained Sentence Transformer model to convert every document into a dense vector.

Sentence Transformers provides fixed-size embeddings that can be used directly for tasks such as similarity and clustering.

[Sentence Transformers Quickstart](https://sbert.net/docs/quickstart.html?utm_source=chatgpt.com)

Run K-means using:

```text
TF-IDF
Sentence Transformer embeddings
```

Use:

```text
K = 4
```

Compare:

```text
ARI
NMI
Silhouette score
```

Explain why different text representations can produce different clustering results.

---

# Part B: Topic Modeling

Learn the basic idea of topic modeling.

At this stage, focus on:

Document

Word

Vocabulary

Topic

Document-topic distribution

Topic-word distribution

Latent Dirichlet Allocation (LDA)

You do not need to derive the full LDA algorithm.

Use scikit-learn to train an LDA model on the same 20 Newsgroups subset.

Try:

```text
5 topics
10 topics
15 topics
```

For each topic, print the top 10 words.

Inspect whether the topics have clear meanings.

The official scikit-learn topic-modeling example demonstrates both LDA and non-negative matrix factorization on text data.

[Scikit-learn Topic Modeling Example](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html?utm_source=chatgpt.com)

Answer:

What is the difference between clustering documents and discovering topics?

Can one document contain several topics?

Why are topic models useful when labels are unavailable?

---

# Part C: Introduction to Variational Autoencoders

Before applying a VAE to text, first understand the basic structure.

An AE learns:

```text
x → z → reconstructed x
```

A VAE instead learns parameters of a distribution, commonly:

```text
x
↓
Encoder
↓
μ, log σ²
↓
Sampling z
↓
Decoder
↓
reconstructed x
```

You should understand the meaning of:

Latent variable

Prior distribution

Posterior distribution

Mean

Variance

Sampling

Reparameterization

Reconstruction loss

Kullback-Leibler (KL) divergence

You are not required to derive the evidence lower bound (ELBO) in full during this assignment.

Run the official PyTorch VAE example on MNIST.

The PyTorch examples repository contains a short VAE implementation with an encoder, Gaussian latent variables, reparameterization, reconstruction loss, and KL divergence.

[Official PyTorch VAE Example](https://github.com/pytorch/examples/tree/main/vae?utm_source=chatgpt.com)

Modify at least two parameters, such as:

```text
Latent dimension
Learning rate
Number of epochs
```

Observe how the reconstruction or generated samples change.

### Week 4 Deliverables

Submit:

```text
week4_text_embeddings.ipynb
week4_topic_modeling.ipynb
week4_vae.ipynb
```

You should also be able to explain the difference between:

```text
AE
VAE
LDA
K-means
```

---

# Week 5: Final Mini-Project

## Project Title

**Learning Latent Text Representations for Document Clustering and Topic Discovery**

**Time: 12–15 hours**

This project connects the main topics from the previous four weeks.

You will use the same four-category subset of the **20 Newsgroups dataset**.

The research question is:

> How does the choice of text representation affect document clustering and topic discovery?

---

# Stage 1: Classical Baseline

Create TF-IDF features.

Run:

```text
TF-IDF
↓
K-means
```

Set:

```text
K = 4
```

Record:

```text
ARI
NMI
Silhouette score
```

This is your baseline.

---

# Stage 2: Pretrained Text Representation

Generate dense document embeddings using Sentence Transformers.

Run:

```text
Sentence embeddings
↓
K-means
```

Again calculate:

```text
ARI
NMI
Silhouette score
```

Compare the results with TF-IDF.

---

# Stage 3: Autoencoder Representation Learning

Use TF-IDF features as the input to a PyTorch AE.

To reduce computational cost, limit the TF-IDF vocabulary to approximately:

```text
2000–5000 features
```

A simple model could be:

```text
Input
↓
512
↓
128
↓
32-dimensional latent representation
↓
128
↓
512
↓
Reconstruction
```

Train the AE.

Extract the 32-dimensional latent representation.

Then run:

```text
AE latent representation
↓
K-means
```

Calculate:

```text
ARI
NMI
Silhouette score
```

Compare this representation with:

```text
TF-IDF
Sentence embeddings
```

---

# Stage 4: Simple Text VAE

This stage is your first small deep generative model experiment.

Create a binary bag-of-words representation using approximately:

```text
2000 vocabulary words
```

Each document becomes a vector:

```text
x ∈ {0,1}^V
```

where `V` is the vocabulary size.

Adapt the PyTorch VAE structure that you used in Week 4.

A simple encoder can be:

```text
2000
↓
512
↓
128
↓
μ and log σ²
```

Use a latent dimension such as:

```text
20 or 32
```

The decoder reconstructs the document vector.

The training objective contains:

```text
Reconstruction loss
+
KL divergence
```

After training, use the posterior mean `μ` as the document representation.

Run:

```text
VAE latent representation
↓
K-means
```

Calculate:

```text
ARI
NMI
Silhouette score
```

Do not spend too much time trying to obtain very high clustering accuracy. The main goal is to understand the model and experimental process.

---

# Stage 5: Topic Modeling Baseline

Train LDA using the same documents.

Try:

```text
10 topics
```

Print the top 10 words for every topic.

Select three topics and manually explain what you think they represent.

Compare the LDA topics with the document clusters.

---

# Stage 6: Visualization

Use PCA or t-SNE to visualize the document representations in two dimensions.

Create visualizations for at least:

```text
TF-IDF or reduced TF-IDF
Sentence embeddings
AE latent representation
VAE latent representation
```

Color the points using the true category labels only for visualization.

Do not use these labels during unsupervised training.

---

# Stage 7: Final Comparison

Create a results table similar to:

| Representation | Dimension | Clustering Method | ARI | NMI | Silhouette |
|---|---:|---|---:|---:|---:|
| TF-IDF | 2000 | K-means | | | |
| Sentence Embedding | model dependent | K-means | | | |
| AE | 32 | K-means | | | |
| VAE | 32 | K-means | | | |

Then discuss:

Which representation gives the best clustering results?

Which representation gives the clearest visualization?

Does the AE learn a useful latent representation?

Does the VAE representation behave differently from the AE representation?

Why might pretrained semantic embeddings perform differently from TF-IDF?

What information may be lost when compressing documents into a small latent space?

How are clustering and topic modeling related?

How are they different?

What would you try next if you had another month?

---

# Final Deliverables

Your final GitHub repository should contain working code and a clear `README.md`.

The final project folder should contain:

```text
data_processing.py or equivalent notebook section

tfidf_clustering.py

autoencoder.py

vae.py

topic_modeling.py

evaluation.py

visualization.py

README.md
```

The exact file structure may differ, but your code should be easy to read and reproduce.

You should submit a short report of approximately **4–6 pages** containing:

```text
1. Problem and motivation
2. Dataset
3. Methods
4. Experimental setup
5. Results
6. Visualization
7. Discussion
8. What you learned
```

Also prepare a **10-minute presentation** explaining your project.

---

# Final Oral Check

After completing the assignment, you should be able to answer the following questions without reading from your notebook.

What is a feature?

What is a representation?

What is TF-IDF?

What does K-means optimize?

What is a latent representation?

Why do we need nonlinear activation functions?

What happens during backpropagation?

What is the role of the learning rate?

What is the difference between an AE and a VAE?

Why does a VAE contain a KL-divergence term?

What does the reparameterization step do?

What is topic modeling?

What is the difference between topic modeling and clustering?

Why can two different representations lead to very different clustering results?

Why should labels not be used when training an unsupervised model?

What are ARI and NMI measuring?

---

# Recommended Reference Material

## Python and Data Processing

[Kaggle Python Course](https://www.kaggle.com/learn/python?utm_source=chatgpt.com)

[Python Official Tutorial](https://docs.python.org/3/tutorial/?utm_source=chatgpt.com)

[NumPy User Guide](https://numpy.org/doc/stable/user/?utm_source=chatgpt.com)

[pandas Getting Started](https://pandas.pydata.org/docs/getting_started/?utm_source=chatgpt.com)

## Machine Learning

[Kaggle Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning?utm_source=chatgpt.com)

[Scikit-learn Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html?utm_source=chatgpt.com)

## PyTorch

[PyTorch Beginner Tutorials](https://docs.pytorch.org/tutorials/beginner/basics/intro.html?utm_source=chatgpt.com)

[PyTorch Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html?utm_source=chatgpt.com)

[Official PyTorch Examples Repository](https://github.com/pytorch/examples?utm_source=chatgpt.com)

[PyTorch VAE Example](https://github.com/pytorch/examples/tree/main/vae?utm_source=chatgpt.com)

## NLP and Representation Learning

[Hugging Face NLP and LLM Course](https://huggingface.co/docs/course/chapter1/1?utm_source=chatgpt.com)

[Sentence Transformers Quickstart](https://sbert.net/docs/quickstart.html?utm_source=chatgpt.com)

## Clustering and Topic Modeling

[Scikit-learn Document Clustering Example](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html?utm_source=chatgpt.com)

[Scikit-learn LDA and NMF Topic Modeling Example](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html?utm_source=chatgpt.com)

---

# Important Advice

The purpose of these five weeks is not to become an expert in every method.

Focus on understanding the full research workflow:

```text
Problem
↓
Data
↓
Representation
↓
Model
↓
Training
↓
Evaluation
↓
Analysis
↓
New questions
```

When something does not work, do not immediately replace the code with code generated by another person or an AI system.

First identify:

```text
What did you expect?
What actually happened?
Where may the problem be?
What experiment can test your explanation?
```

This habit is more important for research than obtaining a high score on one experiment.

After completing this assignment, we will begin working on research problems related to **NLP, representation learning, deep generative models, and topic modeling**.