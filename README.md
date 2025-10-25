
# PyTorch Sentiment Analysis on IMDB Dataset

A complete, step-by-step guide to building and training a sentiment analysis model on the IMDB movie review dataset. This project uses PyTorch and a Long Short-Term Memory (LSTM) recurrent neural network (RNN) to classify reviews as positive or negative.

This repository is primarily structured as a Jupyter Notebook (`IMDB_Sentiment_Analysis.ipynb`) that walks through the entire process from data loading to model inference.

## Project Overview

The goal of this project is to classify movie reviews as either "positive" or "negative." To achieve this, we will:

1.  **Load Data:** Use `pandas` to load the provided `imdb_tr.csv` (for training) and `imdb_te.csv` (for testing).
    
2.  **Preprocess Text:** Clean the raw review text by removing HTML tags, punctuation, and stopwords.
    
3.  **Build a Vocabulary:** Create a word-to-index mapping from the training data.
    
4.  **Create PyTorch Datasets:** Convert the text data into numerical sequences (tensors) and use PyTorch's `Dataset` and `DataLoader` for efficient batching.
    
5.  **Define the Model:** Build an LSTM-based neural network using `torch.nn`. The architecture consists of an Embedding layer, an LSTM layer, and a final Linear layer with a Sigmoid activation.
    
6.  **Train the Model:** Train the network on the training data, using a validation split to monitor for overfitting.
    
7.  **Evaluate the Model:** Test the trained model's performance on the unseen test dataset (`imdb_te.csv`) and report metrics like accuracy, precision, recall, and F1-score.
    
8.  **Run Inference:** Create a function to predict the sentiment of any new, custom review.
    

## File Structure

-   `IMDB_Sentiment_Analysis.ipynb`: The main Jupyter Notebook containing all the code and explanations.
    
-   `imdb_tr.csv`: The training and validation dataset (35,000 reviews).
    
-   `test/imdb_te.csv`: The testing dataset (15,000 reviews).
    
-   `README.md`: This file.
    

## Getting Started

### Prerequisites

You will need Python 3.7+ and the following libraries. You can install them using `pip`:

```
pip install torch pandas numpy nltk scikit-learn tqdm jupyterlab

```

### Installation

1.  Clone this repository:
    
    ```
    git clone https://github.com/Rahmat-ML/pytorch-imdb-sentiment-lstm
    cd pytorch-imdb-sentiment-lstm
    
    ```
    
2.  Ensure you have the data files (`imdb_tr.csv` and `test/imdb_te.csv`) in the correct locations.
    

### Usage

The entire project is contained within the Jupyter Notebook.

1.  Start Jupyter Lab:
    
    ```
    jupyter lab
    
    ```
    
2.  Open `IMDB_Sentiment_Analysis.ipynb`.
    
3.  Run the cells sequentially from top to bottom. The notebook is commented to explain each step of the process.
    

## Model Architecture

The core of our model is an LSTM network, which is well-suited for sequence data like text.

1.  **Embedding Layer:** Converts integer-encoded words (from our vocabulary) into dense vectors (embeddings) of a fixed size.
    
2.  **LSTM Layer:** Processes the sequence of embeddings, capturing contextual information and long-range dependencies in the text.
    
3.  **Dropout Layer:** (Optional but recommended) Helps prevent overfitting during training.
    
4.  **Linear Layer:** A fully-connected layer that maps the final hidden state of the LSTM to a single output.
    
5.  **Sigmoid Activation:** Squashes the output to a value between 0 and 1, representing the probability of the review being "positive."
