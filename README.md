# Fake News Detection with Text Analysis

## Overview

Project aims to identify fake news articles using text analysis techniques. It utilizes a neural network built with PyTorch to classify news articles as either real or fake based on their content.


## Dataset

The dataset used for this project consists of labeled news articles. The data is divided into two files:

- `True.csv`: Contains real news articles.
- `Fake.csv`: Contains fake news articles.

Each file includes a column with the text of the articles.

## Requirements

To run this project, you need to have Python 3.x and the following libraries installed:

- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `torch`

You can install the required libraries using pip.
## Features
+ Text Preprocessing: Cleaning, tokenization, stop word removal, and lemmatization of news articles.
+ Feature Engineering: Representation of text data using TF-IDF (Term Frequency-Inverse Document Frequency).
+ Neural Network Model: A fully connected neural network built with PyTorch for binary classification of news articles.
+ Model Evaluation: Accuracy and classification report for model evaluation.
## Dataset
The dataset used in this project is the `Fake and Real News Dataset` available on Kaggle. It consists of two CSV files:

- True.csv: Contains real news articles.
- Fake.csv: Contains fake news articles.<br/>
Each article is labeled with 1 for real news and 0 for fake news.

## Usage
1. Prepare the dataset: Place the True.csv and Fake.csv files in the data/ directory.

2. Run the model training script

3. Evaluate the model: The script will print the accuracy and classification report after training. The trained model will be saved as` fake_news_detector.pth ` in the model/ directory.
## Model Architecture
The model is a fully connected neural network with the following architecture:

+ Input Layer: Size equal to the number of TF-IDF features (5000).
+ Hidden Layer 1: 512 neurons with ReLU activation and dropout.
+ Hidden Layer 2: 256 neurons with ReLU activation and dropout.
+ Hidden Layer 3: 128 neurons with ReLU activation and dropout.
+ Output Layer: 2 neurons (for binary classification).
## Results
+ Accuracy: Achieved an accuracy of approximately 0.9910 on the test dataset.
+ Classification Report: Detailed precision, recall, and F1-score for each class (Real and Fake).

## License
This project is licensed under the MIT License. See the [LICENCE](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for more details.
