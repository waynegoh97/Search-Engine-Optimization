# Data Preprocessing.ipynb
### Imports
- import pandas as pd
- import numpy as np
- import langdetect
- from sklearn.model_selection import train_test_split
- import spacy
- from textblob import TextBlob
- from textblob.classifiers import NaiveBayesClassifier
- from sklearn.metrics import cohen_kappa_score
- from nltk.corpus import stopwords 
- from textblob import Word
- *Missing library can be downloaded in the .ipynb file by typing !pip install libname

### Steps
1. Run the code blocks in sequence. Below are the descriptions of code running sequence.
2. Non-English reviews will be removed (but some lines would cause an error and would have to be removed manually)
3. Split the review data into 90% train and 10% test data
4. Train dataset are labelled by ratings (1-2: Negative, 3: Neutral, 4-5: Positive)
5. Test dataset are labelled manually (calculate cohen kappa score)
6. Stopwords removal, lowercase applied, removal of other punctuations, lemmatization applied

# RoBERTa.ipynb
### Imports
- import pandas as pd
- from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
- import torch
- from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
- from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
- *Missing library can be downloaded in the .ipynb file by typing !pip install libname

### Steps
1. Run the code blocks in sequence. Below are the descriptions of code running sequence.
2. Read the train_data.csv and test_data.csv using the pd.read_csv
3. Sentiments are mapped to numerical labels for training of classifier
4. Reviews are splitted to a maximum of 512 length as the maximum token length is 512
5. Save this newly processed data for RoBERTa training
6. Prepare the data by feeding the reviews to the RobertaTokenizer
7. Read the data using TabularDataset
8. Data will be passed using the Iterator library
9. Customize RoBERTa classifier by changing the output layer to 3 classes
10. Pretrain the classifier using feature extraction (which freezes the layers) and save the model
11. Use the pretrain model to further train the model using fine-tuning (which unfreezes the layers)
12. Use the sklearn.metrics library to evaluate the predicted results with the test dataset

# ABSA.ipynb
### Imports
- import pandas as pd
- import numpy as np
- from wordcloud import WordCloud, STOPWORDS
- from sklearn.feature_extraction.text import CountVectorizer
- import spacy
- import gensim
- import gensim.corpora as corpora
- from gensim.utils import simple_preprocess
- from spacy.lang.en import English
- import lda
- *Missing library can be downloaded in the .ipynb file by typing !pip install libname

### Steps
1. Run the code blocks in sequence. Below are the descriptions of code running sequence.
2. Using the train_data.csv, create a word cloud
3. Read the sentence_split_train.csv for topic analysis using LDA
4. Vectorize the review sentences using count vectorizer
5. Perform LDA to discover 2 topics, displaying top 10 words that describes the topics
6. The review sentences can then be grouped by similarity of the discovered topics


# DistilBERT.ipynb
### Imports
```bash
#!pip install -q transformers
#!pip install torch
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Input, Dense
import pandas as pd
import numpy as np
import keras
```

### Steps
1. Run the code blocks in sequence. Below are the descriptions of code running sequence.
2. Using the train_data.csv
3. Sentiments are mapped to numerical labels for training of classifier
4. Reviews are splitted to a 128 words for our benchmark model 
5. Save this newly processed data for DistilBERT training
6. Prepare the data by feeding the reviews to the RobertaTokenizer
9. Customize DistilBERT classifier by changing the output layer to 3 classes
10. Pretrain the classifier using feature extraction (which freezes the layers) and save the model
11. Use the pretrain model to further train the model using fine-tuning (which unfreezes the layers)
12. Use the sklearn.metrics library to evaluate the predicted results with the test dataset



# TFIDFClassifier.py (MAIN CLASSIFIER FOR APPLICATION)
### Start
To start, install all requirements using the following in powershell:
```bash
pip install -r requirements.txt
```

For the very first time, you are required to download ntlk packages located in line 20
```bash
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Type the command into powershell or cmd at the directory
```bash
#Configure the EXAMPLE section located at line 630
python3 TFIDFClassifier.py
```
You can choose to call our classifier function as well using import
```bash
# Ensure it is in the same directory
import TFIDFClassifier.py

Results = classifierByTFIDF(["Input your list"])
```

### Steps
1. Run the code blocks in sequence. Below are the descriptions of code running sequence.
2. Using the train_data_preprocessed.csv, remove the excess positive reviews to balance data.
3. train_data.csv has been processed with stopwords, lemmatization and lower case. it is commented away in the function.
4. Vectorize the review sentences using TFIDF vectorizer
5. Perform SMOTE to oversample neutral reviews.
6. Load pre-saved SVC model from pickle.
7. analyse and return a list of results.

