import pandas as pd
import re
from nltk.corpus import stopwords
from pickle import dump, load
import pdb
import nltk
nltk.download('stopwords')


reviews = pd.read_csv(
        "/media/data1/summarizer/tutorial/amazon_data/Reviews.csv")
print(reviews.shape)
print(reviews.head())
print(reviews.isnull().sum())


reviews = reviews.dropna()
reviews = reviews.drop(['Id', 'ProductId', 'UserId', 'ProfileName',
                        'HelpfulnessNumerator', 'HelpfulnessDenominator',
                        'Score', 'Time'], 1)
reviews = reviews.reset_index(drop=True)

print(reviews.head())
for i in range(5):
    print("Review {}".format(i+1))
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()


"""
Number of samples: 568411
Number of unique input tokens: 84
Number of unique output tokens: 48
Max sequence length for inputs: 15074
Max sequence length for outputs: 5
Due to memeory issue when creating input np.array of shape (568411, 15074, 84),
let's shrink the dataset by removing long texts.
"""
index2remove = []
text_length_threshold = 200
print("Getting the index of rows to remove "
      "with text length threshold {}...".format(text_length_threshold))
for i in range(len(reviews)):
    if len(reviews.Text[i]) > text_length_threshold:
        index2remove.append(i)
print("Going to remove {} rows...".format(len(index2remove)))
reviews = reviews.drop(index2remove)
# reviews = reviews.drop(reviews[len(reviews.Text) > text_length_threshold].index)
print("Long text removal is done!")
print(len(reviews))

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def clean_text(text, remove_stopwords=True):
    '''Remove unwanted characters, stopwords, and format the text
    to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]
        text = " ".join(text)

    return text


# Clean the summaries and texts
clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")

stories = list()
for i, text in enumerate(clean_texts):
        stories.append({'story': text, 'highlights': clean_summaries[i]})

# save to file
dump(stories,
     open('/media/data1/summarizer/tutorial/amazon_data/review_dataset.pkl',
          'wb'))
pdb.set_trace()
