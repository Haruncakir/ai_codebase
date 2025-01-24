import re
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def clean_text(text):
    text = text.lower()  # Convert text to lower case
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove email addresses
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove punctuation and special characters
    text = re.sub(r'\d', ' ', text)  # Remove digits
    text = re.sub(r'\s\s+', ' ', text)  # Remove extra spaces

    return text

print(clean_text('Check out the course at www.example.com/sample123'))

# Fetching the 20 Newsgroups Dataset
newsgroups_data = fetch_20newsgroups(subset='train')
nlp_df = pd.DataFrame(newsgroups_data.data, columns = ['text'])

# Applied the cleaning function to the text data
nlp_df['text'] = nlp_df['text'].apply(lambda x: clean_text(x))

# Checking the cleaned text
print(nlp_df.head())

test_texts = ['This is an EXAMPLE!', 'Another ex:ample123 with special characters $#@!', 'example@mail.com is an email address.']
for text in test_texts:
    print(f'Original: {text}')
    print(f'Cleaned: {clean_text(text)}')
    print('--')


