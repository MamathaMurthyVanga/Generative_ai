import pandas as pd
import re
import string
import nltk
from textblob import TextBlob
import emoji

# Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('popular')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer



# Chat acronym dictionary
chat_words = {
    "FYI": "For Your Information",
    "ASAP": "As Soon As Possible",
    "BRB": "Be Right Back",
    "BTW": "By The Way",
    "OMG": "Oh My God",
    "IMO": "In My Opinion",
    "LOL": "Laugh Out Loud",
    "TTYL": "Talk To You Later",
    "GTG": "Got To Go",
    "TTYT": "Talk To You Tomorrow",
    "IDK": "I Don't Know",
    "TMI": "Too Much Information",
    "IMHO": "In My Humble Opinion",
    "ICYMI": "In Case You Missed It",
    "AFAIK": "As Far As I Know",
    "FAQ": "Frequently Asked Questions",
    "TGIF": "Thank God It's Friday",
    "FYA": "For Your Action"
}
# Chat acronym conversion function
def chat_conversion(text):
    new_text = []
    for word in text.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)

# Spell correction function using TextBlob
def correct_spelling(text):
    corrected = TextBlob(text).correct()
    return corrected.string

# Remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Convert emojis to text
def convert_emoji_to_text(text):
    return emoji.demojize(text)

# Stemming
ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])


# Load dataset
file_path = "IMDB Dataset.csv"
# df = pd.read_csv(file_path)
df = pd.read_csv(file_path, nrows=100)


# Preview data
print("Dataset Loaded:")
print(df.head())

# Clean text function
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Replace chat acronyms
    text = chat_conversion(text)
    # Correct spelling
    text = correct_spelling(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Emoji to text (alternative: use remove_emoji instead)
    text = convert_emoji_to_text(text)
    # Tokenize
    tokens = wordpunct_tokenize(text)
    # Remove stopwords
    cleaned_tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Re-join and apply stemming
    stemmed_text = stem_words(" ".join(cleaned_tokens))
    # Rejoin tokens
    return " ".join(stemmed_text)


# testing spelling
# test_text = 'ceertain conditionas duriing seveal ggenerations aree moodified in the saame maner.'
# print(correct_spelling(test_text))

# print(stopwords.words('english'))

# print(remove_emoji("Loved it 😍🔥"))  # Output: Loved it 
# print(convert_emoji_to_text("Loved it 😍🔥"))  # Output: Loved it :smiling_face_with_heart_eyes::fire:


# sample = "walk walks walking walked"
# print("Stemmed Sample:", stem_words(sample))
# Output: walk walk walk walk


# Apply cleaning to dataset
df['cleaned_review'] = df['review'].apply(clean_text)



# Show cleaned data
print("\nCleaned Reviews Sample:")
print(df[['review', 'cleaned_review']].head())

# Save cleaned data (optional)
# df.to_csv("cleaned_imdb_dataset.csv", index=False)
# print("\nCleaned dataset saved to 'cleaned_imdb_dataset.csv'")
