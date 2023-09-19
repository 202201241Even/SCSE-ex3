import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import gutenberg

nltk.download('gutenberg')
nltk.download('vader_lexicon')

def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)
    sentiment_scores = [sia.polarity_scores(sentence)["compound"] for sentence in sentences]
    return sum(sentiment_scores) / len(sentiment_scores)

text = gutenberg.raw("melville-moby_dick.txt")
average_score = get_sentiment_score(text)

if average_score > 0.05:
    sentiment = "positive"
else:
    sentiment = "negative"

print("Average Sentiment Score:", average_score)
print("Overall Text Sentiment:", sentiment)
