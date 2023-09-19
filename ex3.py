import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('gutenberg')
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None

# Step 1: Read the Moby Dick file from the Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Step 2: Tokenization
tokens = word_tokenize(moby_dick)

# Step 3: Stop-words filtering and punctuation removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]

# Step 4: Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# Step 5: Filter out None values from WordNet POS mapping
pos_tags = [(word, tag) for word, tag in pos_tags if get_wordnet_pos(tag) is not None]

# Step 6: POS frequency
pos_freq = FreqDist(tag for (word, tag) in pos_tags)
most_common_pos = pos_freq.most_common(5)

print("POS Frequency")
print("-----------------------------")
for pos, freq in most_common_pos:
    print(f"{pos}: {freq}")

# Step 7: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for (word, pos) in pos_tags[:20]]



# Step 8: Plotting frequency distribution
pos_tags_list = [tag for (word, tag) in pos_tags]
pos_dist = FreqDist(pos_tags_list)

plt.figure(figsize=(10, 5))
pos_dist.plot(30, cumulative=False)
plt.show()
