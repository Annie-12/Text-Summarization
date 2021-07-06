from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def text_preprocessing(sentences: list) -> list:
    """
    Pre processing text to remove unnecessary words.
    """
    print('Preprocessing text')

    stop_words = set(stopwords.words('english'))

    clean_words = []

    for sent in sentences:
        # Tokenizing words.
        words = word_tokenize(sent.lower())
        # Removing non alphabetic and numeric words.
        words = [ps.stem(word) for word in words if word.isalnum()]
        # Removing stopwords
        clean_words += [word for word in words if word not in stop_words]

    return clean_words


def create_word_frequency_table(words: list) -> dict:
    """
    Creating word frequency table which contains frequency of each word used in the text.
    """
    print('Creating word frequency table')

    freq_table = dict()

    for word in words:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    return freq_table


def create_sentence_score_table(sentences: list, freq_table: dict) -> dict:
    """
    Creating a dictionary to keep the score of each sentence.
    Sore is the sum of frequency of words used in the sentence.
    """
    print('Creating sentence score table')

    sent_value = dict()
    for sentence in sentences:
        for word, freq in freq_table.items():
            if ps.stem(word) in sentence.lower():
                if sentence in sent_value:
                    sent_value[sentence] += freq
                else:
                    sent_value[sentence] = freq

    return sent_value

def create_summary(sentence_values : dict,summary_sentence_count) -> dict:
  summary = dict(sorted(sentence_values.items(), key=lambda item: item[1], reverse=True))
  output = ""
  c = 0
  for pairs in summary:
    if c < summary_sentence_count:
        output += pairs

    else:
        break

    c = c+1;
  return output

print("process is starting")
f = open("inputfile.txt", encoding="utf8")
text = ""
for x in f:
    print(x)
    text += x

# tokenize the sentences

sentences = sent_tokenize(text.strip())
print('Sentences', len(sentences), sentences)

clean_words = text_preprocessing(sentences)
print('Clean Words', len(clean_words), clean_words)

freq_table = create_word_frequency_table(clean_words)
print('Frequency Table', freq_table)

sent_values = create_sentence_score_table(sentences, freq_table)
print('Sentence values', sent_values)

total_sentence_count = len(sent_values)
summary_sentence_count = total_sentence_count//2
"""summary = dict(sorted(sent_values.items(), key=lambda item: item[1], reverse=True))

output = ""
c = 0
for pairs in summary:
    if c < summary_sentence_count:
        output += pairs

    else:
        break

    c = c+1;

print(output)"""

summary = create_summary(sent_values,summary_sentence_count)
print(summary)

