from nltk import precision
from textblob import TextBlob  # for sentiment detection

from deep_translator import GoogleTranslator  # for text translation

GoogleTranslator().get_supported_languages(as_dict=True)  # language dictionary
import keys  #api_key for translator api
from deep_translator import single_detection  # detection for a single text

#############
print("\n\n################################ SENTIMENT DETECTION #################################\n\n")
#############


text = "I am happy."

blob = TextBlob(text)

blob.sentiment  # sentiment object positive/negative and objective/subjective

for sentence in blob.sentences:
    print(f'{sentence}\nsentiment: {sentence.sentiment}')

#############
print("\n\n################################ TRANSLATION #################################\n\n")
#############

blob = " Ce faci? "

single_detection(blob, api_key=keys.deep_translator_key)

spanish = GoogleTranslator(source='auto', target='es').translate(blob)

print(spanish)

  # chinese = GoogleTranslator(source='auto', target='zh-CN').translate(blob)
  # print(chinese)

  #  english = GoogleTranslator(source='auto', target='en').translate(blob)
  #  print(english)


#############
print("\n\n################################# TRANSLATION 2 ################################\n\n")
#############


# autodetect source language and translate to English
english=GoogleTranslator(source='auto', target='en').translate(spanish)
print(english)
# autodetect source language and translate to English
chinese=GoogleTranslator(source='auto', target='zh-CN').translate(english)
print(chinese)



#############
print("\n\n################################# SPELL CHECKING 1 ################################\n\n")
#############

from textblob import Word

word=Word('theyr')

print("wrong word:",word)

print("correct words:",word.spellcheck())

print("corrected word:",word.correct())

#############
print("\n\n################################# SPELL CHECKING 2 ################################\n\n")
#############

sentence = TextBlob('Ths sentense has missplled wrds.')

print(sentence)

print(sentence.correct())


#############
print("\n\n################################# Big Text Processing ################################\n\n")
#############



from pathlib import Path

blob = TextBlob(Path('PythonDataScienceFullThrottle/ch11/1513-0_RomeoAndJulietOriginalDownload.txt').read_text())   # load romeo and juliet book

print(blob.word_counts['romeo'])  # counts 'romeo' instances


#############
print("\n\n################################# Deleting Stop Words ################################\n\n")
#############

import nltk
#nltk.download('stopwords')  # must download before first use
from nltk.corpus import stopwords

stops = stopwords.words('english')
stops
blob = TextBlob('These flowers are so beautiful!')
# keep anything that's not a stop word
[word for word in blob.words.lower() if word not in stops]
print(blob)

#############
print("\n\n################################# Visualizing Word Frequencies With Bar Charts and Word Clouds ################################\n\n")
#############



blob = TextBlob(Path('PythonDataScienceFullThrottle/ch11/1513-0_RomeoAndJulietOriginalDownload.txt').read_text())  #load book

items = blob.word_counts.items()  #iterator for word-frequency

items = [item for item in items if item[0] not in stops and item[0] != "'"]

print(items[:10])

#############
print("\n\n################################# Sorting the top 20 Words in descending order ################################\n\n")
#############

from operator import itemgetter

sorted_items = sorted(items, key=itemgetter(1), reverse=True)  # descending

# key=itemgetter(1) - sort tuples by frequency (each tuple's element = 1)

top20 = sorted_items[0:20]

print(top20)



#############
print("\n\n################################# Put Top20 into dataFrame for vizualization ################################\n\n")
#############

from IPython.display import display
import pandas as pd

df = pd.DataFrame(top20)

display(df)

import matplotlib.pyplot as plt

axes= df.plot.bar(x='word', y='count')
plt.gcf().tight_layout()

plt.show()