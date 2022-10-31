from nltk import precision
from textblob import TextBlob  # for sentiment detection

from deep_translator import GoogleTranslator  # for text translation

GoogleTranslator().get_supported_languages(as_dict=True)  # language dictionary

from deep_translator import single_detection  # detection for a single text

#############
print("\n\n################################ SENTIMENT DETECTION #################################\n\n")
#############


text = 'Yesterday was a beautiful day. Tommorow looks like bad weather'

blob = TextBlob(text)

TextBlob('Yesterday was a beautiful day. Tommorow looks like bad weather')

blob.sentiment  # sentiment object positive/negative and objective/subjective

for sentence in blob.sentences:
    print(f'{sentence}\nsentiment: {sentence.sentiment}')

#############
print("\n\n################################ TRANSLATION #################################\n\n")
#############

blob= " Ce faci? "

single_detection(blob, api_key="4e716625bb345516143661b9450ec377")

spanish = GoogleTranslator(source='auto', target='es').translate(blob)

print(spanish)

#chinese = GoogleTranslator(source='auto', target='zh-CN').translate(blob)
#print(chinese)

#english = GoogleTranslator(source='auto', target='en').translate(blob)
#print(english)


#############
print("\n\n################################# TRANSLATION 2 ################################\n\n")
#############



blob
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

