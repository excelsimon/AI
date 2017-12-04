# -*- coding:utf-8 -*-
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class LanguageDetector():
    def __init__(self,classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(
            lowercase=True,
            analyzer='char_wb',
            ngram_range=(1,2),
            max_features=1000,
            preprocessor=self._remove_noise
        )

    def _remove_noise(self,document):
        noise_pattern = re.compile('|'.join(['http\S+', '\@\w+', '\#\w+']))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text.strip()

    def features(self,x):
        return self.vectorizer.transform(x)

    def fit(self,x,y):
        self.vectorizer.fit(x)
        self.classifier.fit(self.features(x),y)

    def predict(self,x):
        return self.classifier.predict(self.features([x]))

    def score(self,x,y):
        return self.classifier.score(self.features(x),y)



data_f = open('language_detector.csv')
lines = data_f.readlines()
data_f.close()

dataset = [(line.strip()[:-3],line.strip()[-2:]) for line in lines]
x,y = zip(*dataset)  #x,y为list,x包含所有句子，y包含对应标签
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

language_detector = LanguageDetector()
language_detector.fit(x_train,y_train)

print(language_detector.score(x_test,y_test))
print(language_detector.predict('This is an english sentence'))

"""
output:
0.977941176471
['en']
"""






















