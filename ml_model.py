from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

data = [
    ['woof', 1],
    ['bark', 1],
    ['ruff', 1],
    ['bowwow', 1],
    ['roar', 0],
    ['bah', 0],
    ['meow', 0],
    ['ribbit', 0],
    ['moo', 0],
    ['yip', 0],
    ['pika', 0]
]

X = []
y = []
for i in data:
    X.append( i[0] )
    y.append( i[1] )

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

regressor = LinearRegression()
regressor.fit(X_vectorized, y)

test_feature = vectorizer.transform(['woof'])
prediction = regressor.predict(test_feature)
print(prediction)
test_feature = vectorizer.transform(['ribbit'])
prediction = regressor.predict(test_feature)
print(prediction)
test_feature = vectorizer.transform(['meoww'])
prediction = regressor.predict(test_feature)
print(prediction)

pickl = {
    'vectorizer': vectorizer,
    'regressor': regressor
}
pickle.dump( pickl, open( 'models' + ".p", "wb" ) )