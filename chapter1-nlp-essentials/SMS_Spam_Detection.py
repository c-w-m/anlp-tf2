# Based on SMS_Spam_Detection
# edited to run on local PC without GPU setup

import io
import re
import stanza
import pandas as pd
import tensorflow as tf
import stopwordsiso as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api


print("TensorFlow Version: " + tf.__version__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nDownload Data\n - do this from notebook code")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nTest data reading:")
lines = io.open('data/SMSSpamCollection').read().strip().split('\n')
print(lines[0])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nPre-Process Data")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
spam_dataset = []
count = 0
for line in lines:
  label, text = line.split('\t')
  if label.lower().strip() == 'spam':
    spam_dataset.append((1, text.strip()))
    count += 1
  else:
    spam_dataset.append(((0, text.strip())))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nprint(spam_dataset[0])")
print(spam_dataset[0])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+'\n\nprint("Spam: ", count)')
print("Spam: ", count)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nData Normalization")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df = pd.DataFrame(spam_dataset, columns=['Spam', 'Message'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Normalization functions

def message_length(x):
  # returns total number of characters
  return len(x)

def num_capitals(x):
  _, count = re.subn(r'[A-Z]', '', x) # only works in english
  return count

def num_punctuation(x):
  _, count = re.subn(r'\W', '', x)
  return count

df['Capitals'] = df['Message'].apply(num_capitals)
df['Punctuation'] = df['Message'].apply(num_punctuation)
df['Length'] = df['Message'].apply(message_length)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nCorpus:")
print(df.describe())

train = df.sample(frac=0.8,random_state=42) #random state is a seed value
test = df.drop(train.index)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nTrain:")
print(train.describe())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nTest:")
print(train.describe())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nModel Building")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Basic 1-layer neural network model for evaluation
def make_model(input_dims=3, num_units=12):
    model = tf.keras.Sequential()

    # Adds a densely-connected layer with 12 units to the model:
    model.add(tf.keras.layers.Dense(num_units,
                                    input_dim=input_dims,
                                    activation='relu'))

    # Add a sigmoid layer with a binary output unit:
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'])
    return model

x_train = train[['Length', 'Punctuation', 'Capitals']]
y_train = train[['Spam']]

x_test = test[['Length', 'Punctuation', 'Capitals']]
y_test = test[['Spam']]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nx_train:")
print(x_train)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80 * "~") + "\n\nmodel = make_model():")
model = make_model()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nmodel.fit(x_train, y_train, epochs=10, batch_size=10)")
model.fit(x_train, y_train, epochs=10, batch_size=10)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nmodel.evaluation(x_test, y_test)")
model.evaluate(x_test, y_test)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\ny_train_pred = model.predict_classes(x_train)")
y_train_pred = model.predict_classes(x_train)
#print((80*"~")+"\n\ny_train_pred = np.argmax(model.predict(x_train), axis=-1)")
#y_train_pred: object = np.argmax(model.predict(x_train), axis=-1)

# confusion matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\ntf.math.confusion_matrix(tf.constant(y_train.Spam), y_train_pred)")
print(tf.math.confusion_matrix(tf.constant(y_train.Spam), y_train_pred))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nsum(y_train_pred)")
print(sum(y_train_pred))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\ny_test_pred = model.predict_classes(x_test)")
y_test_pred = model.predict_classes(x_test)
#print((80*"~")+"\n\ny_train_pred = np.argmax(model.predict(x_test), axis=-1)")
#y_test_pred = np.argmax(model.predict(x_test), axis=-1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\ntf.math.confusion_matrix(tf.constant(y_test.Spam), y_test_pred)")
print(tf.math.confusion_matrix(tf.constant(y_test.Spam), y_test_pred))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nTokenization and Stop Word Removal"+(80*"~"))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sentence = 'Go until jurong point, crazy.. Available only in bugis n great world'
sentence.split()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nen = stanza.download('en')")
en = stanza.download('en')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80 * "~") + "\n\nen = stanza.Pipeline(lang='en')")
en = stanza.Pipeline(lang='en')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nprint(sentence)")
print(sentence)

tokenized = en(sentence)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nprint(len(tokenized.sentences))")
print(len(tokenized.sentences))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nprint(<End of Sentence>)")

for snt in tokenized.sentences:
    for word in snt.tokens:
        print(word.text)
    print("<End of Sentence>")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nDependency Parsing Example\n"+(80*"~"))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nen2 = stanza.Pipeline(lang='en')")
en2 = stanza.Pipeline(lang='en')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nprint(<End of Sentence>)")
pr2 = en2("Hari went to school")
for snt in pr2.sentences:
    for word in snt.tokens:
        print(word)
    print("<End of Sentence>")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nJapanese Tokenization Example"+(80*"~"))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\njp = stanza.download('ja')")
jp = stanza.download('ja')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\njp = stanza.Pipeline(lang='ja')")
jp = stanza.Pipeline(lang='ja')

jp_line = jp("選挙管理委員会")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nsnt.tokens")
for snt in jp_line.sentences:
    for word in snt.tokens:
        print(word.text)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nAdding Word Count Feature"+(80*"~"))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def word_counts(x, pipeline=en):
    doc = pipeline(x)
    count = sum( [ len(sentence.tokens) for sentence in doc.sentences] )
    return count

#en = snlp.Pipeline(lang='en', processors='tokenize')
df['Words'] = df['Message'].apply(word_counts)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nCorpus: (Words added)")
print(df.describe())

#train=df.sample(frac=0.8,random_state=42) #random state is a seed value
#test=df.drop(train.index)

train['Words'] = train['Message'].apply(word_counts)
test['Words'] = test['Message'].apply(word_counts)

x_train = train[['Length', 'Punctuation', 'Capitals', 'Words']]
y_train = train[['Spam']]

x_test = test[['Length', 'Punctuation', 'Capitals' , 'Words']]
y_test = test[['Spam']]

model = make_model(input_dims=4)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nmodel.fit(x_train, y_train, epochs=10, batch_size=10)")
model.fit(x_train, y_train, epochs=10, batch_size=10)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nmodel.evaluate(x_test, y_test)")
model.evaluate(x_test, y_test)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nStop Word Removal")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print((80*"~") + "\n\nprint(stopwords.langs())")
print(stopwords.langs())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(sorted(stopwords.stopwords('en')))")
print(sorted(stopwords.stopwords('en')))

en_sw = stopwords.stopwords('en')

def word_counts(x, pipeline=en):
    doc = pipeline(x)
    count = 0
    for sentence in doc.sentences:
        for token in sentence.tokens:
            if token.text.lower() not in en_sw:
                count += 1
    return count

train['Words'] = train['Message'].apply(word_counts)
test['Words'] = test['Message'].apply(word_counts)

x_train = train[['Length', 'Punctuation', 'Capitals', 'Words']]
y_train = train[['Spam']]

x_test = test[['Length', 'Punctuation', 'Capitals' , 'Words']]
y_test = test[['Spam']]

model = make_model(input_dims=4)
#model = make_model(input_dims=3)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nmodel.fit(x_train, y_train, epochs=10, batch_size=10)")
model.fit(x_train, y_train, epochs=10, batch_size=10)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\n"+(80*"~")+"\nPOS Based Features"+(80*"~"))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~")+"\n\nen = stanza.Pipeline(lang='en')")
en = stanza.Pipeline(lang='en')

txt = "Yo you around? A friend of mine's lookin."
pos = en(txt)

def print_pos(doc):
    text = ""
    for sentence in doc.sentences:
        for token in sentence.tokens:
            text += token.words[0].text + "/" + \
                    token.words[0].upos + " "
        text += "\n"
    return text

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(print_pos(pos))")
print(print_pos(pos))

en_sw = stopwords.stopwords('en')

def word_counts_v3(x, pipeline=en):
    doc = pipeline(x)
    count = 0
    for sentence in doc.sentences:
      for token in sentence.tokens:
          if token.text.lower() not in en_sw and token.words[0].upos not in ['PUNCT', 'SYM']:
              count += 1
    return count

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(word_counts(txt), word_counts_v3(txt))")
print(word_counts(txt), word_counts_v3(txt))

train['Test'] = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(train.describe())")
print(train.describe())

def word_counts_v3(x, pipeline=en):
    doc = pipeline(x)
    totals = 0.
    count = 0.
    non_word = 0.
    for sentence in doc.sentences:
      totals += len(sentence.tokens)  # (1)
      for token in sentence.tokens:
          if token.text.lower() not in en_sw:
              if token.words[0].upos not in ['PUNCT', 'SYM']:
                  count += 1.
              else:
                  non_word += 1.
    non_word = non_word / totals
    return pd.Series([count, non_word], index=['Words_NoPunct', 'Punct'])

x = train[:10]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nx.describe()")
print(x.describe())

train_tmp = train['Message'].apply(word_counts_v3)
train = pd.concat([train, train_tmp], axis=1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\ntrain.describe()")
print(train.describe())

test_tmp = test['Message'].apply(word_counts_v3)
test = pd.concat([test, test_tmp], axis=1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\ntest.describe()")
print(test.describe())

z = pd.concat([x, train_tmp], axis=1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(z.describe())")
print(z.describe())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(z.loc[z['Spam']==0].describe())")
print(z.loc[z['Spam']==0].describe())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(z.loc[z['Spam']==1].describe())")
print(z.loc[z['Spam']==1].describe())

aa = [word_counts_v3(y) for y in x['Message']]

ab = pd.DataFrame(aa)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(ab.describe())")
print(ab.describe())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\n" + (80*"~") +"\nLemmatization")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

text = "Stemming is aimed at reducing vocabulary and aid un-derstanding of" +\
       " morphological processes. This helps people un-derstand the" +\
       " morphology of words and reduce size of corpus."

lemma = en(text)

lemmas = ""
for sentence in lemma.sentences:
        for token in sentence.tokens:
            lemmas += token.words[0].lemma +"/" + \
                    token.words[0].upos + " "
        lemmas += "\n"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(lemmas)")
print(lemmas)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\n" + (80*"~") + "\nTF-IDF Based Model\n" + (80*"~"))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

corpus = [
          "I like fruits. Fruits like bananas",
          "I love bananas but eat an apple",
          "An apple a day keeps the doctor away"
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\n" + (80*"~") +"\nCount Vectorization")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(vectorizer.get_feature_names())")
print(vectorizer.get_feature_names())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(X.toarray())")
print(X.toarray())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(cosine_similarity(X.toarray()))")
print(cosine_similarity(X.toarray()))

query = vectorizer.transform(["apple and bananas"])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(cosine_similarity(X, query))")
print(cosine_similarity(X, query))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\n" + (80*"~") +"\nTF-IDF Vectorization")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X.toarray())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(tfidf.toarray())")
print(pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names()))

tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train['Message']).astype('float32')
X_test = tfidf.transform(test['Message']).astype('float32')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(X.shape)")
print(X.shape)

_, cols = X.shape
model2 = make_model(cols)  # to match tf-idf dimensions
lb = LabelEncoder()
y = lb.fit_transform(y_train)
dummy_y_train = np_utils.to_categorical(y)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nmodel2.fit(X.toarray(), y_train, epochs=10, batch_size=10)")
model2.fit(X.toarray(), y_train, epochs=10, batch_size=10)

model2.evaluate(X_test.toarray(), y_test)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(train.loc[train.Spam == 1].describe())")
print(train.loc[train.Spam == 1].describe())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\n" + (80*"~") +"\nWord Vectors")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(api.info())")
print(api.info())

model_w2v = api.load("word2vec-google-news-300")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(model_w2v.most_similar('cookies',topn=10))")
print(model_w2v.most_similar("cookies",topn=10))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(model_w2v.doesnt_match(['USA','Canada','India','Tokyo']))")
print(model_w2v.doesnt_match(["USA","Canada","India","Tokyo"]))

king = model_w2v['king']
man = model_w2v['man']
woman = model_w2v['woman']

queen = king - man + woman
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print((80*"~") + "\n\nprint(model_w2v.similar_by_vector(queen))")
print(model_w2v.similar_by_vector(queen))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n\n" + (80*"~") +"\n-- end of 'SMS_Spam_Detection.py' --\n" + (
        80*"~"))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
