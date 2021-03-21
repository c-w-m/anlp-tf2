Output from: SMS_Spam_Detection.py
-- manually added markdown to highlight sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

```shell
/home/cwm/git/git.c-w-m/nlp_dev/src/anlp-tf2/.tox/anlp37/bin/python 
/home/cwm/git/git.c-w-m/nlp_dev/src/anlp-tf2/chapter1-nlp-essentials/SMS_Spam_Detection.py

2021-03-20 22:56:28.274801: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] 
Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: 
cannot open shared object file: No such file or directory

2021-03-20 22:56:28.274836: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] 
Ignore above cudart dlerror if you do not have a GPU set up on your machine.

TensorFlow Version: 2.4.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Download Data
- do this from notebook code
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test data reading:
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Pre-Process Data
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(spam_dataset[0])
(0, 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("Spam: ", count)
Spam:  747
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Data Normalization
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Corpus:
              Spam     Capitals  Punctuation       Length
count  5574.000000  5574.000000  5574.000000  5574.000000
mean      0.134015     5.621636    18.942591    80.443488
std       0.340699    11.683233    14.825994    59.841746
min       0.000000     0.000000     0.000000     2.000000
25%       0.000000     1.000000     8.000000    36.000000
50%       0.000000     2.000000    15.000000    61.000000
75%       0.000000     4.000000    27.000000   122.000000
max       1.000000   129.000000   253.000000   910.000000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train:
              Spam     Capitals  Punctuation       Length
count  4459.000000  4459.000000  4459.000000  4459.000000
mean      0.132765     5.519399    18.886522    80.316439
std       0.339359    11.405424    14.602023    59.346407
min       0.000000     0.000000     0.000000     2.000000
25%       0.000000     1.000000     8.000000    35.000000
50%       0.000000     2.000000    15.000000    61.000000
75%       0.000000     4.000000    27.000000   122.000000
max       1.000000   129.000000   253.000000   910.000000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test:
              Spam     Capitals  Punctuation       Length
count  4459.000000  4459.000000  4459.000000  4459.000000
mean      0.132765     5.519399    18.886522    80.316439
std       0.339359    11.405424    14.602023    59.346407
min       0.000000     0.000000     0.000000     2.000000
25%       0.000000     1.000000     8.000000    35.000000
50%       0.000000     2.000000    15.000000    61.000000
75%       0.000000     4.000000    27.000000   122.000000
max       1.000000   129.000000   253.000000   910.000000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Model Building
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x_train:
      Length  Punctuation  Capitals
3690      25            4         1
3527     161           48       107
724       40            7         1
3370      69           17         3
468       37            8         1
...      ...          ...       ...
3280     444          114        44
3186      65           14        50
3953      81           23         2
2768      38            8         2
4223      65           14         2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = make_model():
[4459 rows x 3 columns]
2021-03-20 22:56:30.973534: 
I tensorflow/compiler/jit/xla_cpu_device.cc:41] 
Not creating XLA devices, tf_xla_enable_xla_devices not set

2021-03-20 22:56:30.973852: 
W tensorflow/stream_executor/platform/default/dso_loader.cc:60] 
Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: 
cannot open shared object file: 
No such file or directory

2021-03-20 22:56:30.973881: 
W tensorflow/stream_executor/cuda/cuda_driver.cc:326] 
failed call to cuInit: 
UNKNOWN ERROR (303)

2021-03-20 22:56:30.973927: 
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] 
kernel driver does not appear to be running on this host (flxsa02): /proc/driver/nvidia/version does not exist

2021-03-20 22:56:30.974358: 
I tensorflow/core/platform/cpu_feature_guard.cc:142] 
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) 
to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

2021-03-20 22:56:30.975189: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.fit(x_train, y_train, epochs=10, batch_size=10)
2021-03-20 22:56:31.102943: 
I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] 
None of the MLIR optimization passes are enabled (registered 2)

2021-03-20 22:56:31.120522: 
I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2793595000 Hz

Epoch 1/10
446/446 [==============================] - 1s 1ms/step - loss: 1.5046 - accuracy: 0.8567
Epoch 2/10
446/446 [==============================] - 1s 1ms/step - loss: 0.4440 - accuracy: 0.8705
Epoch 3/10
446/446 [==============================] - 0s 1ms/step - loss: 0.3710 - accuracy: 0.8666
Epoch 4/10
446/446 [==============================] - 0s 1ms/step - loss: 0.3271 - accuracy: 0.8753
Epoch 5/10
446/446 [==============================] - 0s 1ms/step - loss: 0.2398 - accuracy: 0.9135
Epoch 6/10
446/446 [==============================] - 0s 1ms/step - loss: 0.2090 - accuracy: 0.9276
Epoch 7/10
446/446 [==============================] - 0s 1ms/step - loss: 0.2114 - accuracy: 0.9220
Epoch 8/10
446/446 [==============================] - 0s 1ms/step - loss: 0.2083 - accuracy: 0.9274
Epoch 9/10
446/446 [==============================] - 0s 1ms/step - loss: 0.2015 - accuracy: 0.9304
Epoch 10/10
446/446 [==============================] - 0s 1ms/step - loss: 0.2130 - accuracy: 0.9194
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.evaluation(x_test, y_test)
35/35 [==============================] - 0s 884us/step - loss: 0.2010 - accuracy: 0.9345
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

y_train_pred = model.predict_classes(x_train)
/home/cwm/git/git.c-w-m/nlp_dev/src/anlp-tf2/.tox/anlp37/lib/python3.7/site-packages/tensorflow/python/keras/engine/sequential.py:450: 

UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. 

Please use instead:* `np.argmax(model.predict(x), axis=-1)`,
   if your model does multi-class classification
      (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,

   if your model does binary classification
      (e.g. if it uses a `sigmoid` last-layer activation).

  warnings.warn('`model.predict_classes()` is deprecated and '
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tf.math.confusion_matrix(tf.constant(y_train.Spam), y_train_pred)
tf.Tensor(
[[3765  102]
 [ 173  419]], shape=(2, 2), dtype=int32)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sum(y_train_pred)
[521]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

y_test_pred = model.predict_classes(x_test)
/home/cwm/git/git.c-w-m/nlp_dev/src/anlp-tf2/.tox/anlp37/lib/python3.7/site-packages/tensorflow/python/keras/engine/sequential.py:450: 

UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. 

Please use instead:* `np.argmax(model.predict(x), axis=-1)`,
   if your model does multi-class classification
      (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,

   if your model does binary classification
      (e.g. if it uses a `sigmoid` last-layer activation).

  warnings.warn('`model.predict_classes()` is deprecated and '
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tf.math.confusion_matrix(tf.constant(y_test.Spam), y_test_pred)
tf.Tensor(
[[939  21]
 [ 52 103]], shape=(2, 2), dtype=int32)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Tokenization and Stop Word Removal
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

en = stanza.download('en')
Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/master/resources_1.2.0.json: 128kB [00:00, 40.3MB/s]                    
2021-03-20 22:56:37 INFO: Downloading default packages for language: en (English)...
2021-03-20 22:56:38 INFO: File exists: /home/cwm/stanza_resources/en/default.zip.
2021-03-20 22:56:45 INFO: Finished downloading models and saved to /home/cwm/stanza_resources.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

en = stanza.Pipeline(lang='en')
2021-03-20 22:56:45 INFO: Loading these models for language: en (English):
=========================
| Processor | Package   |
-------------------------
| tokenize  | combined  |
| pos       | combined  |
| lemma     | combined  |
| depparse  | combined  |
| sentiment | sstplus   |
| ner       | ontonotes |
=========================

2021-03-20 22:56:45 INFO: Use device: cpu
2021-03-20 22:56:45 INFO: Loading: tokenize
2021-03-20 22:56:45 INFO: Loading: pos
2021-03-20 22:56:46 INFO: Loading: lemma
2021-03-20 22:56:46 INFO: Loading: depparse
2021-03-20 22:56:46 INFO: Loading: sentiment
2021-03-20 22:56:47 INFO: Loading: ner
2021-03-20 22:56:48 INFO: Done loading processors!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(sentence)
Go until jurong point, crazy.. Available only in bugis n great world
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(len(tokenized.sentences))
2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(<End of Sentence>)
Go
until
jurong
point
,
crazy
..
<End of Sentence>
Available
only
in
bugis
n
great
world
<End of Sentence>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Dependency Parsing Example
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

en2 = stanza.Pipeline(lang='en')
2021-03-20 22:56:49 INFO: Loading these models for language: en (English):
=========================
| Processor | Package   |
-------------------------
| tokenize  | combined  |
| pos       | combined  |
| lemma     | combined  |
| depparse  | combined  |
| sentiment | sstplus   |
| ner       | ontonotes |
=========================

2021-03-20 22:56:49 INFO: Use device: cpu
2021-03-20 22:56:49 INFO: Loading: tokenize
2021-03-20 22:56:49 INFO: Loading: pos
2021-03-20 22:56:49 INFO: Loading: lemma
2021-03-20 22:56:49 INFO: Loading: depparse
2021-03-20 22:56:50 INFO: Loading: sentiment
2021-03-20 22:56:50 INFO: Loading: ner
2021-03-20 22:56:51 INFO: Done loading processors!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(<End of Sentence>)
[
  {
    "id": 1,
    "text": "Hari",
    "lemma": "Hari",
    "upos": "PROPN",
    "xpos": "NNP",
    "feats": "Number=Sing",
    "head": 2,
    "deprel": "nsubj",
    "misc": "start_char=0|end_char=4",
    "ner": "S-PERSON"
  }
]
[
  {
    "id": 2,
    "text": "went",
    "lemma": "go",
    "upos": "VERB",
    "xpos": "VBD",
    "feats": "Mood=Ind|Tense=Past|VerbForm=Fin",
    "head": 0,
    "deprel": "root",
    "misc": "start_char=5|end_char=9",
    "ner": "O"
  }
]
[
  {
    "id": 3,
    "text": "to",
    "lemma": "to",
    "upos": "ADP",
    "xpos": "IN",
    "head": 4,
    "deprel": "case",
    "misc": "start_char=10|end_char=12",
    "ner": "O"
  }
]
[
  {
    "id": 4,
    "text": "school",
    "lemma": "school",
    "upos": "NOUN",
    "xpos": "NN",
    "feats": "Number=Sing",
    "head": 2,
    "deprel": "obl",
    "misc": "start_char=13|end_char=19",
    "ner": "O"
  }
]
<End of Sentence>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Japanese Tokenization Example
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

jp = stanza.download('ja') 
Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/master/resources_1.2.0.json: 128kB [00:00, 37.6MB/s]                    
2021-03-20 22:56:52 INFO: Downloading default packages for language: ja (Japanese)...
2021-03-20 22:56:52 INFO: File exists: /home/cwm/stanza_resources/ja/default.zip.
2021-03-20 22:56:55 INFO: Finished downloading models and saved to /home/cwm/stanza_resources.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

jp = stanza.Pipeline(lang='ja')
2021-03-20 22:56:56 INFO: Loading these models for language: ja (Japanese):
=======================
| Processor | Package |
-----------------------
| tokenize  | gsd     |
| pos       | gsd     |
| lemma     | gsd     |
| depparse  | gsd     |
=======================

2021-03-20 22:56:56 INFO: Use device: cpu
2021-03-20 22:56:56 INFO: Loading: tokenize
2021-03-20 22:56:56 INFO: Loading: pos
2021-03-20 22:56:56 INFO: Loading: lemma
2021-03-20 22:56:56 INFO: Loading: depparse
2021-03-20 22:56:57 INFO: Done loading processors!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

snt.tokens:
選挙
管理
委員
会
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Adding Word Count Feature
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Corpus: (Words added)
              Spam     Capitals  Punctuation       Length        Words
count  5574.000000  5574.000000  5574.000000  5574.000000  5574.000000
mean      0.134015     5.621636    18.942591    80.443488    18.739146
std       0.340699    11.683233    14.825994    59.841746    13.765308
min       0.000000     0.000000     0.000000     2.000000     1.000000
25%       0.000000     1.000000     8.000000    36.000000     9.000000
50%       0.000000     2.000000    15.000000    61.000000    15.000000
75%       0.000000     4.000000    27.000000   122.000000    27.000000
max       1.000000   129.000000   253.000000   910.000000   207.000000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.fit(x_train, y_train, epochs=10, batch_size=10)
Epoch 1/10
446/446 [==============================] - 1s 959us/step - loss: 1.0459 - accuracy: 0.6780
Epoch 2/10
446/446 [==============================] - 0s 955us/step - loss: 0.3752 - accuracy: 0.8928
Epoch 3/10
446/446 [==============================] - 0s 972us/step - loss: 0.2869 - accuracy: 0.9143
Epoch 4/10
446/446 [==============================] - 0s 982us/step - loss: 0.2523 - accuracy: 0.9222
Epoch 5/10
446/446 [==============================] - 0s 975us/step - loss: 0.2335 - accuracy: 0.9263
Epoch 6/10
446/446 [==============================] - 0s 957us/step - loss: 0.2147 - accuracy: 0.9283
Epoch 7/10
446/446 [==============================] - 0s 969us/step - loss: 0.2085 - accuracy: 0.9320
Epoch 8/10
446/446 [==============================] - 0s 980us/step - loss: 0.2075 - accuracy: 0.9292
Epoch 9/10
446/446 [==============================] - 0s 967us/step - loss: 0.1926 - accuracy: 0.9369
Epoch 10/10
446/446 [==============================] - 0s 973us/step - loss: 0.2039 - accuracy: 0.9307
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.evaluate(x_test, y_test)
35/35 [==============================] - 0s 831us/step - loss: 0.1998 - accuracy: 0.9300
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Stop Word Removal
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(stopwords.langs())
{'ga', 'so', 'id', 'ms', 'nl', 'hu', 'gl', 'uk', 'mr', 'it', 'ja', 'sv', 'bg', 'pl', 'hy', 'ur', 'ru', 'sl', 'vi', 'tr', 'bn', 'lt', 'es', 'lv', 'fi', 'yo', 'ko', 'ku', 'zh', 'sk', 'eo', 'af', 'he', 'et', 'hr', 'pt', 'ca', 'sw', 'th', 'en', 'gu', 'hi', 'ar', 'no', 'ha', 'fa', 'eu', 'ro', 'el', 'st', 'fr', 'da', 'zu', 'br', 'tl', 'cs', 'la', 'de'}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(sorted(stopwords.stopwords('en')))
["'ll", "'tis", "'twas", "'ve", '10', '39', 'a', "a's", 'able', 'ableabout', 'about', 'above', 'abroad', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'ad', 'added', 'adj', 'adopted', 'ae', 'af', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'ag', 'again', 'against', 'ago', 'ah', 'ahead', 'ai', "ain't", 'aint', 'al', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'also', 'although', 'always', 'am', 'amid', 'amidst', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'ao', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'aq', 'ar', 'are', 'area', 'areas', 'aren', "aren't", 'arent', 'arise', 'around', 'arpa', 'as', 'aside', 'ask', 'asked', 'asking', 'asks', 'associated', 'at', 'au', 'auth', 'available', 'aw', 'away', 'awfully', 'az', 'b', 'ba', 'back', 'backed', 'backing', 'backs', 'backward', 'backwards', 'bb', 'bd', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'beings', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'bf', 'bg', 'bh', 'bi', 'big', 'bill', 'billion', 'biol', 'bj', 'bm', 'bn', 'bo', 'both', 'bottom', 'br', 'brief', 'briefly', 'bs', 'bt', 'but', 'buy', 'bv', 'bw', 'by', 'bz', 'c', "c'mon", "c's", 'ca', 'call', 'came', 'can', "can't", 'cannot', 'cant', 'caption', 'case', 'cases', 'cause', 'causes', 'cc', 'cd', 'certain', 'certainly', 'cf', 'cg', 'ch', 'changes', 'ci', 'ck', 'cl', 'clear', 'clearly', 'click', 'cm', 'cmon', 'cn', 'co', 'co.', 'com', 'come', 'comes', 'computer', 'con', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'copy', 'corresponding', 'could', "could've", 'couldn', "couldn't", 'couldnt', 'course', 'cr', 'cry', 'cs', 'cu', 'currently', 'cv', 'cx', 'cy', 'cz', 'd', 'dare', "daren't", 'darent', 'date', 'de', 'dear', 'definitely', 'describe', 'described', 'despite', 'detail', 'did', 'didn', "didn't", 'didnt', 'differ', 'different', 'differently', 'directly', 'dj', 'dk', 'dm', 'do', 'does', 'doesn', "doesn't", 'doesnt', 'doing', 'don', "don't", 'done', 'dont', 'doubtful', 'down', 'downed', 'downing', 'downs', 'downwards', 'due', 'during', 'dz', 'e', 'each', 'early', 'ec', 'ed', 'edu', 'ee', 'effect', 'eg', 'eh', 'eight', 'eighty', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'end', 'ended', 'ending', 'ends', 'enough', 'entirely', 'er', 'es', 'especially', 'et', 'et-al', 'etc', 'even', 'evenly', 'ever', 'evermore', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'fairly', 'far', 'farther', 'felt', 'few', 'fewer', 'ff', 'fi', 'fifteen', 'fifth', 'fifty', 'fify', 'fill', 'find', 'finds', 'fire', 'first', 'five', 'fix', 'fj', 'fk', 'fm', 'fo', 'followed', 'following', 'follows', 'for', 'forever', 'former', 'formerly', 'forth', 'forty', 'forward', 'found', 'four', 'fr', 'free', 'from', 'front', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthermore', 'furthers', 'fx', 'g', 'ga', 'gave', 'gb', 'gd', 'ge', 'general', 'generally', 'get', 'gets', 'getting', 'gf', 'gg', 'gh', 'gi', 'give', 'given', 'gives', 'giving', 'gl', 'gm', 'gmt', 'gn', 'go', 'goes', 'going', 'gone', 'good', 'goods', 'got', 'gotten', 'gov', 'gp', 'gq', 'gr', 'great', 'greater', 'greatest', 'greetings', 'group', 'grouped', 'grouping', 'groups', 'gs', 'gt', 'gu', 'gw', 'gy', 'h', 'had', "hadn't", 'hadnt', 'half', 'happens', 'hardly', 'has', 'hasn', "hasn't", 'hasnt', 'have', 'haven', "haven't", 'havent', 'having', 'he', "he'd", "he'll", "he's", 'hed', 'hell', 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'herse”', 'hes', 'hi', 'hid', 'high', 'higher', 'highest', 'him', 'himself', 'himse”', 'his', 'hither', 'hk', 'hm', 'hn', 'home', 'homepage', 'hopefully', 'how', "how'd", "how'll", "how's", 'howbeit', 'however', 'hr', 'ht', 'htm', 'html', 'http', 'hu', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'i.e.', 'id', 'ie', 'if', 'ignored', 'ii', 'il', 'ill', 'im', 'immediate', 'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'inc.', 'indeed', 'index', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'inside', 'insofar', 'instead', 'int', 'interest', 'interested', 'interesting', 'interests', 'into', 'invention', 'inward', 'io', 'iq', 'ir', 'is', 'isn', "isn't", 'isnt', 'it', "it'd", "it'll", "it's", 'itd', 'itll', 'its', 'itself', 'itse”', 'ive', 'j', 'je', 'jm', 'jo', 'join', 'jp', 'just', 'k', 'ke', 'keep', 'keeps', 'kept', 'keys', 'kg', 'kh', 'ki', 'kind', 'km', 'kn', 'knew', 'know', 'known', 'knows', 'kp', 'kr', 'kw', 'ky', 'kz', 'l', 'la', 'large', 'largely', 'last', 'lately', 'later', 'latest', 'latter', 'latterly', 'lb', 'lc', 'least', 'length', 'less', 'lest', 'let', "let's", 'lets', 'li', 'like', 'liked', 'likely', 'likewise', 'line', 'little', 'lk', 'll', 'long', 'longer', 'longest', 'look', 'looking', 'looks', 'low', 'lower', 'lr', 'ls', 'lt', 'ltd', 'lu', 'lv', 'ly', 'm', 'ma', 'made', 'mainly', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', "mayn't", 'maynt', 'mc', 'md', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'member', 'members', 'men', 'merely', 'mg', 'mh', 'microsoft', 'might', "might've", "mightn't", 'mightnt', 'mil', 'mill', 'million', 'mine', 'minus', 'miss', 'mk', 'ml', 'mm', 'mn', 'mo', 'more', 'moreover', 'most', 'mostly', 'move', 'mp', 'mq', 'mr', 'mrs', 'ms', 'msie', 'mt', 'mu', 'much', 'mug', 'must', "must've", "mustn't", 'mustnt', 'mv', 'mw', 'mx', 'my', 'myself', 'myse”', 'mz', 'n', 'na', 'name', 'namely', 'nay', 'nc', 'nd', 'ne', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needed', 'needing', "needn't", 'neednt', 'needs', 'neither', 'net', 'netscape', 'never', 'neverf', 'neverless', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nf', 'ng', 'ni', 'nine', 'ninety', 'nl', 'no', 'no-one', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'np', 'nr', 'nu', 'null', 'number', 'numbers', 'nz', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'older', 'oldest', 'om', 'omitted', 'on', 'once', 'one', "one's", 'ones', 'only', 'onto', 'open', 'opened', 'opening', 'opens', 'opposite', 'or', 'ord', 'order', 'ordered', 'ordering', 'orders', 'org', 'other', 'others', 'otherwise', 'ought', "oughtn't", 'oughtnt', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'pa', 'page', 'pages', 'part', 'parted', 'particular', 'particularly', 'parting', 'parts', 'past', 'pe', 'per', 'perhaps', 'pf', 'pg', 'ph', 'pk', 'pl', 'place', 'placed', 'places', 'please', 'plus', 'pm', 'pmid', 'pn', 'point', 'pointed', 'pointing', 'points', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'pr', 'predominantly', 'present', 'presented', 'presenting', 'presents', 'presumably', 'previously', 'primarily', 'probably', 'problem', 'problems', 'promptly', 'proud', 'provided', 'provides', 'pt', 'put', 'puts', 'pw', 'py', 'q', 'qa', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'reasonably', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'reserved', 'respectively', 'resulted', 'resulting', 'results', 'right', 'ring', 'ro', 'room', 'rooms', 'round', 'ru', 'run', 'rw', 's', 'sa', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sb', 'sc', 'sd', 'se', 'sec', 'second', 'secondly', 'seconds', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'sees', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'seventy', 'several', 'sg', 'sh', 'shall', "shan't", 'shant', 'she', "she'd", "she'll", "she's", 'shed', 'shell', 'shes', 'should', "should've", 'shouldn', "shouldn't", 'shouldnt', 'show', 'showed', 'showing', 'shown', 'showns', 'shows', 'si', 'side', 'sides', 'significant', 'significantly', 'similar', 'similarly', 'since', 'sincere', 'site', 'six', 'sixty', 'sj', 'sk', 'sl', 'slightly', 'sm', 'small', 'smaller', 'smallest', 'sn', 'so', 'some', 'somebody', 'someday', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'sr', 'st', 'state', 'states', 'still', 'stop', 'strongly', 'su', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure', 'sv', 'sy', 'system', 'sz', 't', "t's", 'take', 'taken', 'taking', 'tc', 'td', 'tell', 'ten', 'tends', 'test', 'text', 'tf', 'tg', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've", 'thatll', 'thats', 'thatve', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'thered', 'therefore', 'therein', 'therell', 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 'thereve', 'these', 'they', "they'd", "they'll", "they're", "they've", 'theyd', 'theyll', 'theyre', 'theyve', 'thick', 'thin', 'thing', 'things', 'think', 'thinks', 'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh', 'thought', 'thoughts', 'thousand', 'three', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'till', 'tip', 'tis', 'tj', 'tk', 'tm', 'tn', 'to', 'today', 'together', 'too', 'took', 'top', 'toward', 'towards', 'tp', 'tr', 'tried', 'tries', 'trillion', 'truly', 'try', 'trying', 'ts', 'tt', 'turn', 'turned', 'turning', 'turns', 'tv', 'tw', 'twas', 'twelve', 'twenty', 'twice', 'two', 'tz', 'u', 'ua', 'ug', 'uk', 'um', 'un', 'under', 'underneath', 'undoing', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'upwards', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'uucp', 'uy', 'uz', 'v', 'va', 'value', 'various', 'vc', 've', 'versus', 'very', 'vg', 'vi', 'via', 'viz', 'vn', 'vol', 'vols', 'vs', 'vu', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'wasn', "wasn't", 'wasnt', 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'web', 'webpage', 'website', 'wed', 'welcome', 'well', 'wells', 'went', 'were', 'weren', "weren't", 'werent', 'weve', 'wf', 'what', "what'd", "what'll", "what's", "what've", 'whatever', 'whatll', 'whats', 'whatve', 'when', "when'd", "when'll", "when's", 'whence', 'whenever', 'where', "where'd", "where'll", "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'whichever', 'while', 'whilst', 'whim', 'whither', 'who', "who'd", "who'll", "who's", 'whod', 'whoever', 'whole', 'wholl', 'whom', 'whomever', 'whos', 'whose', 'why', "why'd", "why'll", "why's", 'widely', 'width', 'will', 'willing', 'wish', 'with', 'within', 'without', 'won', "won't", 'wonder', 'wont', 'words', 'work', 'worked', 'working', 'works', 'world', 'would', "would've", 'wouldn', "wouldn't", 'wouldnt', 'ws', 'www', 'x', 'y', 'ye', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'youll', 'young', 'younger', 'youngest', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'youve', 'yt', 'yu', 'z', 'za', 'zero', 'zm', 'zr']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model.fit(x_train, y_train, epochs=10, batch_size=10)
Epoch 1/10
446/446 [==============================] - 1s 956us/step - loss: 2.6172 - accuracy: 0.5460
Epoch 2/10
446/446 [==============================] - 0s 965us/step - loss: 0.7889 - accuracy: 0.7505
Epoch 3/10
446/446 [==============================] - 0s 978us/step - loss: 0.4589 - accuracy: 0.8532
Epoch 4/10
446/446 [==============================] - 0s 942us/step - loss: 0.3275 - accuracy: 0.8979
Epoch 5/10
446/446 [==============================] - 0s 971us/step - loss: 0.2487 - accuracy: 0.9231
Epoch 6/10
446/446 [==============================] - 0s 980us/step - loss: 0.2262 - accuracy: 0.9246
Epoch 7/10
446/446 [==============================] - 0s 977us/step - loss: 0.2074 - accuracy: 0.9281
Epoch 8/10
446/446 [==============================] - 0s 958us/step - loss: 0.2106 - accuracy: 0.9248
Epoch 9/10
446/446 [==============================] - 0s 970us/step - loss: 0.1868 - accuracy: 0.9307
Epoch 10/10
446/446 [==============================] - 0s 973us/step - loss: 0.1881 - accuracy: 0.9344
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# POS Based Features
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


2021-03-21 00:28:42 INFO: Loading these models for language: en (English):
=========================
| Processor | Package   |
-------------------------
| tokenize  | combined  |
| pos       | combined  |
| lemma     | combined  |
| depparse  | combined  |
| sentiment | sstplus   |
| ner       | ontonotes |
=========================

2021-03-21 00:28:42 INFO: Use device: cpu
2021-03-21 00:28:42 INFO: Loading: tokenize
2021-03-21 00:28:42 INFO: Loading: pos
2021-03-21 00:28:42 INFO: Loading: lemma
2021-03-21 00:28:42 INFO: Loading: depparse
2021-03-21 00:28:43 INFO: Loading: sentiment
2021-03-21 00:28:44 INFO: Loading: ner
2021-03-21 00:28:45 INFO: Done loading processors!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(print_pos(pos))
Yo/PRON you/PRON around/ADV ?/PUNCT 
A/DET friend/NOUN of/ADP mine/PRON 's/PART lookin/NOUN ./PUNCT 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(word_counts(txt), word_counts_v3(txt))
6 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(train.describe())
              Spam     Capitals  Punctuation       Length        Words    Test
count  4459.000000  4459.000000  4459.000000  4459.000000  4459.000000  4459.0
mean      0.132765     5.519399    18.886522    80.316439     9.215743     0.0
std       0.339359    11.405424    14.602023    59.346407     7.957216     0.0
min       0.000000     0.000000     0.000000     2.000000     0.000000     0.0
25%       0.000000     1.000000     8.000000    35.000000     4.000000     0.0
50%       0.000000     2.000000    15.000000    61.000000     7.000000     0.0
75%       0.000000     4.000000    27.000000   122.000000    13.000000     0.0
max       1.000000   129.000000   253.000000   910.000000   154.000000     0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x.describe()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train.describe()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test.describe()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(z.describe())
       Spam    Capitals  Punctuation  ...  Test  Words_NoPunct        Punct
count  10.0   10.000000    10.000000  ...  10.0    4459.000000  4459.000000
mean    0.0   14.400000    18.300000  ...   0.0       6.444046     0.148006
std     0.0   32.948445    14.772723  ...   0.0       5.601280     0.095298
min     0.0    1.000000     4.000000  ...   0.0       0.000000     0.000000
25%     0.0    1.000000     7.250000  ...   0.0       3.000000     0.090909
50%     0.0    1.500000    13.000000  ...   0.0       5.000000     0.142857
75%     0.0    9.000000    23.750000  ...   0.0       9.000000     0.200000
max     0.0  107.000000    48.000000  ...   0.0      55.000000     0.818182

[8 rows x 8 columns]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(z.loc[z['Spam']==0].describe())
       Spam    Capitals  Punctuation  ...  Test  Words_NoPunct      Punct
count  10.0   10.000000    10.000000  ...  10.0      10.000000  10.000000
mean    0.0   14.400000    18.300000  ...   0.0       5.600000   0.152905
std     0.0   32.948445    14.772723  ...   0.0       8.058122   0.063198
min     0.0    1.000000     4.000000  ...   0.0       1.000000   0.000000
25%     0.0    1.000000     7.250000  ...   0.0       1.250000   0.132775
50%     0.0    1.500000    13.000000  ...   0.0       2.000000   0.177083
75%     0.0    9.000000    23.750000  ...   0.0       6.750000   0.196875
max     0.0  107.000000    48.000000  ...   0.0      27.000000   0.208333

[8 rows x 8 columns]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(z.loc[z['Spam']==1].describe())
       Spam  Capitals  Punctuation  Length  Words  Test  Words_NoPunct  Punct
count   0.0       0.0          0.0     0.0    0.0   0.0            0.0    0.0
mean    NaN       NaN          NaN     NaN    NaN   NaN            NaN    NaN
std     NaN       NaN          NaN     NaN    NaN   NaN            NaN    NaN
min     NaN       NaN          NaN     NaN    NaN   NaN            NaN    NaN
25%     NaN       NaN          NaN     NaN    NaN   NaN            NaN    NaN
50%     NaN       NaN          NaN     NaN    NaN   NaN            NaN    NaN
75%     NaN       NaN          NaN     NaN    NaN   NaN            NaN    NaN
max     NaN       NaN          NaN     NaN    NaN   NaN            NaN    NaN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(ab.describe())
       Words_NoPunct      Punct
count      10.000000  10.000000
mean        5.600000   0.152905
std         8.058122   0.063198
min         1.000000   0.000000
25%         1.250000   0.132775
50%         2.000000   0.177083
75%         6.750000   0.196875
max        27.000000   0.208333
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Lemmatization
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(lemmas)
stemming/NOUN be/AUX aim/VERB at/SCONJ reduce/VERB vocabulary/NOUN and/CCONJ aid/NOUN un/NOUN -/PUNCT derstanding/NOUN of/ADP morphological/ADJ process/NOUN ./PUNCT 
this/PRON help/VERB people/NOUN un/NOUN -/PUNCT derstand/VERB the/DET morphology/NOUN of/ADP word/NOUN and/CCONJ reduce/VERB size/NOUN of/ADP corpus/NOUN ./PUNCT 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# TF-IDF Based Model
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Count Vectorization
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(vectorizer.get_feature_names())
['an', 'apple', 'away', 'bananas', 'but', 'day', 'doctor', 'eat', 'fruits', 'keeps', 'like', 'love', 'the']
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(X.toarray())
[[0 0 0 1 0 0 0 0 2 0 2 0 0]
 [1 1 0 1 1 0 0 1 0 0 0 1 0]
 [1 1 1 0 0 1 1 0 0 1 0 0 1]]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(cosine_similarity(X.toarray()))
[[1.         0.13608276 0.        ]
 [0.13608276 1.         0.3086067 ]
 [0.         0.3086067  1.        ]]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(cosine_similarity(X, query))
[[0.23570226]
 [0.57735027]
 [0.26726124]]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# TF-IDF Vectorization
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(tfidf.toarray())
         an     apple      away  ...      like      love       the
0  0.000000  0.000000  0.000000  ...  0.688081  0.000000  0.000000
1  0.321267  0.321267  0.000000  ...  0.000000  0.479709  0.000000
2  0.275785  0.275785  0.411797  ...  0.000000  0.000000  0.411797

[3 rows x 13 columns]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(X.shape)
(4459, 7741)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model2.fit(X.toarray(), y_train, epochs=10, batch_size=10)
/home/cwm/git/git.c-w-m/nlp_dev/src/anlp-tf2/.tox/anlp37/lib/python3.7/site-packages/sklearn/utils/validation.py:63: 

DataConversionWarning: A column-vector y was passed when a 1d array was expected. 
Please change the shape of y to (n_samples, ), 
for example using ravel().
  return f(*args, **kwargs)

Epoch 1/10
446/446 [==============================] - 1s 1ms/step - loss: 0.4852 - accuracy: 0.8803
Epoch 2/10
446/446 [==============================] - 1s 1ms/step - loss: 0.1275 - accuracy: 0.9693
Epoch 3/10
446/446 [==============================] - 1s 1ms/step - loss: 0.0533 - accuracy: 0.9871
Epoch 4/10
446/446 [==============================] - 1s 2ms/step - loss: 0.0346 - accuracy: 0.9924
Epoch 5/10
446/446 [==============================] - 1s 1ms/step - loss: 0.0206 - accuracy: 0.9962
Epoch 6/10
446/446 [==============================] - 1s 1ms/step - loss: 0.0123 - accuracy: 0.9986
Epoch 7/10
446/446 [==============================] - 1s 1ms/step - loss: 0.0075 - accuracy: 0.9992
Epoch 8/10
446/446 [==============================] - 1s 1ms/step - loss: 0.0060 - accuracy: 0.9989
Epoch 9/10
446/446 [==============================] - 1s 1ms/step - loss: 0.0041 - accuracy: 0.9997
Epoch 10/10
446/446 [==============================] - 1s 2ms/step - loss: 0.0032 - accuracy: 1.0000
35/35 [==============================] - 0s 1ms/step - loss: 0.0577 - accuracy: 0.9830
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(train.loc[train.Spam == 1].describe())
        Spam    Capitals  Punctuation  ...   Test  Words_NoPunct       Punct
count  592.0  592.000000   592.000000  ...  592.0     592.000000  592.000000
mean     1.0   15.320946    29.086149  ...    0.0      13.972973    0.138660
std      0.0   11.635105     7.083572  ...    0.0       4.546724    0.065528
min      1.0    0.000000     2.000000  ...    0.0       1.000000    0.000000
25%      1.0    7.000000    26.000000  ...    0.0      11.000000    0.096774
50%      1.0   14.000000    30.000000  ...    0.0      14.000000    0.133333
75%      1.0   21.000000    34.000000  ...    0.0      17.000000    0.178571
max      1.0  128.000000    49.000000  ...    0.0      25.000000    0.500000

[8 rows x 8 columns]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# Word Vectors
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(api.info())
{'corpora': {'semeval-2016-2017-task3-subtaskBC': {'num_records': -1, 'record_format': 'dict', 'file_size': 6344358, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/semeval-2016-2017-task3-subtaskB-eng/__init__.py', 'license': 'All files released for the task are free for general research use', 'fields': {'2016-train': ['...'], '2016-dev': ['...'], '2017-test': ['...'], '2016-test': ['...']}, 'description': 'SemEval 2016 / 2017 Task 3 Subtask B and C datasets contain train+development (317 original questions, 3,169 related questions, and 31,690 comments), and test datasets in English. The description of the tasks and the collected data is given in sections 3 and 4.1 of the task paper http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-report.pdf linked in section “Papers” of https://github.com/RaRe-Technologies/gensim-data/issues/18.', 'checksum': '701ea67acd82e75f95e1d8e62fb0ad29', 'file_name': 'semeval-2016-2017-task3-subtaskBC.gz', 'read_more': ['http://alt.qcri.org/semeval2017/task3/', 'http://alt.qcri.org/semeval2017/task3/data/uploads/semeval2017-task3.pdf', 'https://github.com/RaRe-Technologies/gensim-data/issues/18', 'https://github.com/Witiko/semeval-2016_2017-task3-subtaskB-english'], 'parts': 1}, 'semeval-2016-2017-task3-subtaskA-unannotated': {'num_records': 189941, 'record_format': 'dict', 'file_size': 234373151, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/semeval-2016-2017-task3-subtaskA-unannotated-eng/__init__.py', 'license': 'These datasets are free for general research use.', 'fields': {'THREAD_SEQUENCE': '', 'RelQuestion': {'RELQ_CATEGORY': 'question category, according to the Qatar Living taxonomy', 'RELQ_DATE': 'date of posting', 'RELQ_ID': 'question indentifier', 'RELQ_USERID': 'identifier of the user asking the question', 'RELQ_USERNAME': 'name of the user asking the question', 'RelQBody': 'body of question', 'RelQSubject': 'subject of question'}, 'RelComments': [{'RelCText': 'text of answer', 'RELC_USERID': 'identifier of the user posting the comment', 'RELC_ID': 'comment identifier', 'RELC_USERNAME': 'name of the user posting the comment', 'RELC_DATE': 'date of posting'}]}, 'description': 'SemEval 2016 / 2017 Task 3 Subtask A unannotated dataset contains 189,941 questions and 1,894,456 comments in English collected from the Community Question Answering (CQA) web forum of Qatar Living. These can be used as a corpus for language modelling.', 'checksum': '2de0e2f2c4f91c66ae4fcf58d50ba816', 'file_name': 'semeval-2016-2017-task3-subtaskA-unannotated.gz', 'read_more': ['http://alt.qcri.org/semeval2016/task3/', 'http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-report.pdf', 'https://github.com/RaRe-Technologies/gensim-data/issues/18', 'https://github.com/Witiko/semeval-2016_2017-task3-subtaskA-unannotated-english'], 'parts': 1}, 'patent-2017': {'num_records': 353197, 'record_format': 'dict', 'file_size': 3087262469, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/patent-2017/__init__.py', 'license': 'not found', 'description': "Patent Grant Full Text. Contains the full text including tables, sequence data and 'in-line' mathematical expressions of each patent grant issued in 2017.", 'checksum-0': '818501f0b9af62d3b88294d86d509f8f', 'checksum-1': '66c05635c1d3c7a19b4a335829d09ffa', 'file_name': 'patent-2017.gz', 'read_more': ['http://patents.reedtech.com/pgrbft.php'], 'parts': 2}, 'quora-duplicate-questions': {'num_records': 404290, 'record_format': 'dict', 'file_size': 21684784, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/quora-duplicate-questions/__init__.py', 'license': 'probably https://www.quora.com/about/tos', 'fields': {'question1': 'the full text of each question', 'question2': 'the full text of each question', 'qid1': 'unique ids of each question', 'qid2': 'unique ids of each question', 'id': 'the id of a training set question pair', 'is_duplicate': 'the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise'}, 'description': 'Over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line contains a duplicate pair or not.', 'checksum': 'd7cfa7fbc6e2ec71ab74c495586c6365', 'file_name': 'quora-duplicate-questions.gz', 'read_more': ['https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs'], 'parts': 1}, 'wiki-english-20171001': {'num_records': 4924894, 'record_format': 'dict', 'file_size': 6516051717, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/wiki-english-20171001/__init__.py', 'license': 'https://dumps.wikimedia.org/legal.html', 'fields': {'section_texts': 'list of body of sections', 'section_titles': 'list of titles of sections', 'title': 'Title of wiki article'}, 'description': 'Extracted Wikipedia dump from October 2017. Produced by `python -m gensim.scripts.segment_wiki -f enwiki-20171001-pages-articles.xml.bz2 -o wiki-en.gz`', 'checksum-0': 'a7d7d7fd41ea7e2d7fa32ec1bb640d71', 'checksum-1': 'b2683e3356ffbca3b6c2dca6e9801f9f', 'checksum-2': 'c5cde2a9ae77b3c4ebce804f6df542c2', 'checksum-3': '00b71144ed5e3aeeb885de84f7452b81', 'file_name': 'wiki-english-20171001.gz', 'read_more': ['https://dumps.wikimedia.org/enwiki/20171001/'], 'parts': 4}, 'text8': {'num_records': 1701, 'record_format': 'list of str (tokens)', 'file_size': 33182058, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/__init__.py', 'license': 'not found', 'description': 'First 100,000,000 bytes of plain text from Wikipedia. Used for testing purposes; see wiki-english-* for proper full Wikipedia datasets.', 'checksum': '68799af40b6bda07dfa47a32612e5364', 'file_name': 'text8.gz', 'read_more': ['http://mattmahoney.net/dc/textdata.html'], 'parts': 1}, 'fake-news': {'num_records': 12999, 'record_format': 'dict', 'file_size': 20102776, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/fake-news/__init__.py', 'license': 'https://creativecommons.org/publicdomain/zero/1.0/', 'fields': {'crawled': 'date the story was archived', 'ord_in_thread': '', 'published': 'date published', 'participants_count': 'number of participants', 'shares': 'number of Facebook shares', 'replies_count': 'number of replies', 'main_img_url': 'image from story', 'spam_score': 'data from webhose.io', 'uuid': 'unique identifier', 'language': 'data from webhose.io', 'title': 'title of story', 'country': 'data from webhose.io', 'domain_rank': 'data from webhose.io', 'author': 'author of story', 'comments': 'number of Facebook comments', 'site_url': 'site URL from BS detector', 'text': 'text of story', 'thread_title': '', 'type': 'type of website (label from BS detector)', 'likes': 'number of Facebook likes'}, 'description': "News dataset, contains text and metadata from 244 websites and represents 12,999 posts in total from a specific window of 30 days. The data was pulled using the webhose.io API, and because it's coming from their crawler, not all websites identified by their BS Detector are present in this dataset. Data sources that were missing a label were simply assigned a label of 'bs'. There are (ostensibly) no genuine, reliable, or trustworthy news sources represented in this dataset (so far), so don't trust anything you read.", 'checksum': '5e64e942df13219465927f92dcefd5fe', 'file_name': 'fake-news.gz', 'read_more': ['https://www.kaggle.com/mrisdal/fake-news'], 'parts': 1}, '20-newsgroups': {'num_records': 18846, 'record_format': 'dict', 'file_size': 14483581, 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/20-newsgroups/__init__.py', 'license': 'not found', 'fields': {'topic': 'name of topic (20 variant of possible values)', 'set': "marker of original split (possible values 'train' and 'test')", 'data': '', 'id': 'original id inferred from folder name'}, 'description': 'The notorious collection of approximately 20,000 newsgroup posts, partitioned (nearly) evenly across 20 different newsgroups.', 'checksum': 'c92fd4f6640a86d5ba89eaad818a9891', 'file_name': '20-newsgroups.gz', 'read_more': ['http://qwone.com/~jason/20Newsgroups/'], 'parts': 1}, '__testing_matrix-synopsis': {'description': '[THIS IS ONLY FOR TESTING] Synopsis of the movie matrix.', 'checksum': '1767ac93a089b43899d54944b07d9dc5', 'file_name': '__testing_matrix-synopsis.gz', 'read_more': ['http://www.imdb.com/title/tt0133093/plotsummary?ref_=ttpl_pl_syn#synopsis'], 'parts': 1}, '__testing_multipart-matrix-synopsis': {'description': '[THIS IS ONLY FOR TESTING] Synopsis of the movie matrix.', 'checksum-0': 'c8b0c7d8cf562b1b632c262a173ac338', 'checksum-1': '5ff7fc6818e9a5d9bc1cf12c35ed8b96', 'checksum-2': '966db9d274d125beaac7987202076cba', 'file_name': '__testing_multipart-matrix-synopsis.gz', 'read_more': ['http://www.imdb.com/title/tt0133093/plotsummary?ref_=ttpl_pl_syn#synopsis'], 'parts': 3}}, 'models': {'fasttext-wiki-news-subwords-300': {'num_records': 999999, 'file_size': 1005007116, 'base_dataset': 'Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/fasttext-wiki-news-subwords-300/__init__.py', 'license': 'https://creativecommons.org/licenses/by-sa/3.0/', 'parameters': {'dimension': 300}, 'description': '1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).', 'read_more': ['https://fasttext.cc/docs/en/english-vectors.html', 'https://arxiv.org/abs/1712.09405', 'https://arxiv.org/abs/1607.01759'], 'checksum': 'de2bb3a20c46ce65c9c131e1ad9a77af', 'file_name': 'fasttext-wiki-news-subwords-300.gz', 'parts': 1}, 'conceptnet-numberbatch-17-06-300': {'num_records': 1917247, 'file_size': 1225497562, 'base_dataset': 'ConceptNet, word2vec, GloVe, and OpenSubtitles 2016', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/conceptnet-numberbatch-17-06-300/__init__.py', 'license': 'https://github.com/commonsense/conceptnet-numberbatch/blob/master/LICENSE.txt', 'parameters': {'dimension': 300}, 'description': 'ConceptNet Numberbatch consists of state-of-the-art semantic vectors (also known as word embeddings) that can be used directly as a representation of word meanings or as a starting point for further machine learning. ConceptNet Numberbatch is part of the ConceptNet open data project. ConceptNet provides lots of ways to compute with word meanings, one of which is word embeddings. ConceptNet Numberbatch is a snapshot of just the word embeddings. It is built using an ensemble that combines data from ConceptNet, word2vec, GloVe, and OpenSubtitles 2016, using a variation on retrofitting.', 'read_more': ['http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14972', 'https://github.com/commonsense/conceptnet-numberbatch', 'http://conceptnet.io/'], 'checksum': 'fd642d457adcd0ea94da0cd21b150847', 'file_name': 'conceptnet-numberbatch-17-06-300.gz', 'parts': 1}, 'word2vec-ruscorpora-300': {'num_records': 184973, 'file_size': 208427381, 'base_dataset': 'Russian National Corpus (about 250M words)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-ruscorpora-300/__init__.py', 'license': 'https://creativecommons.org/licenses/by/4.0/deed.en', 'parameters': {'dimension': 300, 'window_size': 10}, 'description': 'Word2vec Continuous Skipgram vectors trained on full Russian National Corpus (about 250M words). The model contains 185K words.', 'preprocessing': 'The corpus was lemmatized and tagged with Universal PoS', 'read_more': ['https://www.academia.edu/24306935/WebVectors_a_Toolkit_for_Building_Web_Interfaces_for_Vector_Semantic_Models', 'http://rusvectores.org/en/', 'https://github.com/RaRe-Technologies/gensim-data/issues/3'], 'checksum': '9bdebdc8ae6d17d20839dd9b5af10bc4', 'file_name': 'word2vec-ruscorpora-300.gz', 'parts': 1}, 'word2vec-google-news-300': {'num_records': 3000000, 'file_size': 1743563840, 'base_dataset': 'Google News (about 100 billion words)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/__init__.py', 'license': 'not found', 'parameters': {'dimension': 300}, 'description': "Pre-trained vectors trained on a part of the Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. The phrases were obtained using a simple data-driven approach described in 'Distributed Representations of Words and Phrases and their Compositionality' (https://code.google.com/archive/p/word2vec/).", 'read_more': ['https://code.google.com/archive/p/word2vec/', 'https://arxiv.org/abs/1301.3781', 'https://arxiv.org/abs/1310.4546', 'https://www.microsoft.com/en-us/research/publication/linguistic-regularities-in-continuous-space-word-representations/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F189726%2Frvecs.pdf'], 'checksum': 'a5e5354d40acb95f9ec66d5977d140ef', 'file_name': 'word2vec-google-news-300.gz', 'parts': 1}, 'glove-wiki-gigaword-50': {'num_records': 400000, 'file_size': 69182535, 'base_dataset': 'Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 50}, 'description': 'Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-50.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': 'c289bc5d7f2f02c6dc9f2f9b67641813', 'file_name': 'glove-wiki-gigaword-50.gz', 'parts': 1}, 'glove-wiki-gigaword-100': {'num_records': 400000, 'file_size': 134300434, 'base_dataset': 'Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 100}, 'description': 'Pre-trained vectors based on Wikipedia 2014 + Gigaword 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-100.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': '40ec481866001177b8cd4cb0df92924f', 'file_name': 'glove-wiki-gigaword-100.gz', 'parts': 1}, 'glove-wiki-gigaword-200': {'num_records': 400000, 'file_size': 264336934, 'base_dataset': 'Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-200/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 200}, 'description': 'Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-200.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': '59652db361b7a87ee73834a6c391dfc1', 'file_name': 'glove-wiki-gigaword-200.gz', 'parts': 1}, 'glove-wiki-gigaword-300': {'num_records': 400000, 'file_size': 394362229, 'base_dataset': 'Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 300}, 'description': 'Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-300.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': '29e9329ac2241937d55b852e8284e89b', 'file_name': 'glove-wiki-gigaword-300.gz', 'parts': 1}, 'glove-twitter-25': {'num_records': 1193514, 'file_size': 109885004, 'base_dataset': 'Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 25}, 'description': 'Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/).', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-25.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': '50db0211d7e7a2dcd362c6b774762793', 'file_name': 'glove-twitter-25.gz', 'parts': 1}, 'glove-twitter-50': {'num_records': 1193514, 'file_size': 209216938, 'base_dataset': 'Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-50/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 50}, 'description': 'Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/)', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-50.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': 'c168f18641f8c8a00fe30984c4799b2b', 'file_name': 'glove-twitter-50.gz', 'parts': 1}, 'glove-twitter-100': {'num_records': 1193514, 'file_size': 405932991, 'base_dataset': 'Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 100}, 'description': 'Pre-trained vectors based on  2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/)', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-100.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': 'b04f7bed38756d64cf55b58ce7e97b15', 'file_name': 'glove-twitter-100.gz', 'parts': 1}, 'glove-twitter-200': {'num_records': 1193514, 'file_size': 795373100, 'base_dataset': 'Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)', 'reader_code': 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-200/__init__.py', 'license': 'http://opendatacommons.org/licenses/pddl/', 'parameters': {'dimension': 200}, 'description': 'Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/).', 'preprocessing': 'Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-200.txt`.', 'read_more': ['https://nlp.stanford.edu/projects/glove/', 'https://nlp.stanford.edu/pubs/glove.pdf'], 'checksum': 'e52e8392d1860b95d5308a525817d8f9', 'file_name': 'glove-twitter-200.gz', 'parts': 1}, '__testing_word2vec-matrix-synopsis': {'description': '[THIS IS ONLY FOR TESTING] Word vecrors of the movie matrix.', 'parameters': {'dimensions': 50}, 'preprocessing': 'Converted to w2v using a preprocessed corpus. Converted to w2v format with `python3.5 -m gensim.models.word2vec -train <input_filename> -iter 50 -output <output_filename>`.', 'read_more': [], 'checksum': '534dcb8b56a360977a269b7bfc62d124', 'file_name': '__testing_word2vec-matrix-synopsis.gz', 'parts': 1}}}
[==================================================] 100.0% 1662.8/1662.8MB downloaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(model_w2v.most_similar('cookies',topn=10))
/home/cwm/git/git.c-w-m/nlp_dev/src/anlp-tf2/.tox/anlp37/lib/python3.7/site-packages/gensim/models/keyedvectors.py:877: 

FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. 

Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.

  vectors = vstack(self.word_vec(word, use_norm=True) for word in used_words).astype(REAL)

[('cookie', 0.745154082775116), 
('oatmeal_raisin_cookies', 0.6887780427932739), 
('oatmeal_cookies', 0.662139892578125), 
('cookie_dough_ice_cream', 0.6520504951477051), 
('brownies', 0.6479344964027405), 
('homemade_cookies', 0.6476464867591858), 
('gingerbread_cookies', 0.6461867690086365), 
('Cookies', 0.6341644525527954), 
('cookies_cupcakes', 0.6275068521499634), 
('cupcakes', 0.6258294582366943)]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(model_w2v.doesnt_match(['USA','Canada','India','Tokyo']))
Tokyo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(model_w2v.similar_by_vector(queen))
[('king', 0.8449392318725586), 
('queen', 0.7300517559051514), 
('monarch', 0.6454660892486572), 
('princess', 0.6156251430511475), 
('crown_prince', 0.5818676948547363), 
('prince', 0.5777117609977722), 
('kings', 0.5613663792610168), 
('sultan', 0.5376776456832886), 
('Queen_Consort', 0.5344247817993164), 
('queens', 0.5289887189865112)]


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
# -- end of 'SMS_Spam_Detection.py' --
```shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process finished with exit code 0
```