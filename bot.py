
import nltk
from nltk.stem.porter import PorterStemmer
from keras.layers import Input
from tflearn.layers.conv import conv_2d, max_pool_2d

stemmer = PorterStemmer()

import numpy
import tflearn
import tensorflow
import keras
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            doc = nltk.word_tokenize(doc)
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(pattern)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()


net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net,learning_rate=0.001, loss='categorical_crossentropy')

model = tflearn.DNN(net)




# training = tensorflow.keras.utils.normalize(training, axis=1)
# output = tensorflow.keras.utils.normalize(output, axis=1)


# model = tensorflow.keras.models.Sequential()
# # model.add(tensorflow.keras.layers.Flatten(input_shape=[len(training[0])])))
# model.add(tensorflow.keras.layers.Input(shape=training.shape[0]))
# model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))
# model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))
# model.add(tensorflow.keras.layers.Dense(6, activation=tensorflow.nn.softmax))


try:
    
    model.load("model.tflearn")
except:
    # model.compile(optimizer= 'adam',
    #                 loss='categorical_crossentropy',
    #                 metrics=['accuracy'])
    # model.fit(training, output, validation_split=0.2, epochs=12) #training = x, output = y
    # model.summary()
    
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    # model.save('model.keras')
    # new_model = tensorflow.keras.models.load_model('model.keras')
    # predictions = new_model.predict(training)
    # print(predictions)
    


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("say hi! : ")
    while True:
        inp = input("YOU: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print("BOT: " + random.choice(responses))

chat()