import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#file = open("full_bible.txt").read()

def tokenise_words(input_text):
    """
    Return standardised filtered text that can be fed into the NN

    Parameters
    ----------
    input_text : str
        Text file training data
    Returns
    -------
    filtered: str
        contains only words  not in a list of low information common words, or 'stop words'

    """
    #convert to lower case
    input_text = input_text.lower()

    #instantiate the tokenise_words
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(input_text)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'),tokens)
    return " ".join(filtered)


def define_sequences(processed_inputs,seq_length):
    """
    Convert pre-processed text into numerical arrray form for use in the NN

    Parameters
    ----------
    processed_inputs : str
        pre-filtered training data
    seq_length : num
        the length of each sequence
    Returns
    -------
    X : num
        array of numerical values representing the Text
    y : num
        label data
    char : str array
        the key to convert back to characters later

    """
    chars = sorted(list(set(processed_inputs)))
    char_to_num = dict((c,i) for i, c in enumerate(chars))

    input_len = len(processed_inputs)
    vocab_len = len(chars)

    x_data = []
    y_data = []

    # Loop through inputs, start at the beginning and go until we hit
    # the final character we can create a sequence out of
    for i in range(0,input_len - seq_length, 1):
        # Define input and output sequences
        # Input is the current character plus desired sequence seq_length
        in_seq = processed_inputs[i:i+seq_length]

        # Out sequence is the initial character plus total sequence length
        out_seq = processed_inputs[i+seq_length]

        # We now convert list of characters to integers based on the dictionary
        # and add the values to our lists
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append([char_to_num[out_seq]])

    n_patterns = len(x_data)

    X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(vocab_len)

    y = np_utils.to_categorical(y_data)

    return X, y, chars

def lstm_model(X, y, chars):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Save the weights in a checkpoint to skip recalculation
    filepath = "model_weights_saved.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    desired_callbacks = [checkpoint]

    # Convert output of model back into characters
    num_to_char = dict((i, c) for i, c in enumerate(chars))

    start = numpy.random.randint(0, len(X) - 1)
    pattern = X[start,0]
    print("Random Seed:")
    print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

    return ''
