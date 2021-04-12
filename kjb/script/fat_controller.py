import kjb.src.LSTM_functions as lf
import pickle  # This package is used to store variables
import os

skip_fit = True # Skip weights calculation by setting to True

# dirname= os.path.dirname("full_bible.txt")
# sourcename = os.path.join(dirname,"../../test/data/full_bible.txt")
sourcename = ("bible.txt")
input_text = open(sourcename).read()

filename = 'globalsave.pkl' # where to save model model weights
filename2="stored_variables" # and the corresponding variables
seq_length = 100 # length of individual sequence must be specified
eps = 20 # epochs for model.fit

processed_inputs = lf.tokenise_words(input_text)

X, y, chars, x_data = lf.define_sequences(processed_inputs,seq_length)

if skip_fit == False:

    lf.lstm_model(X, y, eps)

    # Store X, y, and chars so they can be loaded from previous calculations
    outfile = open(filename2,'wb')
    pickle.dump([X, y, chars],outfile)
    outfile.close()

else:
    # load the session again:
    infile = open(filename2,'rb')
    [X, y, chars] = pickle.load(infile)
    infile.close()

blasphemy = lf.write_passage(X, y, chars, seq_length, x_data)

print(blasphemy)
