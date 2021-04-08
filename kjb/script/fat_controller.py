import kjb.src.LSTM_functions as lf

#input_text is a placeholder that can be generalised in the future
input_text = open("bible.txt").read()

processed_inputs = lf.tokenise_words(input_text)

# length of individual sequence must be specified
seq_length = 100
X, y, chars = lf.define_sequences(processed_inputs,seq_length)
print(X[2,10],y[1,10],chars[15])
#lf.lstm_model(X, y, chars)
