import kjb.src.LSTM_functions as lf
import numpy as np

def test_sequences():
    seq_length=5
    processed_inputs = "like apples very much"
    X, y, chars = lf.define_sequences(processed_inputs,seq_length)

    #check dimensionality of outputs matches
    assert np.shape(X)[0] == np.shape(y)[0] and np.shape(y)[1] == np.shape(chars)[0]
