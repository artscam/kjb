import kjb.src.LSTM_functions as lf
import numpy as np

def test_write_passage():
    seq_length=50
    input_text = "Thomas’s third book, The Map of Love, appeared in August 1939, a month before war officially broke out in Europe. It comprised a strange union of 16 poems and seven stories, the stories having been previously published in periodicals. The volume was a commercial failure, perhaps because of the war. "
    processed_inputs = lf.tokenise_words(input_text)
    X, y, chars, x_data = lf.define_sequences(processed_inputs,seq_length)

    # eps=4
    # lf.lstm_model(X, y, eps)

    blasphemy = lf.write_passage(X, y, chars, seq_length, x_data)
    print(blasphemy)
    assert True
