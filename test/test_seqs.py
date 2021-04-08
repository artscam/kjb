import kjb.src.LSTM_functions as lf

def test_sequences():
    processed_inputs = "like apples very much"
    X, y, chars = lf.define_sequences(processed_inputs)

    #print([X, y, chars])
    #assert X == True
