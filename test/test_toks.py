import kjb.src.LSTM_functions as lf

def test_tokenise():
    test_string = "I like Apples"
    expected = "like apples"
    processed_inputs = lf.tokenise_words(test_string)

    #print(processed_inputs)
    assert processed_inputs == expected
