import pickle

def test_preservation():
    filename="sauerkraut"

    recipe = { 'Cabbages': 3, 'Salt': 8, 'Wine': 0.1}

    outfile = open(filename,'wb')
    pickle.dump(recipe,outfile)
    outfile.close()

    infile = open(filename,'rb')
    new_recipe = pickle.load(infile)
    infile.close()

    assert recipe == new_recipe
