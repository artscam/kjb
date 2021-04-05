import kjb.src.io as io

def test_get_verses():
    verses = io.get_verses("data/bible.txt")
    assert verses[-1].startswith('1:25')

