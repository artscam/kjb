"""
Input/Output module.

Functions for reading from text files and printing out results.

"""

def get_verses(file_name):
    """
    Return list of Bible verses

    Parameters
    ----------
    file_name : str
        Name of file to read.
    Returns
    -------
    verses : list of str
        Each element is a verse and includes the passage number.

    """
    verses = []

    # read line by line
    with open(file_name) as bib_file:
        verse = ""
        for line in bib_file:
            if line.strip():
                if ":" in line.split()[0]:
                    verses.append(verse)
                    verse = line
                else:
                    verse += line
        verses.append(verse)
    verses.pop(0)

    return verses




