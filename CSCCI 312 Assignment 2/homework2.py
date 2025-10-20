import re
import pickle as pkl
import regex

FILE = 'tenn-plates.pkl'
FILE2 = 'windows-pathnames.pkl'


reg_exp_plates = r'\b(?:[1-9]-(?:[A-Z0-9][A-Z0-9]{4}|[A-Z0-9]{2}[A-Z0-9]{3})|(?:[1-8][0-9]|9[0-6])-(?:[A-Z0-9][A-Z0-9]{3}|[A-Z0-9]{2}[A-Z0-9]{2})|[A-Z0-9]{3}-[A-Z0-9]{3})\b'
reg_exp_windows = r"([\U0001F600-\U0001F64F]\?\\\\[ !#$%&'(),-./0-9;=?@A-Z\[\]^_a-z{|}~]*(\\[\u2C00-\u2C5F\u00A2\x07\x08\x09\x0A\x0B\x0C\x0D]*)+)|([\U0001F600-\U0001F64F]\?\\\\([ !#$%&'(),-./0-9;=?@A-Z\[\]^_a-z{|}~]*$))"
RE = {"tenn-plates": reg_exp_plates, "windows-pathnames": reg_exp_windows}
    
pkl.dump(RE, open( "re.pkl" , "wb"))

