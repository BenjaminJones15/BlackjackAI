
"""
@author: Zaffer
"""
import random
def displayText():
    print("Geeks")

def bet(Probability):
    if Probability > 0.5:
        return 100
    else:
        return 10
def play():
    picker = random.choice([1,2])
    if picker == 1:
        return 'h'
    elif picker == 2:
        return 's'
    return "q"
    