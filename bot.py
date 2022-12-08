
"""
@author: Zaffer
"""
import random
def displayText():
    print("Geeks")

def bet():
    return random.randrange(1,50000)
def play():
    picker = random.choice([1,2])
    if picker == 1:
        return 'h'
    elif picker == 2:
        return 's'
    return "q"
    