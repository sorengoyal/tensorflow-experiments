"""
Script to create a simple linearly separable dataset
"""

from random import random
f = open('linearData.csv', 'w')
count = 0
while count < 500:
    x = random()
    y = random()
    if(y > x):
        f.write(repr(x) +"," +repr(y) + ",1\n")
        count = count + 1
count = 0
while count < 500:
    x = random()
    y = random()
    if(y < x):
        f.write(repr(x) +"," +repr(y) + ",-1\n")
        count = count + 1
f.close()

