import math

proba1 = 1/2
proba2 = 1/2

def semiEntro(value):
    semi_entro = value * math.log(value,2)
    return semi_entro

val1 = semiEntro(proba1)
val2 = semiEntro(proba2)

print(str(semiEntro(proba1)))
print(str(semiEntro(proba2)))

entropy = -(val1+val2)

print(str(entropy))
