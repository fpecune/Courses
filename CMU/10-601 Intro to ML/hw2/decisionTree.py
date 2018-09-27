import csv
import numpy
import math
import inspect

dataset_path = ".\\handout\\small_train.csv"

data = inspect.loadCSV(dataset_path)
print(data)	
inspect.calc_all_entropy(data)