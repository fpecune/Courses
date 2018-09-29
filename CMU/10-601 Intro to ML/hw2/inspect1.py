import csv
import numpy
import math


## Transform a .csv file into a matrix.
def loadCSV(filename):
	matrix = numpy.genfromtxt(filename, dtype='str', delimiter=",", skip_header=1)
	return matrix

def get_header(filename):
	matrix = numpy.genfromtxt(filename, dtype='str', delimiter=",", skip_header=0)
	header = matrix[0]
	return header
	
## Takes a column from the dataset as input. 
## For this column, it returns a dictionnary containing, for each possible value, the probability it occurs in the column.
## e.g : {yes: 14/43, no: 26/43, N/A: 3/43} means that the columns contains 14 yes, 26 no, and 3 N/A.
def calc_probas(col):
	dict_probas = {}
	length = len(col)
	for i in col:
		i = str(i)
		if i not in dict_probas.keys():
			dict_probas[i] = 1
		else:
			dict_probas[i] = dict_probas.get(i)+1
	for key,value in dict_probas.items():
		dict_probas[key] = dict_probas.get(key)/length
	return dict_probas

	
def calc_count(col):
	dict_count = {}
	length = len(col)
	for i in col:
		i = str(i)
		if i not in dict_count.keys():
			dict_count[i] = 1
		else:
			dict_count[i] = dict_count.get(i)+1
	return dict_count
	
	
## Takes a dictionnary of probabilities as input (e.g. {yes: 14/43, no: 26/43, N/A: 3/43} and returns the error rate based on majority vote classifier.
def majority_class_error(probas):
	highest_score = 0
	sum=0
	for key,value in probas.items():
		sum = sum+value
		if value > highest_score:
			highest_score = value
	return (1-(highest_score/sum))

def majority_class(probas):
	highest_score = 0
	majority_class = ""
	for key,value in probas.items():
		if value > highest_score:
			majority_class = key
			highest_score = value
	return majority_class

def calc_y_entropy(y):
	length = len(y)
	entropy = 0
	dict_probas = calc_probas(y)
	for key,value in dict_probas.items():
		entropy = entropy + (-(value*math.log(value,2)))
	return entropy
	
def values_in_col(col):
	list_values_in_col = []
	for i in col:
		i = str(i)
		if i not in list_values_in_col:
			list_values_in_col.append(i)
	return list_values_in_col
	
def calc_conditional_entropy(x, y):
	probas_x = calc_probas(x) 
	length = len(x)
	i = 0
	mutual_info = calc_y_entropy(y)
	for values in probas_x:
		dict_probas = {}
		i=0
		total=0
		cond_entropy = 0
		while i < length:
			if str(x[i]) == str(values):
				if y[i] not in dict_probas.keys():
					dict_probas[y[i]] = 1
				else:
					dict_probas[y[i]] = dict_probas.get(y[i])+1
				total+=1 
			i+=1
		for key,value in dict_probas.items():
			dict_probas[key] = dict_probas.get(key)/total
			value = value/total
			cond_entropy = cond_entropy + (-(value*math.log(value,2)))
		cond_entropy = cond_entropy * probas_x.get(values)
		mutual_info = mutual_info - cond_entropy
	return mutual_info
	
def calc_all_entropy(dataset):
	length = len(dataset[0])
	col_y = dataset[:,length-1]
	i=0
	list_mutual_info = []
	while i < (length-1):
		col_x = dataset[:,[i]]
		list_mutual_info.append(calc_conditional_entropy(col_x,col_y)) 
		i+=1
	#print("mutual infos : ", list_mutual_info)
	return list_mutual_info
		