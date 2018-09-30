import csv
import numpy
import math
import inspect1
import matplotlib.pyplot as plt

class Node:
	def __init__(self,attribute=None,rules=None,distrib=None,depth=None,parent=None,value=None):
			self.attribute = attribute
			self.rules = {}
			self.distrib = {}
			self.depth = 0
			self.parent = None
			self.value = ""
			
			
	def add_rule(self,key,value):
		self.rules[key] = value
	
	def set_depth(self,node_depth):
		self.depth = node_depth
		
	def get_depth(self):
		return self.depth
	
	def set_probas(self,probas):
		self.distrib = probas
	
	def get_probas(self):	
		return self.distrib	

	def set_value(self,value):
		self.value = value
	
	def get_value(self):	
		return self.value	
		
	def get_rules(self):		
		return self.rules
			
	def get_attribute(self):		
		return self.attribute

	def set_parent(self,parent):
		self.parent = parent	
		
	def get_parent(self):
		return self.parent
		
	def get_next_node(self,key):
		if key in self.rules:
			return self.rules.get(key)
		else:
			return None
	
		
def print_tree(tree,header):
	for node in tree:
		i=0
		branch = ""
		while i < node.get_depth():
			branch = branch + "| "
			i+=1
		if node.depth == 0:
			print(branch, " = ", node.get_probas()) 
		else:
			print(branch, "", header[node.get_parent().get_attribute()] ," = ", node.get_value(),"", node.get_probas()) 


def attribute_to_split(mutual_info):
	i = 0
	max = 0
	index = 0
	for val in mutual_info:
		if val > max:
			max = val
			index = i
		i+=1
	return index
	

def split(data, index, value):
	list_indexes = []
	for i_line, line in enumerate(data):
		if line[index] == value:
			list_indexes.append(i_line)
	new_data = data[list_indexes, :]
	return new_data

def stump(dataset, index_to_split, local_depth, parent_node, maximum_depth, node_list):
	list_mutual_info = []
	length = len(dataset[0])
	if index_to_split == -1:
		col_y = dataset[:,length-1]
		probs = inspect1.calc_count(col_y)
		list_mutual_info = inspect1.calc_all_entropy(data)
		index = attribute_to_split(list_mutual_info)
		node = Node(index)
		node.set_probas(probs)
		node_list.append(node)
		node_list = stump(dataset,index,local_depth,node,maximum_depth,node_list)
	else:	
		local_depth+= 1
		val_in_col = inspect1.values_in_col(dataset[:,index_to_split])
		for value in val_in_col:
			new_dataset =  split(dataset, index_to_split, value)
			#print(new_dataset)
			list_mutual_info = inspect1.calc_all_entropy(new_dataset)
			index = attribute_to_split(list_mutual_info)
			col_y = new_dataset[:,length-1]
			probs = inspect1.calc_count(col_y)
			node = Node(index)
			node.set_value(value)
			node.set_parent(parent_node)
			parent_node.add_rule(value, node)
			node.set_probas(probs)
			node.set_depth(local_depth)
			node_list.append(node)
			#print("max_depth", maximum_depth)
			if list_mutual_info[index] > 0 and local_depth < maximum_depth:
				node_list = stump(new_dataset, index,local_depth,node,maximum_depth,node_list)
	return node_list

def predict(line, node, header,maximum_depth):
	#print(maximum_depth)
	if len(node.get_rules().keys()) == 0 or maximum_depth == 0:
		prediction = inspect1.majority_class(node.get_probas())
		return prediction
	else:
		#print("Ma valeur de ", header[node.get_attribute()], " est ", line[node.get_attribute()])
		next_node = node.get_next_node(line[node.get_attribute()])
		prediction = predict(line,next_node,header,maximum_depth)
		return prediction

dataset_path = ".\\handout\\education_train.csv"
to_split = -1
max_depth = 0
local_depth = 0
tree = []
node=Node()
node_list=[]	
number = 0
table_error_test = []
table_error_train = []
node_list = []	

data = inspect1.loadCSV(dataset_path)
#print(data)
num_iterations = len(data[0])
header = inspect1.get_header(dataset_path)
	
while max_depth < num_iterations:
	tree = stump(data,to_split,local_depth,node,max_depth,node_list)
	#print_tree(tree,header)
	#print(tree[0].get_probas())
	node_list=[]
	sum = 0
	sum_wrong = 0
	test_path = ".\\handout\\education_test.csv"
	data_test = inspect1.loadCSV(test_path)
	error_rate = 0
	for i_line, line in enumerate(data_test):
		prediction_value = predict(line, tree[0],header,max_depth)
		if prediction_value != line[len(line)-1]:
			sum_wrong += 1
		sum+=1
	error_rate = sum_wrong/sum
	print("Test error_rate pour max_depth = ", max_depth," : ", error_rate)
	table_error_test.append(error_rate)
	
	node_list=[]
	sum = 0
	sum_wrong = 0
	test_path = ".\\handout\\education_train.csv"
	data_test = inspect1.loadCSV(test_path)
	error_rate = 0
	for i_line, line in enumerate(data_test):
		prediction_value = predict(line, tree[0],header,max_depth)
		if prediction_value != line[len(line)-1]:
			sum_wrong += 1
		sum+=1
	error_rate = sum_wrong/sum
	print("Train error_rate pour max_depth = ", max_depth," : ", error_rate)
	table_error_train.append(error_rate)
	
	max_depth += 1


plt.plot(table_error_train,label='Train error')
plt.plot(table_error_test,label='Test error')
plt.xlabel("Depth")
plt.ylabel("Error rate")
plt.title("Errors for training and test sets")
plt.ylim([0,0.55])
plt.legend()
plt.show()
