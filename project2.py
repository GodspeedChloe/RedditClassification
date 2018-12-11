'''
Main file for completing project 2

Author:		Chloe Jackson
Version:	11-Dec-2018
'''

import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer , CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np


'''
	read_csv

	read a csv file into lists for future use
'''
def read_csv(file_name):
	records = []
	with open(file_name) as data:
		# save all the data 
		lines = data.readlines()
		for line in lines:
			values = line.split(',')
			records.append(values)
	return records

'''
	main function
'''
def main():

	rcrypto = read_csv('rcrypto.csv')
	rtwitch = read_csv('twitch.csv')

	posts1 = []
	posts2 = []
	
	for post in rcrypto:
		posts1.append(post[1])

	for post in rtwitch:
		posts2.append(post[1])


	N = len(posts1)
	n2 = N // 2
	n4 = n2 // 2

	# split into 50-25-25 portions
	posts1_train = posts1[:n2]
	posts2_train = posts2[:n2]
	posts1_dev = posts1[n2:n4+n2]
	posts2_dev = posts2[n2:n4+n2]
	posts1_test = posts1[n4+n2:]
	posts2_test = posts2[n4+n2:]

	training_data = posts1_train + posts2_train
	test_data = posts1_test + posts2_test	

	training_targets = []
	for _ in range(0,n2):
		training_targets.append(1)
	for _ in range(0,n2):
		training_targets.append(2)

	test_targets = []
	for _ in posts1_test:
		test_targets.append(1)
	for _ in posts2_test:
		test_targets.append(2)
	
	

	# SVM pipeline
	text_clf_svm = Pipeline([('vect', CountVectorizer()),
							 ('tfidf', TfidfTransformer()),
							 ('clf-svm', SGDClassifier(loss='hinge',
									alpha=1e-3,random_state=42))])

	# train the svm
	_ = text_clf_svm.fit(training_data, training_targets)

	# test with test data
	predicted_svm = text_clf_svm.predict(test_data)
	m = np.mean(predicted_svm == test_targets)

	print ''
	print "Classification success rate for Support Vector Classifier: " + str(m)
	print ''

	# Random forest pipeline
	text_clf_ran = Pipeline([('vect', CountVectorizer()),
							 ('tfidf', TfidfTransformer()),
							 ('clf-ran', RandomForestClassifier(random_state=42))])

	# train the random forest
	_ = text_clf_ran.fit(training_data, training_targets)

	# test with test data
	predicted_ran = text_clf_ran.predict(test_data)
	m = np.mean(predicted_ran == test_targets)

	print ''
	print "Classification success rate for Random Forest Classifier: " + str(m)
	print ''

# send it
main()
