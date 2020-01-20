from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#the above lines import classes and packages from different libraries

human = pd.read_table('C:/Users/aarus/Documents/CSA/BioPython/human_data.txt') 
chimp = pd.read_table('C:/Users/aarus/Documents/CSA/BioPython/chimp_data.txt')
dog = pd.read_table('C:/Users/aarus/Documents/CSA/BioPython/dog_data.txt')

#this line will use the pandas library in order to reorganize the data in the text file
#it will use the following format:
'''
                                            sequence  class
0  ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCA...      4
1  ATGAACGAAAATCTGTTCGCTTCATTCATTGCCCCCACAATCCTAG...      4
2  ATGTGTGGCATTTGGGCGCTGTTTGGCAGTGATGATTGCCTTTCTG...      3
3  ATGTGTGGCATTTGGGCGCTGTTTGGCAGTGATGATTGCCTTTCTG...      3
4  ATGCAACAGCATTTTGAATTTGAATACCAGACCAAAGTGGATGGTG...      3
...

where each row represents all the nucleotides in one gene (under the sequence header)
and also has the class of the gene (0-6)
'''
'''
print(human.head())
print(chimp.head())
print(dog.head())
'''
#the .head() function will print the first 5 rows of the pandas table

'''
the following function will collect all possible overlapping k-mers of a specified 
length from any sequence string.

For example:
ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCA

will become:
['ATGCCC', 'TGCCCC', 'GCCCCA', 'CCCCAA', ....]
'''
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
dog['words'] = dog.apply(lambda x: getKmers(x['sequence']), axis=1)
#the .apply() function from pandas will apply a function to every single
#value in a pandas series. (the function in questions is getKmers).
#we now have a new column in the pandas table named 'words' and each row
#will now contain a list of all possible k-mers from the sequence.
human = human.drop('sequence', axis=1)
chimp = chimp.drop('sequence', axis=1)
dog = dog.drop('sequence', axis=1)
#there is now no need for the DNA sequences, so we can delete that column
#from the pandas table to be more efficient. 

human_texts = list(human['words'])
#transforming the 'words' column into its own list and storing it in human_texts
for item in range(len(human_texts)): #each index of this list represents the output of getKmers for the sequence
    human_texts[item] = ' '.join(human_texts[item]) #since this is a 2D list, we are just connecting each list into a string separated with spaces
y_h = human.iloc[:, 0].values  
#y_h contains all the class numbers of each sequence of DNA in the entire file
#it will be a list of numbers as so: [4 4 3 ... 6 6 6]
'''
The class labels mean that a sequence will code for a specific protein:

GENE FAMILY                         CLASS LABEL
G protein coupled receptors         0
Tyrosine kinase                     1
Tyrosine phosphatase                2
Synthetase                          3
Synthase                            4
Ion Channel                         5
Transcription Factor                6
'''
#we will be doing the same things for the chimp and dog data
chimp_texts = list(chimp['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
y_c = chimp.iloc[:, 0].values                       # y_c for chimp

dog_texts = list(dog['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
y_d = dog.iloc[:, 0].values                         # y_d for dog

cv = CountVectorizer(ngram_range=(4,4))
#the CountVectorizer function will first find all the unique tetragrams (sequences of 4 "words" (6 nucleotides like 'aattcc') in a row)
#in the sequence. For each sequence, the CountVectorizer function will return a list of how many
#times the given tetragram occurs in the sequence.
'''
For example...
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
'''
X = cv.fit_transform(human_texts)
'''
To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation.

x′=x−μσ
You do that on the training set of data. But then you have to apply the same transformation to your testing set (e.g. in cross-validation), or to newly obtained examples before forecast. But you have to use the same two parameters μ and σ (values) that you used for centering the training set.

Hence, every sklearn's transform's fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state. Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.

fit_transform() joins these two steps and is used for the initial fitting of parameters on the training set x, but it also returns a transformed x′. Internally, it just calls first fit() and then transform() on the same data.
'''
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)

print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)
#will return a tuple that represents the dimensionality of the DataFrame
'''
For example:
(4380, 232414)
'''

'''
print(human['class'].value_counts().sort_index().plot.bar())
print(chimp['class'].value_counts().sort_index().plot.bar())
print(dog['class'].value_counts().sort_index().plot.bar())
'''

'''
.value_counts():
Return a Series containing counts of unique values.

The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.

.sort_index():
Pandas dataframe.sort_index() function sorts objects by labels along the given axis.
Basically the sorting alogirthm is applied on the axis labels rather than the actual data in the dataframe and based on that the data is rearranged. 

.plot:
 Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc.

.bar():
 Make a bar plot.

The bars are positioned at x with the given alignment. Their dimensions are given by width and height. The vertical baseline is bottom (default 0).


'''
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y_h, 
                                                    test_size = 0.20, 
                                                    random_state=42)

'''
Split arrays or matrices into random train and test subsets

For example
x = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
y = [0,1,2,3,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

Since the test_size is 0.33 that means the train_size is 0.67, the function will randomly select (2/3)*5 or roughly 3 elements 
from the list. 

X_train = [[4,5], [0,1], [6,7]]
y_train = [2,0,3]


The data set for the test set will contain only the elements that haven't been selected by the training set yet.
It will randomly order these elements.
X_test = [[8,9], [2,3]]
y_test = [1,4]

In our situation, we can see that the training set will have randomly selected of 80% of the elements
while the testing set will contain the rest of the elements in selected in a random order

the random_state parameter just specifies the seed of the random number generator (takes in int)

'''
print(X_train.shape)
print(X_test.shape)
#print(X_test)

'''
.shape gives the dimensionality of the data frame
'''


classifier = MultinomialNB(alpha=0.1)
'''
Naive Bayes classifier for multinomial models

The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial 
distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
'''
classifier.fit(X_train, y_train)
'''
fit naive bayes classifier according to X, y
'''
y_pred = classifier.predict(X_test) 

'''
given a trained model, predict the label of a new set of data. This method accepts one argument, the new data X_new (e.g. model.predict(X_new)),
and returns the learned label for each object in the array.

The dimensionality of X_test is (3504, 876) so the y_pred prediction will contain 876 elements
'''

print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

#Will construct a pandas table with a row named Predicted, and a column named Acutal
#the Predicted row will have 0-6 for each class label, and the Actual column will have
#the same 0-6 for each class label.
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

'''
The get_metrics function will calculate the accuracy score, precision score, recall score
and f1 score.

The accuracy score is the set of labels predicted for a sample must exactly match the corresponding
set of labels in y_test

The precision score is equal to tp / (tp + fp), where tp = # of true positives, fp = # of false positives
The ability of the classifier to not label as positive a sample that is negative

The recall score is equal to tp / (tp + fn), where tp = # true positives, and fn = # false negatives
This ist he ability of the model to find all positive samples

The f1 score is a weighted average of the precision and recall. 
'''
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

y_pred_chimp = classifier.predict(X_chimp)
y_pred_dog = classifier.predict(X_dog)

print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_c, y_pred_chimp)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_d, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_d, y_pred_dog)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
