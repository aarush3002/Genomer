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
print(human_texts)
#transforming the 'words' column into its own list and storing it in human_texts
for item in range(len(human_texts)): #each index of this list represents the output of getKmers for the sequence
    human_texts[item] = ' '.join(human_texts[item]) #since this is a 2D list, we are just connecting each list into a string separated with spaces
y_h = human.iloc[:, 0].values  
#y_h contains all the class numbers of each sequence of DNA in the entire file
#it will be a list of numbers as so: [4 4 3 ... 6 6 6]

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
human['class'].value_counts().sort_index().plot.bar()
chimp['class'].value_counts().sort_index().plot.bar()
dog['class'].value_counts().sort_index().plot.bar()

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

Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting
(and optionally subsampling) data in a oneliner.
'''

print(X_train.shape)
print(X_test.shape)

classifier = MultinomialNB(alpha=0.1)
'''
Naive Bayes classifier for multinomial models

The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial 
distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.
'''
classifier.fit(X_train, y_train)
print(classifier)
y_pred = classifier.predict(X_test)
'''
given a trained model, predict the label of a new set of data. This method accepts one argument, the new data X_new (e.g. model.predict(X_new)),
and returns the learned label for each object in the array.
'''

print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
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
'''from __future__ import with_statement
import sys
import os
import operator

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from BCBio import GFF

def main(glimmer_file, ref_file):
    with open(ref_file) as in_handle:
        ref_recs = SeqIO.to_dict(SeqIO.parse(in_handle, "fasta"))

    base, ext = os.path.splitext(glimmer_file)
    out_file = "%s-proteins.fa" % base
    with open(out_file, "w") as out_handle:
        SeqIO.write(protein_recs(glimmer_file, ref_recs), out_handle, "fasta")

def glimmer_predictions(in_handle, ref_recs):
    """Parse Glimmer output, generating SeqRecord and SeqFeatures for predictions
    """
    for rec in GFF.parse(in_handle, target_lines=1000, base_dict=ref_recs):
        yield rec

def protein_recs(glimmer_file, ref_recs):
    """Generate protein records from GlimmerHMM gene predictions.
    """
    with open(glimmer_file) as in_handle:
        for rec in glimmer_predictions(in_handle, ref_recs):
            for feature in rec.features:
                seq_exons = []
                for cds in feature.sub_features:
                    seq_exons.append(rec.seq[
                        cds.location.nofuzzy_start:
                        cds.location.nofuzzy_end])
                gene_seq = reduce(operator.add, seq_exons)
                if feature.strand == -1:
                    gene_seq = gene_seq.reverse_complement()
                protein_seq = gene_seq.translate()
                yield SeqRecord(protein_seq, feature.qualifiers["ID"][0], "", "")
'''