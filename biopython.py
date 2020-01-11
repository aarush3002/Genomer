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

human = pd.read_table('C:/Users/aarus/Documents/CSA/BioPython/human_data.txt')
human.head()

chimp = pd.read_table('C:/Users/aarus/Documents/CSA/BioPython/chimp_data.txt')
dog = pd.read_table('C:/Users/aarus/Documents/CSA/BioPython/dog_data.txt')
chimp.head()
dog.head()

#Image("../input/image2/Capture1.PNG")

def getKmers(sequence, size=12):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
human = human.drop('sequence', axis=1)
chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp = chimp.drop('sequence', axis=1)
dog['words'] = dog.apply(lambda x: getKmers(x['sequence']), axis=1)
dog = dog.drop('sequence', axis=1)

human_texts = list(human['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
y_h = human.iloc[:, 0].values  

chimp_texts = list(chimp['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
y_c = chimp.iloc[:, 0].values                       # y_c for chimp

dog_texts = list(dog['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
y_d = dog.iloc[:, 0].values                         # y_d for dog

cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)

print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)

human['class'].value_counts().sort_index().plot.bar()
chimp['class'].value_counts().sort_index().plot.bar()
dog['class'].value_counts().sort_index().plot.bar()

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y_h, 
                                                    test_size = 0.20, 
                                                    random_state=42)

print(X_train.shape)
print(X_test.shape)

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

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