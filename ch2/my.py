import numpy as np
# import pandas as pd

file_name = 'seeds.tsv'
dataset = np.genfromtxt(file_name, delimiter='\t')
features = dataset[:, :-1]
dataset = np.genfromtxt(file_name, delimiter='\t', dtype=str)
labels = dataset[:, -1]
# data_pd = pd.DataFrame(dataset[:, :-1])
# print data_pd.head()

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
from sklearn.cross_validation import KFold

kf = KFold(len(features), n_folds=5, shuffle=True)
means = []
for training,testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])

    curmean = np.mean(prediction == labels[testing])
    means.append(curmean)

print('Mean accuracy: {:.1%}'.format(np.mean(means)))

