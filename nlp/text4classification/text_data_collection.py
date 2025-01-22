# Importing necessary libraries
from sklearn.datasets import fetch_20newsgroups

# Fetch data
newsgroups = fetch_20newsgroups(subset='all')

# Understanding the structure of the data
print("\n\nData Structure\n-------------")
print(f'Type of data: {type(newsgroups.data)}')
print(f'Type of target: {type(newsgroups.target)}')

'''
Data Structure
-------------
Type of data: <class 'list'>
Type of target: <class 'numpy.ndarray'>
'''

print("\n\nData Exploration\n----------------")
print(f'Number of datapoints: {len(newsgroups.data)}')
print(f'Number of target variables: {len(newsgroups.target)}')
print(f'Possible classes: {newsgroups.target_names}')

'''
Data Exploration
----------------
Number of datapoints: 18846
Number of target variables: 18846
Possible classes: ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
    'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
    'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
'''

print("\n\nSample datapoint\n----------------")
print(f'\nArticle:\n-------\n{newsgroups.data[10]}')
print(f'\nCorresponding Topic:\n------------------\n{newsgroups.target_names[newsgroups.target[10]]}')

'''
Sample datapoint
----------------

Article:
-------
From: sandvik@newton.apple.com (Kent Sandvik)
Subject: Re: 14 Apr 93   God's Promise in 1 John 1: 7
Organization: Cookamunga Tourist Bureau
Lines: 17

In article <1qknu0INNbhv@shelley.u.washington.edu>, > Christian:  washed in
the blood of the lamb.
> Mithraist:  washed in the blood of the bull.
> 
> If anyone in .netland is in the process of devising a new religion,
> do not use the lamb or the bull, because they have already been
> reserved.  Please choose another animal, preferably one not
> on the Endangered Species List.  

This will be a hard task, because most cultures used most animals
for blood sacrifices. It has to be something related to our current
post-modernism state. Hmm, what about used computers?

Cheers,
Kent
---
sandvik@newton.apple.com. ALink: KSAND -- Private activities on the net.


Corresponding Topic:
------------------
talk.religion.misc
'''


