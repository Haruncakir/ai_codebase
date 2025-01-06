groups = [
    [['Red'], ['Blue'], ['Red']],
    [['Blue'], ['Red'], ['Blue'], ['Blue']],
]
classes = ['Red', 'Blue']

n_instances = float(sum([len(group) for group in groups]))

def gini_index(groups, classes):
    gini = 0.0
    for group in groups:
        size = len(group)
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))  # Find unique classes in the dataset
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):    # Exclude the last column which is the class
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Age, Movie Genre, Decision (watch or not)

dataset = [
    [18, 1, 0],
    [20, 0, 1],
    [23, 2, 1],
    [25, 1, 1],
    [30, 1, 0],
]

split = get_split(dataset)
print('\nBest Split:')
print('Column Index: %s, Value: %s' % ((split['index']), (split['value'])))
# Output: Column Index: 0, Value: 20
