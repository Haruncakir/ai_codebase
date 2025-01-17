import numpy as np
import matplotlib.pyplot as plt


data_points = np.array([
    [1.2, 1.9], [2.1, 2], [2, 3.5], [3.3, 3.9], [3.2, 5.1],
    [8.5, 7.9], [8.1, 7.8], [9.5, 6.5], [9.5, 7.2], [7.7, 8.6],
    [6.0, 6.0]
])

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


def dbscan(data, Eps, MinPt):
    point_label = [0] * len(data)
    # Initialize list to maintain count of surrounding points within radius Eps for each point.
    point_count = []
    core = []
    noncore = []

    # Check for each point if it falls within the Eps radius of point at index i
    for i in range(len(data)):
        point_count.append([])
        for j in range(len(data)):
            if euclidean_distance(data[i], data[j]) <= Eps and i != j:
                point_count[i].append(j)

        # If a point has atleast MinPt points within its Eps radius (excluding itself), classify it as a core point, and vice versa
        if len(point_count[i]) >= MinPt:
            core.append(i)
        else:
            noncore.append(i)
    ID = 1
    for point in core:
        # If the point has not been assigned to a cluster yet
        if point_label[point] == 0:
            point_label[point] = ID
            # Create an empty list to hold 'neighbour points'
            queue = []
            for x in point_count[point]:
                if point_label[x] == 0:
                    point_label[x] = ID
                    # If neighbor point is also a core point, add it to the queue
                    if x in core:
                        queue.append(x)

            # Check points from the queue
            while queue:
                neighbours = point_count[queue.pop(0)]
                for y in neighbours:
                    if point_label[y] == 0:
                        point_label[y] = ID
                        if y in core:
                            queue.append(y)
            ID += 1

    return point_label


labels = dbscan(data_points, 2, 2)

for i in range(len(labels)):
    if labels[i] == 1:
        plt.scatter(data_points[i][0], data_points[i][1], s=100, c='r')
    elif labels[i] == 2:
        plt.scatter(data_points[i][0], data_points[i][1], s=100, c='g')
    else:
        plt.scatter(data_points[i][0], data_points[i][1], s=100, c='b')

plt.show()
