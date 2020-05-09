import numpy as np

def generate_graph(N, type = "Moran", flag=None):

    adjMat = [[0 for _ in range(N)] for _ in range(N)]
    childrenList = [[] for _ in range(N)]
    if type == "Moran":
        for i, vertex in enumerate(adjMat):
            for j in range(N):
                if j != i:
                    vertex[j] = 1.0/(N - 1)
                    childrenList[i].append(j)

    if type == "Funnel":
        if flag is None:
            flag = 5
        layer = 1
        delta = 0
        oldStart = 0
        oldStop = 1
        for i in range(1,N):
            if delta >= flag ** layer:
                layer += 1
                delta = 0
                oldStart = oldStop
                oldStop = i

            p = 1.0/(oldStop - oldStart)
            for j in range(oldStart,oldStop):
                adjMat[i][j] = p
                childrenList[i].append(j)
            delta += 1
        p = 1.0/(N - oldStop)
        for j in range(oldStop,N):
            adjMat[0][j] = p
            childrenList[0].append(j)

    return adjMat, childrenList

def run_iteration(vertexValues, vertexMomentums, vertexFitnesses, adjMat, n=1, flag=None):

    N = len(vertexValues)

    fitSum = float(sum(vertexFitnesses))
    p = [fit/fitSum for fit in vertexFitnesses]
    reproducing = np.random.choice(range(N),size=n,p=p)

    for parent in reproducing:
        p = adjMat[parent]
        if sum(p) > 0:
            child = np.random.choice(range(N), p=p)
            vertexValues = updateValue(vertexValues, parent, child, vertexMomentums[parent], flag=flag)

            vertexMomentums[parent] += 1
            vertexMomentums[child] = 0 # should this be set to parent momentum?

    return None

# currently not set
def updateValue(vertexValues, parent, child, momentum, flag=None):
    vertexValues[child] = vertexValues[parent]
    return vertexValues



N = 156
iter = 2000
n = 5

vertexValues = [np.random.uniform(0.7,1.3) for _ in range(N)]
vertexMomentums = [0 for _ in range(N)]

adjMat, childrenList = generate_graph(N, "Funnel")

for it in range(iter):
    vertexFitnesses = [value for value in vertexValues]
    # print(vertexFitnesses)
    run_iteration(vertexValues, vertexMomentums, vertexFitnesses, adjMat, n=100)
    print(it)
print(vertexValues)
