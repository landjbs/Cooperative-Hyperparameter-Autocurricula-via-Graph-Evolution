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

    # Funnel type densely connected layers of exponential size
    if type == "Funnel":
        # flag used to specify multiplier between layers
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
            # each layer densely connected to layer below
            p = 1.0/(oldStop - oldStart)
            for j in range(oldStart,oldStop):
                adjMat[i][j] = p
                childrenList[i].append(j)
            delta += 1
        # bottom vertex densely connected to top layer
        p = 1.0/(N - oldStop)
        for j in range(oldStop,N):
            adjMat[0][j] = p
            childrenList[0].append(j)

    return adjMat, childrenList

def visualize_structure(vertexValues, childrenList, type=None, flag=None, exclude = []):
    N = len(vertexValues)
    positions = [(0,0) for _ in range(N)]

    if type == "Moran":
        delta = 2*np.pi/N
        for i in range(N):
            positions[i] = (math.cos(delta*i),math.sin(delta*i))

    if type == "Funnel":
        numLayers = math.ceil(math.log(N,flag))
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
            x = (delta+1)/(flag**layer + 1)
            y = layer/(numLayers + 1)
            positions[i] = (x,y)
            delta += 1

        positions[0] = (0.5,0)
        exclude.append(0)

    if type == "Superfan":
        numLayers = math.ceil(math.log(N,flag))
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

            layerSize = flag**(layer-1)
            theta1 = (delta//layerSize)*2*np.pi/flag
            frac = 2*np.pi/flag/numLayers
            theta2 = (layer-1)*frac
            theta3 = (delta%layerSize)*(frac/layerSize)
            r = ((delta%layerSize)+1)/layerSize
            theta = theta1 + theta2 + theta3
            positions[i] = (r*math.cos(theta),r*math.sin(theta))
            delta += 1

        positions[0] = (0,0)

    for i in range(N):
        plt.scatter(positions[i][0], positions[i][1],c='K')
        if i in exclude:
            print(i)
            continue
        else:
            for j in childrenList[i]:
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                plt.arrow(positions[i][0],positions[i][1],dx,dy)
    plt.show()

# currently not set
def updateValue(vertexValues, parent, child, momentum, flag=None):
    vertexValues[child] = vertexValues[parent]
    return vertexValues


if __name__=='__main__':
    '''
    sample experiment
    '''
    N = 156
    iter = 100
    n = 5

    vertexValues = [1 for _ in range(N-1)]
    vertexValues.append(1.5)
    vertexMomentums = [0 for _ in range(N)]

    adjMat, childrenList = generate_graph(N, "Funnel")

    for it in range(iter):
        vertexFitnesses = [value for value in vertexValues]
        # print(vertexFitnesses)
        run_iteration(vertexValues, vertexMomentums, vertexFitnesses, adjMat, n=n)
        print(it)
        print(vertexValues)
