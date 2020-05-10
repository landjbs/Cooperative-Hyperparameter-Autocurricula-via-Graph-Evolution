from evolution import generate_graph
import numpy as np
import matplotlib.pyplot as plt

def run_trial(vertexFitnesses, adjMat, steps):

    for step in range(steps):
        N = len(vertexFitnesses)
        fitSum = float(sum(vertexFitnesses))
        p = [fit/fitSum for fit in vertexFitnesses]
        parent = np.random.choice(range(N),p=p)

        p = adjMat[parent]
        if sum(p) > 0:
            child = np.random.choice(range(N), p=p)
            vertexFitnesses[child] = vertexFitnesses[parent]
        if step % 100 == 0:
            if min(vertexFitnesses) == max(vertexFitnesses):
                return vertexFitnesses
    return vertexFitnesses

r = 1.5
trials = 100
max_steps = 25000

for N, flag in [(7,2),(13,3),(21,4),(31,5),(43,6),(111,10)]:
    for type in ["Moran","Funnel","Superfan","Star"]:
        print(N,type,flag)
        if type == "Star":
            adjMat,_ = generate_graph(N, "Funnel", N - 1)
        else:
            adjMat,_ = generate_graph(N, type, flag)
        count = 0
        for trial in range(trials):
            vertexFitnesses = np.ones(N)
            idx = np.random.choice(range(N))
            vertexFitnesses[idx] = r
            vertexFitnesses = run_trial(vertexFitnesses, adjMat, max_steps)
            if np.median(vertexFitnesses) == r:
                count += 0.01
            print(vertexFitnesses)
        if type == "Moran":
            plt.scatter(N, count, c="k")
        elif type == "Funnel":
            plt.scatter(N, count, c="r")
        elif type == "Star":
            plt.scatter(N, count, c="y")
        else:
            plt.scatter(N, count, c="b")

x = np.linspace(0,115)
y = (1 - 1/r)/(1 - 1/(r**x))
plt.plot(x,y, c="k")
plt.xlabel("Population Size")
plt.ylabel("Mutant Fixation Rate")
plt.title("Fixation Rate of Mutatant r = 1.5")
plt.show()
