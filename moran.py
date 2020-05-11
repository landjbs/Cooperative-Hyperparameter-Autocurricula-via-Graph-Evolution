from evolution import generate_graph
import numpy as np
import matplotlib.pyplot as plt

def run_trial(vertexFitnesses, adjMat, steps):

    for step in range(steps):
        N = len(vertexFitnesses)
        fitSum = float(sum(vertexFitnesses))
        p = [fit/fitSum for fit in vertexFitnesses]
        reproducing = np.random.choice(range(N), size =1,p=p)
        for parent in reproducing:
            p = adjMat[parent]
            if sum(p) > 0:
                child = np.random.choice(range(N), p=p)
                vertexFitnesses[child] = vertexFitnesses[parent]
            if step % 100 == 0:
                if min(vertexFitnesses) == max(vertexFitnesses):
                    return vertexFitnesses
    return vertexFitnesses

r = 1.5
trials = 1000
max_steps = 50000

with open("moranresults.txt","w") as f:
    for N, flag in [(7,2),(13,3),(21,4),(31,5)]:
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
                    count += 0.001
                print(vertexFitnesses)
            if type == "Moran":
                plt.scatter(N, count, c="k")
            elif type == "Funnel":
                plt.scatter(N, count, c="r")
            elif type == "Star":
                plt.scatter(N, count, c="y")
            else:
                plt.scatter(N, count, c="b")
            f.write(str(f'{type,N,count}\n'))
x = np.linspace(1,31)
y = (1 - 1/r)/(1 - 1/(r**x))
y2 = (1 - 1/(r**2))/(1 - 1/(r**(2*x)))
plt.plot(x,y, c="k")
plt.plot(x,y2, c="y")
plt.xlabel("Population Size")
plt.ylabel("Mutant Fixation Rate")
plt.title("Fixation Rate of Mutatant r = 1.5")
plt.show()
