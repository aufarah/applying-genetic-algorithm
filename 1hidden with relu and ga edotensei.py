import struct
import copy
import numpy as np
import matplotlib
from matplotlib import pyplot

numChild = 60
numGeneration = 50
initialpop = 40

def havesex(population):
    for i in range(numChild):
        #choosing parent
        p1 = np.random.randint(len(population))
        p2 = np.random.randint(len(population))

        #switching parameters (gen) between parent
        child = copy.deepcopy(population[p1])

        #swapping 5 element randomly
        limxW1 = child.weights1.shape[0]
        limyW1 = child.weights1.shape[1]
        chosen= np.random.choice(limyW1,2,replace=False)
        z = 2
        constnar = np.random.uniform(-z,z)
        child.weights1[np.random.randint(limxW1)][chosen]  = constnar * population[p2].weights1[np.random.randint(limxW1)][chosen]
        
        limxW1 = child.bias1.shape[0]
        limyW1 = child.bias1.shape[1]
        chosen= np.random.choice(limxW1,3,replace=False)
        constnar = np.random.uniform(-z,z)
        child.bias1[chosen]      = constnar  * population[p2].bias1[chosen]
        
        limxW1 = child.weights2.shape[0]
        limyW1 = child.weights2.shape[1]
        chosen= np.random.choice(limyW1,3,replace=False)
        constnar = np.random.uniform(-z,z)
        child.weights2[np.random.randint(limxW1)][chosen]   = constnar  * population[p2].weights2[np.random.randint(limxW1)][chosen]
        
        limxW1 = child.bias2.shape[0]
        limyW1 = child.bias2.shape[1]
        chosen= np.random.choice(limxW1,1,replace=False)
        constnar = np.random.uniform(-z,z)
        child.bias2[chosen][np.random.randint(limyW1)]      = constnar  * population[p2].bias2[chosen][np.random.randint(limyW1)]
        
        population  = np.append(population,child)
    return population

def findfit(population):
    result = np.array([])
    loops  = len(population)

    for i in range(loops):
        #feed forwarding all sample
        for j in range(0, y.shape[0]):
            population[i].feedind(j)

        #finding average cost of all sample
        cost = population[i].cost(y,population[i].output)
        result = np.append(result,cost)
        
    return result
    
def sort(input,output):
    for i in range(len(output)):
        lowIndex = i
        for j in range(i+1,len(output)):
            if output[j] < output[lowIndex]:
                lowIndex = j

        output[i],output[lowIndex] = output[lowIndex],output[i]
        input[i],input[lowIndex] = input[lowIndex],input[i]

def mutation(individu):
    individu.randominit()

class NeuralNetwork:
    def cost(self, y,out):
        cost = 0
        for i in range(0, self.y.shape[0]):
            errind = self.y[i][0] - self.output[i] 
            errind = np.array(errind)
            costind = np.dot(errind,errind)
            cost = cost + costind
        
        cost = cost / self.y.shape[0]
        return cost
    

    def sigmoid(self, x):
        #applying the sigmoid function, okay revised to relu
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, y):
        #computing derivative to the Sigmoid function, its sure after penurunan matematik
        return y * (1 - y)

    def relu(self, x):
        #applying the sigmoid function, okay revised to relu
        #return 1 / (1 + np.exp(-x))
        return x*(x>=0)
        
    def drelu(self, y):
        #computing derivative to the Sigmoid function, its sure after penurunan matematik
        #return y * (1 - y)

        #d(sumh)/dh = d(y)/dh, kalau y+ otomatis sumh+ then derivat 1, kalau y0 otomatis then derivat 0 (we dont care about sumh)
        return 1.*(y>=0)

    def __init__(self, x, y):
        self.h1          = np.array([np.zeros(6)]).T                  #hidden layer 1, ada 6 neuron
        self.input      = x                                                               #horizontal m, in vertical sample            
        self.y          = y                                                               #horizontal n, in vertical sample
        
		# 2|6|1
        self.weights2   = np.random.random( (self.y.shape[1], self.h1.shape[0]) )      #horizontal per input m, verticaled as y n
        self.bias2      = np.array([np.random.random(self.y.shape[1])]).T
        self.bias1      = np.array([np.random.random(self.h1.shape[0])]).T
        self.weights1   = np.random.random( (self.h1.shape[0], self.input.shape[1]) )        #horizontal aligned
        self.randominit()
        
        # eta adalah learning rate
        self.eta = 1
        self.output     = np.zeros(y.shape)
        self.error      = self.y - self.output
        self.wat        = self.sigmoid_derivative(self.output)

    def randominit(self):
        self.weights2   = np.random.random( (self.y.shape[1], self.h1.shape[0]) )      #horizontal per input m, verticaled as y n
        self.bias2      = np.array([np.random.random(self.y.shape[1])]).T
        self.bias1      = np.array([np.random.random(self.h1.shape[0])]).T
        self.weights1   = np.random.random( (self.h1.shape[0], self.input.shape[1]) ) 

    def feedind(self,x):
        #jadi fungsi ini ngefeedfwd semua sampel... padahal ga perlu
        #nah makanya ini akan urg bikin indivnya perindex sample x
        sumh1 = np.matrix(self.weights1) * np.matrix(self.input[x]).T
        outh1 = sumh1 + self.bias1
        outh1 = np.array(outh1)
        self.h1 = self.relu(outh1)

        sumir = np.matrix(self.weights2) * np.matrix(self.h1)
        out   = sumir + self.bias2
        out   = np.array(out).T
        self.output[x] = self.relu(out)              #horizontal
        self.output[x] = np.array(self.output[x])
    
    def test(self,x):
        #jadi fungsi ini ngefeedfwd masukan (bukan sampel)
        sumh1 = np.matrix(self.weights1) * np.matrix(x).T
        outh1 = sumh1 + self.bias1
        outh1 = np.array(outh1)
        h1 = self.relu(outh1)

        sumir = np.matrix(self.weights2) * np.matrix(h1)
        out   = sumir + self.bias2
        out   = np.array(out).T
        out   = self.relu(out)

        return out

if __name__ == "__main__":
    x = np.array([[1,0],[2,0],[3,0],[4,0]])
    y = np.array([[2],[3],[4],[5]])

    #creating initial population / ancestors
    population = np.array([])
    newpop     = np.array([])
    for i in range(initialpop):
        population = np.append(population, NeuralNetwork(x,y))

    for c in range(numGeneration):
        #mating
        population = havesex(population)

        #find fitness of populations
        fitness = findfit(population)

        #sorting
        sort(population,fitness)

        #kill the unfits (in this case is the biggest 3)
        initial = len(population)
        for i in range(numChild):
            index = i + 1
            population = np.delete(population,initial-index)
            fitness   = np.delete(fitness,initial-index)
        print(fitness[0])
        if(fitness[0]<1e-3):
            newpop = np.append(newpop,population[0])

        #random mutation every 10 iteration
        """
        if (c%10==0):
            max = len(population)
            indiv = population[np.random.randint(max)]                     #individu that will be mutated entirely
            mutation(indiv)
        #loop"""

    population = newpop
    for c in range(numGeneration):
        #mating
        population = havesex(population)

        #find fitness of populations
        fitness = findfit(population)

        #sorting
        sort(population,fitness)

        #kill the unfits (in this case is the biggest 3)
        initial = len(population)
        for i in range(numChild):
            index = i + 1
            population = np.delete(population,initial-index)
            fitness   = np.delete(fitness,initial-index)
        print(fitness[0])
        if(fitness[0]<1e-7):
            newpop = np.append(newpop,population[0])
    
    for i in range(0, y.shape[0]):
        yt = population[0].test(x[i])
        print(yt)
        
    population = newpop
    for c in range(numGeneration):
        #mating
        population = havesex(population)

        #find fitness of populations
        fitness = findfit(population)

        #sorting
        sort(population,fitness)

        #kill the unfits (in this case is the biggest 3)
        initial = len(population)
        for i in range(numChild):
            index = i + 1
            population = np.delete(population,initial-index)
            fitness   = np.delete(fitness,initial-index)
        print(fitness[0])
        if(fitness[0]<1e-10):
            newpop = np.append(newpop,population[0])
    
    for i in range(0, y.shape[0]):
        yt = population[0].test(x[i])
        print(yt)
    """
    for i in range(numGeneration):
        havesex
        costfunc
        sort
        kill
        mutate
        breaking

    binar = NeuralNetwork(x,y)

    for i in range(0, y.shape[0]):
        binar.feedind(i)
        yt = binar.output[i]
        print(yt)
    cost = binar.cost(y,binar.output)
    print("Cost: ",cost)

    #sudah ditraining ya om, otw minimum global brow!, Cost:  4.838387113974794e-05
    epoch = 10000
    nepoch= np.array([0])
    ncost = np.array([cost])
    binar.eta = 0.05
    for i in range(0,epoch):
        v = np.random.randint(0, y.shape[0])
        binar.backprop(v)
        for j in range(0, y.shape[0]):
            binar.feedind(j)
            #print(binar.h1)
            #print(binar.output[j])
        cost = binar.cost(y,binar.output)
        print("Cost: ",cost)

        ncost = np.append(ncost, cost)
        nepoch= np.append(nepoch,i+1)

        if(ncost[i+1]==ncost[i]==ncost[i-1]==ncost[i-2]==ncost[i-3] and cost>=1e-5):
            binar.randominit()
            #if die, mutate

        if(cost<=1e-25):
            break

    
    pyplot.plot(nepoch,ncost)
    pyplot.show()    

    print()
    print(binar.weights1)
    print(binar.weights2)
    print(binar.bias1)
    print(binar.bias2)
    print()

    for i in range(0, y.shape[0]):
        yt = binar.test(x[i])
        print(yt)
    
    print()

    yt = binar.test(np.array([13,0]))
    print(yt)
    """