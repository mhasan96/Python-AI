import numpy as np import random
def fitness(population, n):

clashes = np.zeros(len(population))
for q,chromosome in enumerate(population): cr = chromosome.tolist()
row_col_clashes = sum([cr.count(col)-1 for col in cr])/2 
print(row_col_clashes)
clashes[q]+=row_col_clashes 
cl=0
for i in range(len(chromosome)):
    for j in range(len(chromosome)):
        if ( i != j):
            dx = abs(i-j)
            dy = abs(chromosome[i] - chromosome[j])
            if dx == dy:
                cl+=1 clashes[q]+=(cl//2)
                return 28-clashes

def select(population, fit):

    fitness_values = fit(population, n)
    probability= fitness_values/sum(fitness_values)
    selected_chromosome_fitness = np.random.choice(fitness_values , size=1 ,replace=True ,p=probability)
    selected_parent_index =fitness_values.tolist().index(selected_chromosome_fitness[0]) 
    return population[selected_parent_index]

def crossover(x, y):
    index = random.randint(0,4)
    child =np.append(x[0:index] , y[index:8]) 
    return child
 
def mutate(child):
    index = random.randint(0,7) 
    gene = random.randint(1,8) 
    child[index]=gene
    return child


def GA(population, n, mutation_threshold = 0.3):

#list to store fitness value in each generation generation_fitness=[]
    generations = 0




while 28 not in fitness(population,n) : new_population = []
for i in range(len(population)): 
    x=select(population,fitness)
    y=select(population,fitness) 
    child = crossover(x,y)
    if random.uniform(0.0,0.5) < mutation_threshold:
         child = mutate(child) 
         new_population.append(child)
         population = np.array(new_population) 
         population_fitness = fitness(population,n) 
         generation_fitness.append(np.max(population_fitness)) generation_fitness.append(np.min(population_fitness)) generation_fitness.append(np.mean(population_fitness)) 
         generations+=1

#getting the best fitness from the last generation best_fitness = np.max(fitness(population,n))

#adding the best fitness value to a list to plot the evolution generation_fitness.append(best_fitness)
#finding the index of the best fitness value
index =	fitness(population,n).tolist().index(best_fitness)
 
#using this index value to find result chromosome individual= population[index]


#return the result #chromosome | #fitness value | #generations |
#list of all fitness	value in each generation
return individual , best_fitness ,generations,generation_fitness

'''for 8 queen problem, n = 8'''
n = 8

'''start_population denotes how many individuals/chromosomes are there in the initial population n = 8'''
start_population = 10

'''if you want you can set mutation_threshold to a higher value, to increase the chances of mutation'''
mutation_threshold = 0.3

'''creating the population with random integers between 0 to 7 inclusive for n = 8 queen problem'''
population = np.random.randint(0, n, (start_population, n))

'''calling the GA function'''
chromosome , fitness , generations , generationfitnesses = GA(population, n, mutation_threshold)
print(f'selected individual --> {chromosome} \n') print(f'fitness		--> {fitness} \n')
print(f'Generations	--> {generations} \n')
print(f'{generationfitnesses} \n')

