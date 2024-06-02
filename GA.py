import numpy
from sklearn.ensemble import RandomForestClassifier

def cal_pop_fitness(population, x_train2, y_train, x_val2, y_val):
    fitness = []
    for x in range(4):
        model_fit = RandomForestClassifier(n_estimators=population[x][0], 
                                       max_depth=population[x][1])
        model_fit.fit(x_train2, y_train)
        accuracy = model_fit.score(x_val2, y_val)
        fitness.append(accuracy)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Memilih individu-individu terbaik di generasi saat ini sebagai orang tua untuk menghasilkan keturunan generasi berikutnya.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # Titik di mana persilangan terjadi antara dua orang tua. Biasanya berada di tengah.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Indeks dari induk pertama yang melakukan perkawinan.
        parent1_idx = k%parents.shape[0]
        #  Indeks dari induk kedua yang akan dikawinkan.
        parent2_idx = (k+1)%parents.shape[0]
        # Keturunan baru akan mendapatkan separuh gen pertama yang diambil dari induk pertama.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # Keturunan baru akan mendapatkan separuh gen kedua yang diambil dari induk kedua.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutasi mengubah satu gen di setiap keturunan secara acak.
    for idx in range(offspring_crossover.shape[0]):
        # Nilai acak yang akan ditambahkan ke gen.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 1] = offspring_crossover[idx, 1] + random_value
    return offspring_crossover

