from typing import List,Optional,Union
import matplotlib.pyplot as plt
import random
import numpy as np
class GA():
    def __init__(self,pop_num=20,max_gen=50,p_m=0.1,p_c=0.9) -> None:
        self.pop=0
        self.pop_num=pop_num
        self.pop_best=0
        self.max_gen=max_gen
        self.p_m=p_m
        self.p_c=p_c
        self.perf=np.ones(pop_num)
        self.max_perf_trace=np.zeros(max_gen)
        self.min_perf_trace=np.zeros(max_gen)
        self.dna_unit=2
    def Init_pop(self,p_dims:List[int],stg_num=32,d_num=256):
        assert(len(p_dims)==stg_num)
        self.dna_unit=stg_num
        ori_pop=[-1]*d_num
        ori_pop=np.array(ori_pop)
        self.pop=np.zeros((self.pop_num,d_num))
        #print(ori_pop)
        dd=d_num//stg_num
        for i in range(stg_num):
            for j in range(dd):
                ori_pop[i*dd+j]=i
        #random gen
        for i in range(self.pop_num):
            temp=ori_pop.copy()
            for j in range(d_num//10):
                idx=random.randint(0,d_num-1)
                temp[idx]=-1
            shuffle_flag=random.choice([True, False])
            if shuffle_flag:
                random.shuffle(temp)
            #print(temp)
            self.pop[i]=temp
    def Genetic_Operator(self):
        fitness=1/self.perf
        fitness_sum=fitness.sum()
        idx =np.random.choice(np.arange(self.pop_num), size=self.pop_num, replace=True,p=fitness/fitness_sum)
        pop_temp=self.pop[idx]
        self.pop=pop_temp
        self.pop_best=self.pop[0]
    def Crossover_Operator(self):
        DNA_SIZE=len(self.pop[0])
        for i in range(self.pop_num):
            parent=self.pop[i].copy()
            if np.random.rand() < self.p_c:
                    i_ = np.random.randint(0, self.pop_num, size=1)
                    cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)
                    #print(self.pop[i_, cross_points])
                    parent[cross_points] = self.pop[i_, cross_points]
            self.pop[i]=parent
    def Mutation_Operator(self):
        DNA_SIZE=len(self.pop[0])
        for pop_i in range(self.pop_num):
            child=self.pop[pop_i].copy()
            for point in range(DNA_SIZE):
                if np.random.rand() < self.p_m:
                    temp=np.random.randint(0, self.dna_unit, size=1)
                    child[point]= temp
                    #print(child[point])
    def Fitness(self,perf_func):
        for i in range(self.pop_num):
            self.perf[i]=perf_func(self.pop[i])
    def Evolution(self,perf_func):
        for i in range(self.max_gen):
            self.Crossover_Operator()
            self.Mutation_Operator()
            self.Fitness(perf_func)
            self.Genetic_Operator()
            self.max_perf_trace[i]=max(self.perf)
            self.min_perf_trace[i]=min(self.perf)
def perf_func_test(pop):
    return pop.sum()
    
if __name__ == '__main__':
    test=GA(pop_num=50,max_gen=200,p_m=0.1,p_c=0.5)
    test.Init_pop(p_dims=[2]*32,stg_num=32,d_num=256)
    test.Evolution(perf_func_test)
    #print(test.pop_best)
    #plt.plot(test.max_perf_trace)
    #plt.plot(test.min_perf_trace)
    plt.show()