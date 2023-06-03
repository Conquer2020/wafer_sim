import simpy
def producer(env, store):
     for i in range(100):
         yield env.timeout(2)
         yield store.put(f'spam {i}')
         print(f'Produced spam at', env.now)


def consumer(name, env, store):
     while True:
         yield env.timeout(1)
         print(name, 'requesting spam at', env.now)
         item = yield store.get()
         print(name, 'got', item, 'at', env.now)


env = simpy.Environment()
store = simpy.Store(env, capacity=2)

prod = env.process(producer(env, store))
consumers = [env.process(consumer(i, env, store)) for i in range(2)]

env.run(until=10)