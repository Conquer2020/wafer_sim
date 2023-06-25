

import simpy
import matplotlib.pyplot as plt
from enum import Enum
from typing import List
from queue import Queue
from operator import mul
from functools import reduce
def my_process(env,time=5):
    print("my_process start @ {:.3f} ms".format(env.now))
    yield env.timeout(time)
    yield env.timeout(20)
    print("my_process end @ {:.3f} ms".format(env.now))

def gen_event(env):
    events=[]
    event=my_process(env)
    events.append(event)
    event=my_process(env,10)
    events.append(event)
    return events
def execute(env,events): 
    execute_event=[env.process(event) for event in events]
    print("execute start @ {:.3f} ms".format(env.now))
    yield simpy.AnyOf(env,execute_event)
    yield env.timeout(5)
    print("execute end @ {:.3f} ms".format(env.now))

def test_1(env,events):
    for event in events:
        yield env.process(event)
def test(env):
    return env.process(my_process(env))
if __name__ == '__main__':
    env=simpy.Environment()
    #print(gen_event(env))
    env.process(execute(env,gen_event(env)))
    env.run(until=200)