

import simpy
import copy
def my_process(env,time=5):
    yield env.timeout(13)
    print("my_process end   @ {0:.3f} ms".format(env.now))

def gen_event(env):
    events=[]
    event=my_process(env)
    events.append(event)
    event=my_process(env,10)
    events.append(event)
    return events
def execute(env,events): 
    for et in events:
        execute_event=[env.process(event) for event in et ]
        print("execute start @ {:.3f} ms".format(env.now))
        yield simpy.AllOf(env,execute_event)
        #yield env.timeout(5)
        print("execute end @ {:.3f} ms".format(env.now))

def test_1(env,events):
    for event in events:
        yield env.process(event)
def test(env):
    return env.process(my_process(env))
if __name__ == '__main__':
    env=simpy.Environment()
    events=[gen_event(env),gen_event(env)]
    #events1=[gen_event(env),gen_event(env)]
    events1=copy.deepcopy(events)
    env.process(execute(env,events))
    env.process(execute(env,events1))
    env.run(until=300)