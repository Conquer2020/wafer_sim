import simpy
class comm_overlap():
    def __init__(self,env) -> None:
        self.env=env
        self.cp_worker= simpy.Resource(env, capacity=1)
        self.cm_worker= simpy.Resource(env, capacity=1)
    def cp_process(self):
        '''
        process 1
        '''
        with self.cp_worker.request() as req:
                yield req
                yield self.env.timeout(20)
        print('process 1 done @{:.3f} '.format(self.env.now))
    def cm_process(self):
        '''
        process 2
        '''
        with self.cm_worker.request() as req:
                yield req
                yield self.env.timeout(30)
        print('process 2 done @{:.3f} '.format(self.env.now))
    def overlap_process(self):
         event_list=[]
         while(True):
              event_list.append(self.env.process(self.cp_process()))
              event_list.append(self.env.process(self.cm_process()))
              yield simpy.AllOf(env,event_list)
              print('process overlap_process done @{:.3f} '.format(self.env.now))
              break
    def order_process(self):
         while(True):
              yield self.env.process(self.cp_process())
              yield self.env.process(self.cm_process())
              print('process order_process done @{:.3f} '.format(self.env.now))
              break
    def short_process(self):
        while(True):
              yield self.env.timeout(20)
              yield self.env.timeout(30)
              print('process short_process done @{:.3f} '.format(self.env.now))
              break
if __name__ == '__main__':
    env=simpy.Environment()
    test=comm_overlap(env)
    env.process(test.overlap_process())
    env.process(test.order_process())
    env.process(test.short_process())
    env.run(until=100)