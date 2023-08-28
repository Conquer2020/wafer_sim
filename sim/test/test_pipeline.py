import simpy
import util
#GPIPE=[0,1,-1]
#GPIPE_1=[1,0,-1]
from  util import BaseEnum as Enum
pipe_strategy=Enum('pipe_strategy',('GPipe','Megatron1F1B','Interleaved1F1B','Cerebras'))
class stage():
    def __init__(self,env,time_out=[50,100,20]) -> None:
        self.time_out=time_out
        self.env=env
        self.res=simpy.PriorityResource(env, capacity=1)
        self.trace=[]
        self.res_fd_cnt=0
        self.prio=1
    def res_use(self,c_type=1,wait=1e-15):
        #yield self.env.timeout(wait)
        with self.res.request(priority=self.prio) as req:
                yield req
                t_last=env.now
                yield self.env.timeout(self.time_out[c_type])
                if c_type==0:
                    self.res_fd_cnt+=1
                else:
                    self.res_fd_cnt-=1
                self.trace.append((t_last,env.now,c_type))

class Stages():
    def __init__(self,env,stage_num=3,micro_batch_num=3,strategy=pipe_strategy.GPipe) -> None:
        self.stage_num=stage_num
        self.stgs=[]
        self.reg=[]
        self.env=env
        self.micro_batch_num=micro_batch_num
        self.cur_fd_times=0
        self.cur_bd_times=0
        self.one_epoch_finish=simpy.Store(self.env,capacity=1)
        self.one_fd_finish=simpy.Store(self.env,capacity=1)
        self.strategy=strategy
    def res_set(self):
        self.stgs=[]
        for i in range(self.stage_num):
            self.stgs.append(stage( self.env))
            self.reg.append(simpy.PriorityStore(self.env, capacity=1))
    def forward(self):
        for i,stg in enumerate(self.stgs):
            yield self.env.process(stg.res_use(c_type=0))
            if self.strategy==pipe_strategy.Megatron1F1B:
                if i==self.stage_num-2:
                    yield self.reg[i].put(1)
                else :
                    self.reg[i].put(1)
            elif self.strategy==pipe_strategy.GPipe:
                if i==self.stage_num-1:
                    self.cur_fd_times+=1
                #print('self.cur_time',self.cur_time)
                if self.cur_fd_times==self.micro_batch_num:
                    self.one_fd_finish.put(1)
                    print('self.one_fd_finish.put(1)',self.cur_fd_times)   
            else:
                raise NotImplementedError
    def backward(self):
        for i in range(len(self.stgs)-1,-1,-1):
            if self.strategy==pipe_strategy.Megatron1F1B:
                with self.reg[i].get() as get:
                    a=yield get
                    stg=self.stgs[i]
                    yield self.env.process(stg.res_use(c_type=1))  
                    if i==0:
                        self.cur_bd_times+=1
                    if self.cur_bd_times==self.micro_batch_num:
                        self.one_epoch_finish.put(1)
            elif  self.strategy==pipe_strategy.GPipe:  
                stg=self.stgs[i]
                yield self.env.process(stg.res_use(c_type=1))  
                if i==0:
                    self.cur_bd_times+=1
                if self.cur_bd_times==self.micro_batch_num:
                    self.one_epoch_finish.put(1)      
    def parameter_syn(self):
        while(True):
            with self.one_epoch_finish.get() as get:
                a=yield get
                for i,stg in enumerate(self.stgs):
                    self.env.process(stg.res_use(c_type=2))
                break
    def run(self,until=10000):
        def all_backward():
            while(True):
                with self.one_fd_finish.get() as get:
                    a=yield get    
                    for i in range(self.micro_batch_num):
                        self.env.process(self.backward()) 
                break
        self.res_set()
        for i in range(self.micro_batch_num):
            self.env.process(self.forward())
        if self.strategy==pipe_strategy.GPipe:  
            self.env.process(all_backward())
        elif self.strategy==pipe_strategy.Megatron1F1B:  
            for i in range(self.micro_batch_num):
                self.env.process(self.backward())
        self.env.process(self.parameter_syn())
        self.env.run(until=until)
env=simpy.Environment()
ss=Stages(env,stage_num=10,micro_batch_num=5,strategy=pipe_strategy.GPipe)#Megatron1F1B#
ss.run()
all_trace=[]
for stg in ss.stgs:
    all_trace.append(stg.trace)
pipe_endtime=all_trace[0][-1][1]
util.draw_pipeline(all_trace,path='./',title='test',endtime=pipe_endtime,name='test')


