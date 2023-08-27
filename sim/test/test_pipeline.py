import simpy
import util
#GPIPE=[0,1,-1]
GPIPE_0=[0,1,-1]#[1,0,-1]
GPIPE_1=[1,0,-1]
#GPIPE_1=[1,0,-1]
class stage():
    def __init__(self,env,time_out=[50,100,20]) -> None:
        self.time_out=time_out
        self.env=env
        self.res=simpy.PriorityResource(env, capacity=1)
        self.trace=[]
        self.fd_cnt=0
        self.prio=1
    def res_use(self,c_type=1,wait=1e-15):
        #yield self.env.timeout(wait)
        with self.res.request(priority=self.prio) as req:
                yield req
                #print('{} start @ {}'.format(prio,self.env.now))
                t_last=env.now
                yield self.env.timeout(self.time_out[c_type])
                if c_type==0:
                    self.fd_cnt+=1
                else:
                    self.fd_cnt-=1
                if self.fd_cnt==0:
                    self.prio=GPIPE_1[c_type]
                else:
                    self.prio=GPIPE_0[c_type]
                #print('c_type={},fd_cnt={}'.format(c_type,self.fd_cnt))
                self.trace.append((t_last,env.now,c_type))
                #print('{} end @ {}'.format(prio,self.env.now))

class Stages():
    def __init__(self,env,stage_num=3,times=3) -> None:
        self.stage_num=stage_num
        self.stgs=[]
        self.reg=[]
        self.env=env
        self.times=times
        self.cur_time=0
        self.one_epoch=simpy.Store(self.env,capacity=1)
    def set(self):
        self.stgs=[]
        for i in range(self.stage_num):
            self.stgs.append(stage( self.env))
            self.reg.append(simpy.PriorityStore(self.env, capacity=0.5))
    def fd(self):
        for i,stg in enumerate(self.stgs):
            print('fd {} start @ {}'.format(i,self.env.now))
            yield self.env.process(stg.res_use(c_type=0))
            print('fd {} end @ {}'.format(i,self.env.now))
            if i==self.stage_num-2:
                yield self.reg[i].put(1)
            else :
                self.reg[i].put(1)
    def bd(self):
        for i in range(len(self.stgs)-1,-1,-1):
            #print(i)
            with self.reg[i].get() as get:
                a=yield get
                stg=self.stgs[i]
                print('bd {} start @ {}'.format(i,self.env.now))
                yield self.env.process(stg.res_use(c_type=1))  
                print('bd {} end @ {}'.format(i,self.env.now)) 
                if i==0:
                    self.cur_time+=1
                print('self.cur_time',self.cur_time)
                if self.cur_time==self.times:
                    self.one_epoch.put(1)
                    print('self.one_epoch.put(1)',self.cur_time)               
    def syn(self):
        while(True):
            print(self.one_epoch.items)
            with self.one_epoch.get() as get:
                a=yield get
                for i,stg in enumerate(self.stgs):
                    #print('syn {} start @ {}'.format(i,self.env.now))
                    self.env.process(stg.res_use(c_type=2))
                    #print('syn {} end @ {}'.format(i,self.env.now)) 
                break
    def run(self):
        self.set()
        for i in range(self.times):
            self.env.process(self.fd())
            self.env.process(self.bd()) 
        self.env.process(self.syn())
env=simpy.Environment()
ss=Stages(env,stage_num=10,times=5)
ss.run()
env.run(until=10000)
all_trace=[]
for stg in ss.stgs:
    all_trace.append(stg.trace)
pipe_endtime=all_trace[0][-1][1]*1.1 
#print(all_trace)
util.draw_pipeline(all_trace,path='./',title='test',endtime=pipe_endtime,name='test')


