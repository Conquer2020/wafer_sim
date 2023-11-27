import simpy
import math
import time
from typing import List,Optional
from util import *
from tile_dataflow import Tile
from wafer_device import Wafer_Device as wd
from wafer_device import Packet
from ML import *

class Stage():
    __stage_id=0
    def __init__(self,env,tile_config:dict,op_list,last_core_id:Optional[List[int]],\
                 cur_core_id:Optional[List[int]],next_core_id:Optional[List[int]],noc:wd)-> None:
        self.tile=Tile(
                env=env,
                tile_name=tile_config['tile_name'],       
                sram_capacity_MB=tile_config['sram_capacity_MB'],
                macs=tile_config['macs'],
                freq_GHz=tile_config['freq_GHz'],
                with_dram=tile_config['with_dram'],
                dram_bw_GB=noc.tile_dram_bw_GB,
                dram_capacity_GB=noc.tile_dram_capacity_GB,
                opt=tile_config['opt'],
                ZeRO=tile_config['ZeRO'],
                Analytical=tile_config['Analytical']
                )
        self.op_list=op_list
        self.i_shape=[]
        self.o_shape=[]
        self.last_core_id=last_core_id
        self.cur_core_id=cur_core_id
        self.next_core_id=next_core_id
        self.stage_info=[]
        self.map_ana=[]
        #simpy env 
        self.env=env
        self.res=simpy.PriorityResource(env, capacity=1)
        self.trace=[]
        self.res_fd_cnt=0
        self.prio=1
        self.trace=[]
        self.__class__.__stage_id+=1
    def init_info(self,micro_batch):
        for op in self.op_list:
            op.param_dim[0]=micro_batch
            op.update()
        self.i_shape=self.op_list[0].i_shape
        self.o_shape=self.op_list[-1].o_shape
    def up_state(self,noc:wd,c_type=ML_STATE.FORWARD,wait=1e-15):
        #yield self.env.timeout(wait)
        with self.res.request(priority=self.prio) as req:
                yield req
                t_last=self.env.now
                if c_type==ML_STATE.FORWARD:
                    yield self.env.process(self.tile.forward_process())
                    self.trace.append((t_last,self.env.now,c_type))    
                    self.res_fd_cnt+=1
                    if self.next_core_id!=None and self.next_core_id!=[]:
                        task_info=self.__class__.__stage_id
                        pks=Packet('',self.o_shape)
                        yield self.env.process(noc.STAGE_PASS_process(pks,self.cur_core_id,self.next_core_id,task_info))
                elif c_type==ML_STATE.BACKWARD:
                    yield self.env.process(self.tile.backward_process())
                    self.trace.append((t_last,self.env.now,c_type))    
                    self.res_fd_cnt-=1
                    if self.next_core_id!=None and self.next_core_id!=[]:
                        task_info=self.__class__.__stage_id
                        pks=Packet('',self.i_shape)
                        yield self.env.process(noc.STAGE_PASS_process(pks,self.cur_core_id,self.next_core_id,task_info))
                else:
                    yield self.env.process(self.tile.update_process())
                    self.trace.append((t_last,self.env.now,c_type))      
class Pipeline():
    def __init__(self,env,mini_batch_size,micro_batch_size,stages:List[Stage],\
                 noc:wd,pipe_type:pipe_strategy=pipe_strategy.Megatron1F1B,train=True) -> None:
        #simpy env 
        self.env=env
        #pipeline info
        self.stages=stages
        self.stage_num=len(stages)
        self.noc=noc
        self.mini_batch=mini_batch_size
        self.micro_batch=micro_batch_size
        self.micro_batch_num=math.ceil(self.mini_batch/self.micro_batch)
        self.reg=[]
        self.train=train
        self.cur_fd_times=0
        self.cur_bd_times=0
        self.one_epoch_finish=simpy.Store(self.env,capacity=1)
        self.one_fd_finish=simpy.Store(self.env,capacity=1)
        self.one_data_fetch=simpy.Store(self.env,capacity=1)
        self.strategy=pipe_type
        self.boost_mode=False
        self.boost_times=3  if self.stages[0].tile.Analytical else 6
        self.__set_stage()
    def __set_stage(self):
        #TODO 需要检查device 在stage段无重复，否则映射不符合流水规则
        for i in range(self.stage_num):
            self.stages[i].init_info(self.micro_batch)
            self.reg.append(simpy.PriorityStore(self.env, capacity=1))
            if self.strategy==pipe_strategy.GPipe:
                self.stages[i].stage_info=[self.strategy,self.mini_batch,self.micro_batch]
            elif self.strategy==pipe_strategy.Megatron1F1B:
                self.stages[i].stage_info=[self.strategy,i,self.stage_num]
            elif self.strategy==pipe_strategy.Cerebras:
                self.stages[i].stage_info=[self.strategy,i ,self.stage_num]
            else:
                raise NotImplementedError
            self.stages[i].tile.mapping_analysis(self.stages[i].stage_info,self.stages[i].cur_core_id,self.stages[i].op_list,self.noc,self.train)
    def forward(self,times):
        with self.one_data_fetch.get() as get:
            a=yield get
            for i,stg in enumerate(self.stages):
                #print( hasattr(stg,"up_state"))
                yield self.env.process(stg.up_state(self.noc,c_type=ML_STATE.FORWARD,wait=1e-15))
                if self.strategy==pipe_strategy.Megatron1F1B:
                    if i==self.stage_num-2:
                        yield self.reg[i].put(1)
                    else :
                        self.reg[i].put(1)
                elif self.strategy==pipe_strategy.GPipe:
                    if i==self.stage_num-1:
                        self.cur_fd_times+=1
                    #print('self.cur_time',self.cur_time)
                    if self.cur_fd_times==times:
                        self.one_fd_finish.put(1)
                        #print('self.one_fd_finish.put(1)',self.cur_fd_times)   
                else:
                    raise NotImplementedError        
    def backward(self,times): 
        for i in range(self.stage_num-1,-1,-1):
            if self.strategy==pipe_strategy.Megatron1F1B:
                with self.reg[i].get() as get:
                    a=yield get
                    stg=self.stages[i]
                    yield self.env.process(stg.up_state(self.noc,c_type=ML_STATE.BACKWARD,wait=1e-15))  
                    if i==0:
                        self.cur_bd_times+=1
                    if self.cur_bd_times==times:
                        self.one_epoch_finish.put(1)
            elif  self.strategy==pipe_strategy.GPipe:  
                stg=self.stages[i]
                yield self.env.process(stg.up_state(self.noc,c_type=ML_STATE.BACKWARD,wait=1e-15))  
                if i==0:
                    self.cur_bd_times+=1
                if self.cur_bd_times==times:
                    self.one_epoch_finish.put(1)  
    def parameter_syn(self):
        while(True):
            with self.one_epoch_finish.get() as get:
                a=yield get
                for stg in self.stages:
                    self.env.process(stg.up_state(self.noc,c_type=ML_STATE.PARAM_SYNC,wait=1e-15))
                break
    def start(self):
        times=self.boost_times if self.boost_mode else self.micro_batch_num
        for i in range(times):
            task_info='input_data_fetch_'+str(i)
            i_shape=self.stages[0].i_shape
            with self.one_data_fetch.put(Packet(task_info,i_shape))as put:
                yield put
                yield self.env.process(self.noc.dram_read_group_process(i_shape,self.stages[0].cur_core_id,task_id=task_info,multicast=False))
                
    def register(self,boost_mode=True): 
        print('----------pipe_info----------')
        print('stage num={}, extute times={}'.format(len(self.stages),self.micro_batch_num))
        print('mini batch={}, micro batch={}'.format(self.mini_batch,self.micro_batch))
        self.boost_mode=boost_mode
        times=self.boost_times if self.boost_mode else self.micro_batch_num
        #self.boost_times=1 
        def all_backward(times):
            while(True):
                with self.one_fd_finish.get() as get:
                    a=yield get    
                    for i in range(times):
                        self.env.process(self.backward(times)) 
                break
        self.env.process(self.start())
        for i in range(times):
            self.env.process(self.forward(times))
        if self.train:
            if self.strategy==pipe_strategy.GPipe:  
                self.env.process(all_backward(times))
            elif self.strategy==pipe_strategy.Megatron1F1B:  
                for i in range(times):
                    self.env.process(self.backward(times))
            self.env.process(self.parameter_syn())
        
    def simpy_run(self,until_ms=2000):
        print('----------simpy_run----------')
        sim_start_t=time.time()
        print('start simpy simulation...')
        self.env.run(until=until_ms)
        sim_end_t=time.time()
        print('finish simpy simulation with {:.3f}s\n'.format(sim_end_t-sim_start_t))
    def status(self,path='./status/pipeline/',draw_pipe=True,write_log=False,clear=True):
        exe_mode='training' if self.train else  'inference'
        tm=time.strftime('_%m_%d_%H_%M_%S',time.localtime())
        name='pipeline'+str(tm)
        name_log=name+'.log'
        all_trace=[]
        pipe_endtime=0
        title=str(self.strategy) if self.train else 'Inference'
        for stage in self.stages:
            all_trace.append(stage.trace)
            if stage.trace[-1][1]>pipe_endtime:
                pipe_endtime=stage.trace[-1][1]
        #print(all_trace)
        #pipe_endtime=all_trace[0][-1][1]
        #print(all_trace[0])
        if self.boost_mode :
            #add boosted time
                max_unit_time_1F_1B=max_ave_1F_1B_time(all_trace,self.train)#all_trace[-1][1][1]-all_trace[-1][0][0]#
                #print(max_unit_time_1F1B)
                pipe_endtime=pipe_endtime+(self.micro_batch_num-self.boost_times)*max_unit_time_1F_1B 
        endtime_secs=pipe_endtime/1000
        endtime_days=endtime_secs/60/60/24
        if not os.path.exists(path):
            os.makedirs(path)
        elif clear:
            ls = os.listdir(path)
            for i in ls: 
                f_path = os.path.join(path, i)
                #print(f_path)    
                os.remove(f_path)
        if write_log:
            with open(path+name_log, 'w') as f:
                f.write(str(all_trace))
        if draw_pipe:
            draw_pipeline(all_trace,path=path,title=title,throughout=self.mini_batch/endtime_secs,name=name)
        print('{} ML {} pipeline endtime {:.4f} days [{:.4f}s]'.format(title,exe_mode,endtime_days,endtime_secs))
        print('{} ML {} pipeline throughout= {:.4f} sample/s'.format(title,exe_mode,self.mini_batch/endtime_secs))
        return endtime_days


