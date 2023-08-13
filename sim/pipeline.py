import simpy
import math
import time
from typing import List,Optional

from util import *
from tile_dataflow import Tile
from wafer_device import Wafer_Device as wd
from wafer_device import Packet
from ML import *
#TODO 修改流水线调度策略
# reference Megatron2.0 
# @fangjh21.20230602
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
        #self.init_info()

        self.last_core_id=last_core_id
        self.cur_core_id=cur_core_id
        self.next_core_id=next_core_id
        self.stage_info=[]
        self.map_ana=[]
        #simpy env 
        self.worker= simpy.Resource(env, capacity=1)
        self.trace=[]
        
        self.__class__.__stage_id+=1
        
    def init_info(self,micro_batch):
        for op in self.op_list:
            op.param_dim[0]=micro_batch
            op.update()
        self.i_shape=self.op_list[0].i_shape
        self.o_shape=self.op_list[-1].o_shape
        
    def stage_forward_process(self,last_q:simpy.Store,next_q:simpy.Store,env:simpy.Environment,noc:wd):
        def pro():
            with self.worker.request() as req:
                yield req
                pks = yield last_q.get()
                #print('i_shape',self.i_shape)
                #print('pks_shape',pks.shape)
                assert(self.i_shape==pks.shape)
                t_last=env.now
                #print('time:{},start forward'.format(round(env.now, 2)))
                #TODO 修改为数据流执行
                #yield env.timeout(20)
                #print('time:{},start forward'.format(round(env.now, 2)))
                yield env.process(self.tile.execute_forward_process())
                #print('time:{},end forward'.format(round(env.now, 2)))
                self.trace.append((t_last,env.now,0))
            if self.next_core_id!=None and self.next_core_id!=[]:
                task_info=self.__class__.__stage_id
                yield env.process(noc.STAGE_PASS_process(pks,self.cur_core_id,self.next_core_id,task_info))
            else:
                pass
        while True:
                yield env.process(pro())
                yield next_q.put(Packet('',self.o_shape))
                break
    def stage_backward_process(self,last_q:simpy.Store,next_q:simpy.Store,env:simpy.Environment,noc:wd):
        def pro():
            with self.worker.request() as req:
                yield req
                pks = yield next_q.get()
                #print('o_shape',self.o_shape)
                #print('pks_shape',pks.shape)
                assert(self.o_shape==pks.shape)
                t_last=env.now
                #TODO 修改为数据流执行
                yield env.process(self.tile.execute_backward_process())
                self.trace.append((t_last,env.now,1))
            if self.last_core_id!=None and self.last_core_id!=[]:
                task_info=self.__class__.__stage_id
                yield env.process(noc.STAGE_PASS_process(pks,self.cur_core_id,self.last_core_id,task_info))
            else:
                pass
        while True:
                yield env.process(pro())
                yield last_q.put(Packet('',self.i_shape))
                break
class Stages():
    def __init__(self,env,mini_batch_size,micro_batch_size,stages:List[Stage],\
                 noc:wd,pipe_type:pipe_strategy=pipe_strategy.Megatron1F1B) -> None:
        #simpy env 
        self.env=env
        self.loss_q=simpy.Store(env=self.env,capacity=1) 
        self.f_q=[]
        self.b_q=[]

        #pipeline info
        self.stages=stages
        self.pipe_type=pipe_type
        self.noc=noc
        self.mini_batch=mini_batch_size
        self.micro_batch=micro_batch_size
        self.pipe_times=math.ceil(self.mini_batch/self.micro_batch)
        self.__set_stage()

        self.boost_mode=False
        self.boost_times=4


    def __set_stage(self):
        #TODO 需要检查device 在stage段无重复，否则映射不符合流水规则
        stages_len=len(self.stages)
        for i in range(stages_len):
            self.stages[i].init_info(self.micro_batch)
            if self.pipe_type==pipe_strategy.GPipe:
                self.stages[i].stage_info=[self.pipe_type,self.mini_batch,self.micro_batch]
            elif self.pipe_type==pipe_strategy.Megatron1F1B:
                self.stages[i].stage_info=[self.pipe_type,i,stages_len]
            elif self.pipe_type==pipe_strategy.Cerebras:
                self.stages[i].stage_info=[self.pipe_type,i ,stages_len]
            else:
                raise NotImplementedError
            self.stages[i].tile.mapping_analysis(self.stages[i].stage_info,self.stages[i].cur_core_id,self.stages[i].op_list,self.noc)
    def pipeline_execute_forward_process(self):
        def pro():
            stage_len=len(self.stages)
            for i in range(stage_len):
                yield self.env.process(self.stages[i].stage_forward_process(self.f_q[i],self.f_q[i+1],self.env,self.noc))
                #print('stage {} @ {:.3f} ms'.format(i,self.env.now))
            #print('finish forward @ {:.3f} ms'.format(self.env.now))
        yield self.env.process(pro())
        
    def pipeline_execute_backward_process(self): 
        def pro():
            stage_len=len(self.stages)
            for i in range(stage_len-1,-1,-1):
                yield self.env.process(self.stages[i].stage_backward_process(self.b_q[i],self.b_q[i+1],self.env,self.noc))
            #TODO bug  
            #print('finish backward @ {:.3f} ms'.format(self.env.now))
        with self.f_q[len(self.stages)].get() as get:
            a=yield get
            yield self.b_q[len(self.stages)].put(a)
            yield self.env.process(pro())
            
    def start(self):
        #TODO 修改 DP相关
        #TODO 
        times=self.boost_times if self.boost_mode else self.pipe_times
        for i in range(times):
            task_info='input_data_fetch_'+str(i)
            i_shape=self.stages[0].i_shape
            with self.f_q[0].put(Packet(task_info,i_shape)) as put:
                #yield self.env.timeout(0)
                yield put
                yield self.env.process(self.noc.dram_read_group_process(i_shape,self.stages[0].cur_core_id,task_id=task_info,multicast=False))
                
    def pipeline_set(self,boost_mode=True): 
        def Megatron1F1B():
            for _ in range(len(self.stages)+1):
                self.f_q.append(simpy.Store(self.env,capacity=1))
                self.b_q.append(simpy.Store(self.env,capacity=1))
            self.env.process(self.start())
            times=self.boost_times if self.boost_mode else self.pipe_times
            for i in range(times):
                self.env.process(self.pipeline_execute_forward_process())
                self.env.process(self.pipeline_execute_backward_process())
        def Gpipe():
            pass
        def Cerebras():
            pass
        print('----------pipe_info----------')
        print('stage num={}, extute times={}'.format(len(self.stages),self.pipe_times))
        print('mini batch={}, micro batch={}'.format(self.mini_batch,self.micro_batch))
        self.boost_mode=boost_mode
        self.boost_times=4
        if self.pipe_type==pipe_strategy.Megatron1F1B:
            Megatron1F1B()
        elif self.pipe_type==pipe_strategy.GPipe:
            Gpipe()
        elif self.pipe_type==pipe_strategy.GPipe:
            Cerebras()


    def simpy_run(self,until=2000):
        print('----------simpy_run----------')
        sim_start_t=time.time()
        print('start simpy simulation...')
        self.env.run(until=until)
        sim_end_t=time.time()
        print('finish simpy simulation with {:.3f}s\n'.format(sim_end_t-sim_start_t))
    def pipeline_status(self,path='./status/pipeline/',draw_pipe=True,write_log=False,clear=True):
        tm=time.strftime('_%m_%d_%H_%M_%S',time.localtime())
        name='pipeline'+str(tm)
        name_log=name+'.log'
        all_trace=[]
        title=str(self.pipe_type)
        for stage in self.stages:
            all_trace.append(stage.trace)

        #add boosted time
        pipe_endtime=all_trace[0][-1][1]
        unit_time_1F1B=all_trace[-1][1][1]-all_trace[-1][0][0]
        #print(unit_time_1F1B)
        if self.boost_mode:
            pipe_endtime=pipe_endtime+(self.pipe_times-self.boost_times)*unit_time_1F1B 
        endtime_days=pipe_endtime/1000/60/60/24
        endtime_secs=pipe_endtime/1000
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
            draw_pipeline(all_trace,path=path,title=title,endtime=endtime_days,name=name)
        print('{} ML training pipeline endtime {:.1f} days [{:.1f}s]'.format(title,endtime_days,endtime_secs))
        return endtime_days


