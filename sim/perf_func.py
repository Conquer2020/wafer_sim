
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
from model_map import mapping
import pipeline_copy as pipe
from ML import *
import simpy
import math

class Mapping():
    def __init__(self,env,graph,tile_config,wd,stage_num) -> None:
        self.env=env
        self.graph=graph
        self.tile_config=tile_config
        self.wd=wd
        self.stage_num=stage_num
    def _mapping(self,tiles:list):
        tiles_id=self.wd.device_list() 
        STG_NUM=len(tiles)
        DATA_PARALLELISM=1
        #tiles=[]
        for i in range(STG_NUM):
            #print(tiles_id[i::STG_NUM])  
            tiles.append(tiles_id[i::STG_NUM])
        Layers_num=len(self.graph)
        nums_per_stg=math.ceil(Layers_num/STG_NUM)
        #print(tiles)
        j=0
        ops=[]
        ops_per_stg=[]
        for i,op_name in enumerate(self.graph.op_dict):
            d_size=len(tiles[j])
            dp=DATA_PARALLELISM
            mp=d_size//dp
            assert mp*dp==d_size,'make sure that mp*dp=d_size'
            op=self.graph.op_dict[op_name]
            op.dpmap(device_id=tiles[j],p_sgy=[dp,mp])
            ops.append(op)
            if i % nums_per_stg==nums_per_stg-1:
                j+=1
                ops_per_stg.append(ops)
                ops=[]
        if ops!=[]:
            ops_per_stg[-1].append(op)
        #write graph with device to file
        #CompGraph.gwrite(gpt_gp,path='mljson',name='gpt_map.json')
        stgs=[]
        for i in range(STG_NUM):
            last_core_id=None if i==0 else tiles[i-1]
            cur_core_id=tiles[i]
            next_core_id=None if i==STG_NUM-1 else tiles[i+1]
            stgs.append(pipe.Stage(self.env,self.tile_config,ops_per_stg[i],last_core_id,cur_core_id,next_core_id,noc=self.wd))
        return stgs
    def perf_func_4ga(self,mapping_ori):
        def  mapping_decode(mapping_ori):
            stgs_device=[[]]*self.stage_num
            for i in stgs_device:
                for j in mapping_ori:
                    if j==i:
                        stgs_device[i].append(j)
        stgs_device=mapping_decode(mapping_ori)
        print(mapping_ori)
        print(stgs_device)
        stgs=self._mapping(stgs_device)
        STG_NUM=len(stgs)
        batch_size=self.graph.batch_size
        micro_batch=batch_size//STG_NUM
        gpt_pipe_sim=pipe.Pipeline(
            env=self.env,
            mini_batch_size=batch_size,
            micro_batch_size=micro_batch,#TODO
            stages=stgs,
            noc=self.wd,
            pipe_type=pipe_strategy.Megatron1F1B#pipe_strategy.GPipe#
            )
        gpt_pipe_sim.register(boost_mode=True)
        #5.simpy run  
        ONE_WEEK_MS=24*60*60*7*1000
        scale_sim_time=ONE_WEEK_MS*10
        gpt_pipe_sim.simpy_run(until_ms=scale_sim_time)
        #6. log and info output
        return gpt_pipe_sim.status(draw_pipe=False,clear=False)


