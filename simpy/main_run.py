
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import pipeline as pipe
from ML import *
import simpy
import math

if __name__ == '__main__':
    
    #1.define simpy environment
    env=simpy.Environment()

    #2.define hardware
    #defualt:tile=Tile(with_dram=True)
    #256x16 tile
    wd=Wafer_Device(env,with_3ddram_per_tile=True,tile_inter_shape=[4,4],tile_intra_shape=[4,4])
    tiles_id=wd.device_list() 

    #read ml compute graph from json file or define ml compute graph by yourself
    gp=CompGraph.gread(path='mljson',name='gpt-3.json')
    batch_size=gp.root.param_dim[0]
    #print(batch_size)

    #3.mapping by hand
    #TODO mapping with graph arch info
    STG_NUM=16
    DATA_PARALLELISM=4
    tiles=[]
    for i in range(STG_NUM):  
        tiles.append(tiles_id[i::STG_NUM])
    Layers_num=len(gp)
    nums_per_stg=math.ceil(Layers_num/STG_NUM)

    j=0
    ops=[]
    ops_per_stg=[]
    for i,op_name in enumerate(gp.op_dict):
        d_size=len(tiles[j])
        dp=DATA_PARALLELISM
        mp=d_size//dp
        #make sure that product of model parallelism and data parallelism is equal to numbers of device 
        assert(mp*dp==d_size)
        op=gp.op_dict[op_name]
        op.dpmap(device_id=tiles[j],p_sgy=[dp,mp])
        ops.append(op)
        if i % nums_per_stg==nums_per_stg-1:
            j+=1
            ops_per_stg.append(ops)
            ops=[]
    if ops!=[]:
        ops_per_stg[-1].append(op)
    #write graph with device to file
    #CompGraph.gwrite(gp,path='mljson',name='gpt_dp_test.json')

    #4.pipeline define and set
    stgs=[]
    for i in range(STG_NUM):
        last_core_id=[] if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=[] if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,ops_per_stg[i],last_core_id,cur_core_id,next_core_id))
    #micro_batch=batch_size//STG_NUM
    micro_batch=batch_size//3
    stages=pipe.Stages(env=env,mini_batch_size=batch_size,micro_batch_size=micro_batch,stages=stgs,noc=wd)
    stages.pipeline_set()

    #5.simpy run  
    one_weeks_ms=24*60*60*7*1000
    scale_sim_time=one_weeks_ms*1000000
    stages.simpy_run(until=scale_sim_time)

    #6. log and info output
    stages.pipeline_status()
    #for index,dram_res in enumerate(wd.edge_dram_resource):
    #wd.visualize_resource(dram_res.access_resource,res_type='edge_dram',name=str(index))








