
from wafer_device import Wafer_Device 
from tile_dataflow import Tile
from comp_graph import CompGraph,OpNode
import pipeline as pipe

from ML import *

import simpy
import time
import math

if __name__ == '__main__':
    
    #1.define simpy environment
    env=simpy.Environment()

    #2.define hardware
    #defualt:tile=Tile(with_dram=True)
    #256x16 tile
    wd=Wafer_Device(env,with_3ddram_per_tile=True,tile_inter_shape=[64,4],tile_intra_shape=[4,4])
    tiles_id=wd.device_list() 

    #read ml compute graph from json file or define ml compute graph by yourself
    gp=CompGraph.gread(path='mljson',name='gpt-3.json')
    batch_size=gp.root.param_dim[0]


    #3.mapping by hand
    #TODO mapping with graph arch info

    STG_NUM=16
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
        dp=2
        mp=d_size//2
        assert(mp*dp==d_size)
        op=gp.op_dict[op_name]
        op.dpmap(device_id=tiles[j],p_sgy=[dp,mp])
        if i % nums_per_stg==nums_per_stg-1:
            j+=1
            ops.append(op)
            ops_per_stg.append(ops)
            ops=[]
    if ops!=[]:
        ops_per_stg[-1].append(op)

    #CompGraph.gwrite(gp,path='mljson',name='gpt_dp_test.json')

    #4.pipeline define
    stgs=[]
    for i in range(STG_NUM):
        last_core_id=[] if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=[] if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,ops_per_stg[i],last_core_id,cur_core_id,next_core_id))
    stages=pipe.Stages(env=env,mini_batch_size=batch_size,micro_batch_size=batch_size//10,stages=stgs,noc=wd)
    stages.pipeline()

    #5.simpy run  
    sim_start_t=time.time()
    print('start simpy simulation...')
    env.run(until=2000)
    sim_end_t=time.time()
    print('finish simpy simulation with {:.3f}s\n'.format(sim_end_t-sim_start_t))
    stages.pipe_status(path='./pic/')
    #for index,dram_res in enumerate(wd.edge_dram_resource):
    #wd.visualize_resource(dram_res.access_resource,res_type='edge_dram',name=str(index))








