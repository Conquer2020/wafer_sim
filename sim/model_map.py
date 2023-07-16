
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import pipeline as pipe
from ML import *
import simpy
import math


def mapping(env:simpy.Environment,gpt_gp:CompGraph,tile_config:dict,wd:Wafer_Device):
    tiles_id=wd.device_list() 
    STG_NUM=96
    DATA_PARALLELISM=1
    tiles=[]
    for i in range(STG_NUM):
        print(tiles_id[i::STG_NUM])  
        tiles.append(tiles_id[i::STG_NUM])
    Layers_num=len(gpt_gp)
    nums_per_stg=math.ceil(Layers_num/STG_NUM)
    #print(tiles)
    j=0
    ops=[]
    ops_per_stg=[]
    for i,op_name in enumerate(gpt_gp.op_dict):
        d_size=len(tiles[j])
        dp=DATA_PARALLELISM
        mp=d_size//dp
        assert mp*dp==d_size,'make sure that mp*dp=d_size'
        op=gpt_gp.op_dict[op_name]
        op.dpmap(device_id=tiles[j],p_sgy=[dp,mp])
        ops.append(op)
        if i % nums_per_stg==nums_per_stg-1:
            j+=1
            ops_per_stg.append(ops)
            ops=[]
    if ops!=[]:
        ops_per_stg[-1].append(op)
    #write graph with device to file
    CompGraph.gwrite(gpt_gp,path='mljson',name='gpt_map.json')
    stgs=[]
    for i in range(STG_NUM):
        last_core_id=None if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=None if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,tile_config,ops_per_stg[i],last_core_id,cur_core_id,next_core_id,noc=wd))
    return stgs
