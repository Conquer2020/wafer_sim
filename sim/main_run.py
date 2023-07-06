 
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import pipeline as pipe
from ML import *
import simpy
import math
if __name__ == '__main__':
    #0 TODO set config info by configparser
    wafer_config={
        'wafer_name':'test',
        'tile_inter_shape':[4,4],#scale out dimension
        'tile_intra_shape':[4,4],
        'tile_intra_noc_bw_GB':1024,
        'tile_inter_noc_bw_GB':1024*0.6*1000,
        'tile_dram_bw_GB':12288/16/8,
        'tile_dram_capacity_GB':6/16,
        'edge_die_dram_bw_GB':512*1000,
        'clk_freq_Ghz':1,
        'with_3ddram_per_tile':True
        }  
    tile_config={
        'tile_name':'tx8',
        'sram_capacity_MB':3,
        'macs':4000,
        'freq_GHz':1,
        'with_dram':True,
        'opt':OPTIMIZER.ADAM,
        'ZeRO':ZeRO_strategy.ZeRO_3 
        }  
    
    #1.define simpy environment
    env=simpy.Environment()
    #2.set hardware parameters
    wd=Wafer_Device(
        env=env,
        wafer_name=wafer_config['wafer_name'],
        tile_inter_shape=wafer_config['tile_inter_shape'],
        tile_intra_shape=wafer_config['tile_intra_shape'],
        tile_intra_noc_bw_GB=wafer_config['tile_intra_noc_bw_GB'],
        tile_inter_noc_bw_GB=wafer_config['tile_inter_noc_bw_GB'],
        with_3ddram_per_tile=wafer_config['with_3ddram_per_tile'],
        tile_dram_bw_GB=wafer_config['tile_dram_bw_GB'],
        tile_dram_capacity_GB=wafer_config['tile_dram_capacity_GB'],
        edge_die_dram_bw_GB=wafer_config['edge_die_dram_bw_GB'],
        clk_freq_Ghz=wafer_config['clk_freq_Ghz'],
        )
    #read ml compute graph from json file or define ml compute graph by yourself
    gpt_gp=CompGraph.gread(path='mljson',name='gpt-3.json')
    batch_size=gpt_gp.root.param_dim[0]
    #print(batch_size)
    #print(wd.tile_inter_noc_bw_GB)
    #3.mapping by hand
    #TODO mapping with graph arch info
    tiles_id=wd.device_list() 
    STG_NUM=16
    DATA_PARALLELISM=2
    tiles=[]
    for i in range(STG_NUM):  
        tiles.append(tiles_id[i::STG_NUM])
    Layers_num=len(gpt_gp)
    nums_per_stg=math.ceil(Layers_num/STG_NUM)

    j=0
    ops=[]
    ops_per_stg=[]
    for i,op_name in enumerate(gpt_gp.op_dict):
        d_size=len(tiles[j])
        dp=DATA_PARALLELISM
        mp=d_size//dp
        #make sure that product of model parallelism and data parallelism is equal to numbers of device 
        assert(mp*dp==d_size)
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
    #CompGraph.gwrite(gpt_gp,path='mljson',name='gpt_dp_test.json')

    #4.pipeline define and set
    stgs=[]
    for i in range(STG_NUM):
        last_core_id=None if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=None if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,tile_config,ops_per_stg[i],last_core_id,cur_core_id,next_core_id,noc=wd))

    micro_batch=batch_size//STG_NUM
    gpt_pipe_sim=pipe.Stages(
        env=env,
        mini_batch_size=batch_size,
        micro_batch_size=micro_batch,#TODO
        stages=stgs,
        noc=wd,
        pipe_type=pipe_strategy.Megatron1F1B
        )
    gpt_pipe_sim.pipeline_set(boost_mode=False)
  
    #5.simpy run  
    ONE_WEEK_MS=24*60*60*7*1000
    scale_sim_time=ONE_WEEK_MS*1000
    gpt_pipe_sim.simpy_run(until=scale_sim_time)

    #6. log and info output
    gpt_pipe_sim.pipeline_status(clear=False)

    #res_type='edge_dram' or '3ddram' or 'noc' or 'all'
    #wd.resource_visualize(res_type='edge_dram',clear=True)








