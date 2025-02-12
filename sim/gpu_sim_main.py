 
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import model_map as mp
import pipeline_copy as pipe
from ML import *
import simpy
if __name__ == '__main__':
    #TODO set config info by configparser
    Analytical=True
    model_name='T_18B'
    MODEL_CFG_TPD_3D={'T_18B':[8,1,32],
                      'T_39B':[8,2,32],
                      'T_76B':[8,4,32],
                      'T_145B':[8,8,24],
                      'T_310B':[8,16,15],
                      'T_530B':[8,35,9],
                      'T_1T':[8,64,6],
                      }
    cp_bound=1
    [TP,PP,DP]=MODEL_CFG_TPD_3D[model_name]
    parallelism_3d=MODEL_CFG_TPD_3D[model_name]
    A100_CLUSTER_CFG={
        'wafer_name':'test',
        'tile_inter_shape':[PP,DP],#scale out dimension
        'tile_intra_shape':[4,2],
        'tile_intra_noc_bw_GB':600*1000,
        'tile_inter_noc_bw_GB':600/8*1000,
        'tile_dram_bw_GB':2039*4*2*1000 ,
        'tile_dram_capacity_GB':80*4*2,
        'edge_die_dram_bw_GB':64*1000 ,
        'clk_freq_GHz':1,
        'with_dram_per_tile':True,
        'Analytical':Analytical
        }  

    CFG=A100_CLUSTER_CFG
    A100_DIE_CFG={
        'tile_name':'test',
        'sram_capacity_MB':3,
        'macs':624/2*1000,#@FP16 
        'freq_GHz':1,
        'with_dram':True,
        'opt':OPTIMIZER.ADAM,#TODO
        'ZeRO':ZeRO_strategy.none ,#
        'Analytical':Analytical
        }  
    #1.define simpy environment
    env=simpy.Environment()
    #2.set hardware parameters
    wd=Wafer_Device(
        env=env,
        wafer_name=CFG['wafer_name'],
        tile_inter_shape=CFG['tile_inter_shape'],
        tile_intra_shape=CFG['tile_intra_shape'],
        tile_intra_noc_bw_GB=CFG['tile_intra_noc_bw_GB'],
        tile_inter_noc_bw_GB=CFG['tile_inter_noc_bw_GB'],
        with_dram_per_tile=CFG['with_dram_per_tile'],
        tile_dram_bw_GB=CFG['tile_dram_bw_GB'],
        tile_dram_capacity_GB=CFG['tile_dram_capacity_GB'],
        edge_die_dram_bw_GB=CFG['edge_die_dram_bw_GB'],
        clk_freq_GHz=CFG['clk_freq_GHz'],
        Analytical=CFG['Analytical'],
        )

    #read ml compute graph from json file or define ml compute graph by yourself
    #3.mapping by hand
    model=CompGraph.gread(path='model',name=model_name)
    stgs=mp.mapping_Megatron_LM(env,model,A100_DIE_CFG,wd,parallelism_3d)  
    batch_size=model.batch_size
    STG_NUM=len(stgs)  
    micro_batch=batch_size//STG_NUM
    pipe_sim=pipe.Pipeline(
        env=env,
        mini_batch_size=batch_size,
        micro_batch_size=micro_batch,#TODO
        stages=stgs,
        noc=wd,
        pipe_type=pipe_strategy.Megatron1F1B,#pipe_strategy.GPipe#
        train=True
        )
    pipe_sim.register(boost_mode=True)
    #5.simpy run  
    ONE_WEEK_MS=24*60*60*7*1000
    scale_sim_time=ONE_WEEK_MS*1000
    pipe_sim.simpy_run(until_ms=scale_sim_time)
    #6. log and info output
    pipe_sim.status(draw_pipe=True,clear=True,write_log=False)
    #res_type='edge_dram' or '3ddram' or 'noc' or 'all'
    #if not CFG['Analytical']:
    #    wd.resource_visualize(res_type='edge_dram',clear=True)








