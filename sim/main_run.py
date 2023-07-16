 
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
from model_map import mapping
import pipeline as pipe
from ML import *
import simpy
if __name__ == '__main__':
    #0 TODO set config info by configparser
    wafer_config={
        'wafer_name':'test',
        'tile_inter_shape':[1,4],#scale out dimension
        'tile_intra_shape':[4,4],
        'tile_intra_noc_bw_GB':1024,
        'tile_inter_noc_bw_GB':1024*0.6,
        'tile_dram_bw_GB':12288/16/8,
        'tile_dram_capacity_GB':6/16,
        'edge_die_dram_bw_GB':512,
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
    #3.mapping by hand
    stgs=mapping(env,gpt_gp,tile_config,wd)
    STG_NUM=len(stgs)
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








