from wafer_device import Wafer_Device 
from comp_graph import CompGraph
from model_map import mapping
import pipeline_copy as pipe
from ML import *
import simpy
import perf_func as pf
from algo import ga
if __name__ == '__main__':
    #TODO set config info by configparser
    Analytical=False
    wafer_config={
        'wafer_name':'test',
        'tile_inter_shape':[4,4],#scale out dimension
        'tile_intra_shape':[4,4],
        'tile_intra_noc_bw_GB':256,
        'tile_inter_noc_bw_GB':256*0.6,
        'tile_dram_bw_GB':12288/16/8,
        'tile_dram_capacity_GB':6/16,
        'edge_die_dram_bw_GB':512,
        'clk_freq_GHz':1,
        'with_dram_per_tile':True,
        'Analytical':Analytical
        }  
    tile_config={
        'tile_name':'tx8',
        'sram_capacity_MB':3,
        'macs':4000,
        'freq_GHz':1,
        'with_dram':True,
        'opt':OPTIMIZER.ADAM,
        'ZeRO':ZeRO_strategy.ZeRO_3 ,
        'Analytical':Analytical
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
        with_dram_per_tile=wafer_config['with_dram_per_tile'],
        tile_dram_bw_GB=wafer_config['tile_dram_bw_GB'],
        tile_dram_capacity_GB=wafer_config['tile_dram_capacity_GB'],
        edge_die_dram_bw_GB=wafer_config['edge_die_dram_bw_GB'],
        clk_freq_GHz=wafer_config['clk_freq_GHz'],
        Analytical=wafer_config['Analytical'],
        )
    #read ml compute graph from json file or define ml compute graph by yourself
    gpt_gp=CompGraph.gread(path='mljson',name='gpt-3.json')
    mp=pf.Mapping(env,gpt_gp,tile_config,wd,stage_num=32)
    algo=ga.GA(pop_num=20,max_gen=20,p_m=0.1,p_c=0.5)
    algo.Init_pop(p_dims=[2]*32,stg_num=32,d_num=256)
    algo.Evolution(mp.perf_func_4ga)
    #print(test.pop_best)
    #plt.plot(test.max_perf_trace)
    #plt.plot(test.min_perf_trace)
    algo.show()








