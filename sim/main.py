 
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import model_map as mp
import pipeline_copy as pipe
from ML import *
import simpy
if __name__ == '__main__':
    #TODO set config info by configparser
    Analytical=False
    model_list=['GPT3','BERT_LARGE','ResNet50']
    network_name=model_list[2]
    assert(network_name in model_list)
    cp_bound=1
    ResNet50_wafer_config={
        'wafer_name':'test',
        'tile_inter_shape':[5,4],#scale out dimension
        'tile_intra_shape':[4,4],
        'tile_intra_noc_bw_GB':1024*2*cp_bound,
        'tile_inter_noc_bw_GB':100*2*cp_bound,
        'tile_dram_bw_GB':25.6*32/16*cp_bound,#?
        'tile_dram_capacity_GB':48/16,#?
        'edge_die_dram_bw_GB':25.6*cp_bound,#?
        'clk_freq_GHz':1,
        'with_dram_per_tile':True,
        'Analytical':Analytical
        }  
    BERT_LARGE_wafer_config={
        'wafer_name':'test',
        'tile_inter_shape':[5,4],#scale out dimension
        'tile_intra_shape':[4,4],
        'tile_intra_noc_bw_GB':1024*2*cp_bound,
        'tile_inter_noc_bw_GB':100*2*cp_bound,
        'tile_dram_bw_GB':25.6*32/16*cp_bound,#?
        'tile_dram_capacity_GB':48/16,#?
        'edge_die_dram_bw_GB':25.6*cp_bound,#?
        'clk_freq_GHz':1,
        'with_dram_per_tile':True,
        'Analytical':Analytical
        }  
    scale_out_x=8
    scale_out_y=1
    GPT3_wafer_config={
        'wafer_name':'test',
        'tile_inter_shape':[5*scale_out_x,4*scale_out_y],#scale out dimension
        'tile_intra_shape':[4,4],
        'tile_intra_noc_bw_GB':1024*2,
        'tile_inter_noc_bw_GB':100*2,
        'tile_dram_bw_GB':25.6*32/16,#?
        'tile_dram_capacity_GB':48/16,#?
        'edge_die_dram_bw_GB':25.6,#?
        'clk_freq_GHz':1,
        'with_dram_per_tile':True,
        'Analytical':Analytical
        }      
    if network_name=='GPT3':
        wafer_config=GPT3_wafer_config
    elif network_name=='BERT_LARGE':
        wafer_config=BERT_LARGE_wafer_config
    elif network_name=='ResNet50':
        wafer_config=ResNet50_wafer_config
    else:
        raise NotImplementedError
    CNN_tile_config={
        'tile_name':'test',
        'sram_capacity_MB':3,
        'macs':8000,#@FP16 
        'freq_GHz':1,
        'with_dram':True,
        'opt':OPTIMIZER.SGD,#TODO
        'ZeRO':ZeRO_strategy.none ,#
        'Analytical':Analytical
        }  
    Tranformer_tile_config={
        'tile_name':'test',
        'sram_capacity_MB':3,
        'macs':8000,#@FP16 
        'freq_GHz':1,
        'with_dram':True,
        'opt':OPTIMIZER.ADAM,#TODO
        'ZeRO':ZeRO_strategy.ZeRO_3 ,#
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
    #3.mapping by hand
    model=CompGraph.gread(path='model',name=network_name)
    if network_name=='GPT3':
        stgs=mp.mapping_GPT3(env,model,Tranformer_tile_config,wd)
    elif network_name=='BERT_LARGE':
        stgs=mp.mapping_BERT_LARGE(env,model,Tranformer_tile_config,wd)
    elif network_name=='ResNet50':
        #print('00000')
        stgs=mp.mapping_ResNet50(env,model,CNN_tile_config,wd)
        #print('11111')
    else:
        raise NotImplementedError
    batch_size=model.batch_size

    STG_NUM=len(stgs)
    micro_batch=1#batch_size//STG_NUM
    pipe_sim=pipe.Pipeline(
        env=env,
        mini_batch_size=batch_size,
        micro_batch_size=micro_batch,#TODO
        stages=stgs,
        noc=wd,
        pipe_type=pipe_strategy.Megatron1F1B,#pipe_strategy.GPipe#
        train=False
        )
    pipe_sim.register(boost_mode=True)
    #5.simpy run  
    ONE_WEEK_MS=24*60*60*7*1000
    scale_sim_time=ONE_WEEK_MS*1000
    pipe_sim.simpy_run(until_ms=scale_sim_time)
    #6. log and info output
    pipe_sim.status(draw_pipe=True,clear=False,write_log=True)
    #res_type='edge_dram' or '3ddram' or 'noc' or 'all'
    #if not wafer_config['Analytical']:
    #    wd.resource_visualize(res_type='edge_dram',clear=True)








