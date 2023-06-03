
from wafer_device import Wafer_Device 
from tile_dataflow import Tile
from comp_graph import CompGraph,OpNode
import pipeline as pipe

import ML

import simpy
import time

if __name__ == '__main__':
    #define simpy environment
    env=simpy.Environment()
    #define hardware
    tile=Tile(with_dram=True)
    wd=Wafer_Device(env,with_3ddram_per_tile=True,tile_inter_shape=[2,2],tile_intra_shape=[2,2])
    #print(wd.device_list())
    #define op and ML compute graph

    #TODO stages,device_group=map(graph,device)
    batch_size=4
    op1=OpNode(op_type=ML.OP.Linear,op_param=[batch_size,256,128,512],hint_name='s1')
    op2=OpNode(op_type=ML.OP.Linear,op_param=[batch_size,64,256,128],hint_name='s2')
    op3=OpNode(op_type=ML.OP.Linear,op_param=[batch_size,128,64,256],hint_name='s3')
    op4=OpNode(op_type=ML.OP.Linear,op_param=[batch_size,1024,128,64],hint_name='s4')
    gp=CompGraph()
    gp.AddEdge(op1)
    gp.AddEdge(op2)
    gp.AddEdge(op3,op2)
    gp.AddEdge(op4,op1)
    gp.AddEdge(op3,op4)

    #mapping by hand
    tiles_1=[0,1,2,3]
    tiles_2=[4,5]
    tiles_3=[6,7,10,11,14,15]
    tiles_4=[12,13]
    op1.dpmap(device_id=tiles_1,parallel_dim=[1])
    op2.dpmap(device_id=tiles_2,parallel_dim=[1])
    op3.dpmap(device_id=tiles_3,parallel_dim=[1])
    op4.dpmap(device_id=tiles_4,parallel_dim=[1])
    CompGraph.gwrite(gp,path='mljson',name='gh.json')
    #00
    stg0=pipe.Stage(tile,env,[op1],last_core_id=[],cur_core_id=tiles_1,next_core_id=tiles_2)
    stg1=pipe.Stage(tile,env,[op2],last_core_id=tiles_1,cur_core_id=tiles_2,next_core_id=tiles_3)
    stg2=pipe.Stage(tile,env,[op3],last_core_id=tiles_2,cur_core_id=tiles_3,next_core_id=tiles_4)
    stg3=pipe.Stage(tile,env,[op4],last_core_id=tiles_3,cur_core_id=tiles_4,next_core_id=[])
    stages=pipe.Stages(env=env,mini_batch_size=batch_size,micro_batch_size=1,stages=[stg0,stg1,stg2,stg3],noc=wd)
    stages.pipeline()

    #TODO 
    #1.原语构建 ,SRAM,DRAM单独act访存构建
    #2.并行解析，插入通信原语
    #3.图解析   
    sim_start_t=time.time()
    print('start simpy simulation...')
    env.run(until=2000)
    sim_end_t=time.time()
    print('finish simpy simulation with {:.3f}s\n'.format(sim_end_t-sim_start_t))
    stages.pipe_status(path='./pic/')
    for index,dram_res in enumerate(wd.edge_dram_resource):
        wd.visualize_resource(dram_res.access_resource,res_type='edge_dram',name=str(index))








