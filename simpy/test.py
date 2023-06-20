
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import pipeline as pipe
from ML import *
import simpy

if __name__ == '__main__':
    
    #1.define simpy environment
    env=simpy.Environment()

    #2.define hardware
    #defualt:tile=Tile(with_dram=True)
    #256x16 tile
    a=[i for i in range(256)]

    wd=Wafer_Device(env,with_3ddram_per_tile=True,tile_inter_shape=[64,4],tile_intra_shape=[4,4])
    env.process(wd.tile_dram_access_process(10,63,'TEST_3DDRAM',DEBUG_MODE=True))
    env.process(wd.dram_read_group_process(1,group_id=[0],task_id='TEST',multicast=False))
    env.run(until=1000000)