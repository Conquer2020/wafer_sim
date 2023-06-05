import math
import simpy
from  util import BaseEnum as Enum
from typing import List,Optional
from util import *
from wafer_device import Wafer_Device as wd
from wafer_device import Packet


import ML
from comp_graph import CompGraph,OpNode
from op_pd import CommOp
dataflow=Enum('dataflow',('IS','WS','OS'))
comp_model=Enum('comp_model',('simple','SCALE_SIM'))
sram_strategy=Enum('sram_strategy',('cache','weight','ACT','ACT_weight'))
recompute_strategy=Enum('recompute_strategy',('none','half','all'))
traffic=Enum('traffic',('act_store','act_fetch','comm','act_fd','grad_bd','wt_load','wt_store'))

class Tile():# for compute process
    def __init__(self,tile_name='tx8',
                 sram_capacity_MB=3,macs=4000,freq_GHz=1,\
                 with_dram=True,dram_bw_GB=12288/16/8,dram_capacity_GB=6/16,
                    opt=ML.OPTIMIZER,ZeRO=ML.ZeRO_strategy.none) -> None:
        #info
        self.tile_name=tile_name

        # define store byte
        self.act_bytes=ML.BYTES['FP16'] if opt!=ML.OPTIMIZER.NONE else ML.BYTES['NONE']
        self.weight_bytes=ML.BYTES['FP16']#+ML.BYTES['FP32']
        self.grad_bytes=ML.BYTES['FP16']#+ML.BYTES['FP32']
        self.opt_states_bytes=ML.BYTES['NONE'] if opt!=ML.OPTIMIZER.ADAM else 3*ML.BYTES['FP32']
        self.buffer_bytes=ML.BYTES['FP16']

        #define buffer & sram size & dram_size
        self.sram_capacity=sram_capacity_MB
        self.with_dram=with_dram
        self.dram_bw=dram_bw_GB
        self.dram_capacity=dram_capacity_GB
        self.ifmap_size=0
        self.weight_size=0
        self.ofmap_size=0
        self.max_dim=1024

        #define compute 
        self.macs=macs
        self.cp_byte=ML.BYTES['FP16']
        self.array_group=[2,2]
        self.array_shape=self.__shape_suppose(self.macs)
        self.cp_model=comp_model.SCALE_SIM
        self.freq=freq_GHz
        self.dataflow=dataflow.IS
        #self.tflops=self.macs*2*self.freq

        self.ZeRO=ZeRO

    def __shape_suppose(self,size):
        R=0
        C=0
        if size==8000 or size==8192:
            R=128
            C=64
        if size==4000 or size==4096:
            R=64
            C=64
        if size==1000 or size== 1024:
            R=32
            C=32
        return [R,C]
    def compute_cycles(self,param:List[int]):
        '''
        define matrix multiply [m,n,k]: (m,k)*(k,n)=(m,n)
        SR:m SC:n T: k
        PE array num:PR*RC
        each PE macs units: R*C
        #reference: https://github.com/ARM-software/SCALE-Sim
        '''
        [SR,SC,T]=param
        [R,C]=self.array_shape
        [PR,PC]=self.array_group
        if self.cp_model==comp_model.SCALE_SIM:
            sr=math.ceil(SR/PR)
            sc=math.ceil(SC/PC)
            return (2*R+C+T-2)*math.ceil(sr/R)*math.ceil(sc/C)
        elif self.cp_model==comp_model.simple:
            return T*math.ceil(SR/(R*PR))*math.ceil(SC/(C*SC))
        else :
            raise NotImplementedError
    #TODO 
    #each tile group may process one subgraph rather than one op
    #if there is one simple op，it is not nesscessary to use reccompute strategy
    #@fangjh21.20230602
    @staticmethod
    def mapping_analysis(tile,stage_info,device,op_list:List[OpNode],wd1:wd):
        #init 
        device_gp=device
        acc_op_weight_size=0
        acc_op_intra_act_size=0
        acc_op_input_act_size=0
        #acc_op_output_act_size=0          # mbytes(op_list[-1].i_shape) no store
        df0=dataflow.WS
        ss1=sram_strategy.cache
        rs2=recompute_strategy.none
        zs3=ML.ZeRO_strategy.none

        [pipe_strategy,info1,info2]=stage_info

        #dram/sram allocation for each op with parallism  and recompute strategy
        # @fangjh21.20230602
        for op in op_list:
            acc_op_input_act_size+=mbytes(op.i_shape)
            acc_op_weight_size+=op.param_size_m
            acc_op_intra_act_size+=op.intra_act_size_m
            
        act_times_coe=0
        if pipe_strategy==ML.pipe_strategy.GPipe:
            act_times_coe=info1
        elif pipe_strategy==ML.pipe_strategy.Cerebras:
            act_times_coe=2*(info2-info1)
        elif pipe_strategy==ML.pipe_strategy.Megatron1F1B:
            #TODO 激活生存时长不完全相符
            act_times_coe=(info2-info1) #@fangjh21.20230602 
        else:
            raise NotImplementedError
        mem_occupy_by_weight_and_states=acc_op_weight_size*(tile.weight_bytes+tile.opt_states_bytes)
        mem_occupy_by_act_with_stageflow=act_times_coe*(acc_op_input_act_size+acc_op_intra_act_size)*tile.act_bytes
        
        if tile.with_dram:
            #match dram setting
            assert(wd1.dram_per_tile_resource!=[])
            total_mem_size=tile.dram_capacity #GB
            ss1=sram_strategy.cache
            df0=dataflow.WS
            if mem_occupy_by_weight_and_states+mem_occupy_by_act_with_stageflow<total_mem_size:
                rs2=recompute_strategy.none   
            else:
                rs2=recompute_strategy.all   
        else:
            total_mem_size=tile.sram_capacity #MB
            rs2=recompute_strategy.none
            if mem_occupy_by_weight_and_states+mem_occupy_by_act_with_stageflow<total_mem_size:
                ss1=sram_strategy.ACT_weight
                df0=dataflow.IS
            elif mem_occupy_by_act_with_stageflow<total_mem_size:
                ss1=sram_strategy.weight
                df0=dataflow.IS
            elif mem_occupy_by_weight_and_states<total_mem_size:
                ss1=sram_strategy.ACT
                df0=dataflow.WS
            else:
                rs2=recompute_strategy.all
                ss1=sram_strategy.cache
                df0=dataflow.OS
        return [df0,ss1,rs2]

    @staticmethod
    def execute_comm_process(tile,env,comm_op:CommOp,wd1:wd):
        yield env.timeout(5)
    @staticmethod
    def execute_forward_process(tile,env,map_ana:list,device:List[int],op:OpNode,wd1:wd):
        yield env.timeout(5)
    @staticmethod 
    def execute_backward_process(tile,env,map_ana,device:List[int],op:OpNode,wd1:wd):
        yield env.timeout(10)



