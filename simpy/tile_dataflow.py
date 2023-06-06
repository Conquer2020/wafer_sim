import math
import simpy
from  util import BaseEnum as Enum
from typing import List,Optional
from util import *
from wafer_device import Wafer_Device as wd
from wafer_device import Packet

import numpy as np
from ML import *
from comp_graph import CompGraph,OpNode
from op_pd import CommOp

class Tile():# for compute process
    def __init__(self,tile_name='tx8',
                 sram_capacity_MB=3,macs=4000,freq_GHz=1,\
                 with_dram=True,dram_bw_GB=12288/16/8,dram_capacity_GB=6/16,
                    opt=OPTIMIZER,ZeRO=ZeRO_strategy.ZeRO_2) -> None:
        #info
        self.tile_name=tile_name

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
        self.array_group=[2,2]
        self.array_shape=self.__shape_suppose(self.macs)
        self.cp_model=comp_model.SCALE_SIM
        self.freq=freq_GHz
        self.dataflow=dataflow.IS
        #self.tflops=self.macs*2*self.freq

        self.ZeRO=ZeRO
        self.opt=opt
        # define store byte
        self.act_bytes=0
        self.wsg_bytes=[0,0,0]
        self.buffer_bytes=0
        self.comm_bytes=0
        self.__set_bytes()

    def __set_bytes(self):
        #TODO Mixed-precision is popular in  ML training process.
        #However,many AI archs have their float numberbprecision like TF32(Nvdia),CFP16(Dojo),etc.
        #@fangjh21.20230606
        print('Mixed-precision')
        self.cp_bytes=BYTES['FP16']
        self.act_bytes=BYTES['FP16'] if self.opt!=OPTIMIZER.NONE else BYTES['NONE']
        w=BYTES['FP16']+BYTES['FP32']
        g=BYTES['FP16']+BYTES['FP32']
        s=BYTES['NONE'] if self.opt!=OPTIMIZER.ADAM else 2*BYTES['FP32']
        self.wsg_bytes=[w,s,g]
        self.buffer_bytes=BYTES['FP16']
        self.comm_bytes=BYTES['FP16']

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
        acc_op_wsg_size=0
        acc_op_intra_act_size=0
        acc_op_input_act_size=0
        #acc_op_output_act_size=0          # mulc(op_list[-1].i_shape) no store
        df0=dataflow.WS
        ss1=sram_strategy.cache
        rs2=recompute_strategy.none
        zs3=tile.self.ZeRO
        [pipe_strategy,info1,info2]=stage_info
        #input_act_size=mulc(op_list[0].i_shape)
        #ouput_act_size=mulc(op_list[-1].o_shape)

        #dram/sram allocation for each op with parallism  and recompute strategy
        # @fangjh21.20230602
        for op in op_list:
            op.set_ZeRO(zs3)
            temp=np.array(op.w_s_g_size_m)*np.array(tile.wsg_bytes)
            acc_op_wsg_size+=mulc(temp.tolist())
            acc_op_intra_act_size+=op.intra_act_size_m*tile.act_bytes
            
        act_times_coe=0
        if pipe_strategy==pipe_strategy.GPipe:
            act_times_coe=info1
        elif pipe_strategy==pipe_strategy.Cerebras:
            act_times_coe=2*(info2-info1)
        elif pipe_strategy==pipe_strategy.Megatron1F1B:
            #TODO 激活生存时长不完全相符
            act_times_coe=(info2-info1) #@fangjh21.20230602 
        else:
            #TODO 
            raise NotImplementedError
        mem_occupy_by_wsg=acc_op_wsg_size
        mem_occupy_by_act_stage=act_times_coe*acc_op_intra_act_size
        
        if tile.with_dram:
            #match dram setting
            assert(wd1.dram_per_tile_resource!=[])
            total_mem_size=tile.dram_capacity #GB
            ss1=sram_strategy.cache
            df0=dataflow.WS
            if mem_occupy_by_wsg+mem_occupy_by_act_stage<total_mem_size:
                rs2=recompute_strategy.none   
            else:
                rs2=recompute_strategy.all   
        else:
            total_mem_size=tile.sram_capacity #MB
            rs2=recompute_strategy.none
            if mem_occupy_by_wsg+mem_occupy_by_act_stage<total_mem_size:
                ss1=sram_strategy.ACT_weight
                df0=dataflow.IS
            elif mem_occupy_by_act_stage<total_mem_size:
                ss1=sram_strategy.weight
                df0=dataflow.IS
            elif mem_occupy_by_wsg<total_mem_size:
                ss1=sram_strategy.ACT
                df0=dataflow.WS
            else:
                rs2=recompute_strategy.all
                ss1=sram_strategy.cache
                df0=dataflow.OS
        return [df0,ss1,rs2]

    @staticmethod
    def execute_comm_process(tile,env,comm_op:CommOp,wd1:wd):
        if comm_op.type==COMM.ALL_REDUCE:
            pass
        elif comm_op.type==COMM.ALL_REDUCE:
            pass
        elif comm_op.type==COMM.ALL_REDUCE:
            pass
        else:
            pass












        yield env.timeout(5)
    @staticmethod
    def execute_forward_process(tile,env,map_ana:list,device:List[int],op:OpNode,wd1:wd):
        yield env.timeout(5)
    @staticmethod 
    def execute_backward_process(tile,env,map_ana,device:List[int],op:OpNode,wd1:wd):
        yield env.timeout(10)



