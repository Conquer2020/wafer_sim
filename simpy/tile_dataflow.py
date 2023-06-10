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
        self.tflops=self.macs*2*self.freq

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
        #acc_op_output_act_size=0          # mulc(op_list[-1].i_shape) no store
        dataflow0=dataflow.WS
        sram1=store_strategy.cache
        recomputes2=recompute_strategy.none
        tiledram3=store_strategy.none
        edgedram4=store_strategy.none
        ZeRO=tile.self.ZeRO
        [pipe_strategy,info1,info2]=stage_info
        input_act_size_m=mulc(op_list[0].i_shape)/1000/1000
        #ouput_act_size=mulc(op_list[-1].o_shape)

        #dram/sram allocation for each op with parallism  and recompute strategy
        # @fangjh21.20230602
        for op in op_list:
            op.set_ZeRO(ZeRO)
            temp=np.array(op.w_s_g_size_m)*np.array(tile.wsg_bytes)
            acc_op_wsg_size+=mulc(temp.tolist())+0 #TODO 计算所需冗余空间
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

        sram_size=tile.sram_capacity #MB
        tiledram_size=tile.dram_capacity*1000 #MB 

        if mem_occupy_by_wsg+mem_occupy_by_act_stage<sram_size:
            dataflow0=dataflow.OS
            sram1=store_strategy.ACT_weight
            recomputes2=recompute_strategy.none
            tiledram3=store_strategy.noneSS
        elif mem_occupy_by_wsg<sram_size: 
            dataflow0=dataflow.WS
            sram1=store_strategy.weight
            if tile.with_dram:
                assert(wd1.dram_per_tile_resource!=[])
                if mem_occupy_by_act_stage<tiledram_size:
                    recomputes2=recompute_strategy.none
                    tiledram3=store_strategy.ACT
                else:
                    dataflow0=dataflow.WS
                    sram1=store_strategy.cache
                    recomputes2=recompute_strategy.all
                    tiledram3=store_strategy.weight
                    edgedram4=store_strategy.ACT
            else:
                    dataflow0=dataflow.WS
                    sram1=store_strategy.cache
                    recomputes2=recompute_strategy.all
                    tiledram3=store_strategy.none
                    edgedram4=store_strategy.ACT_weight
        elif mem_occupy_by_act_stage<sram_size: 
            pass
        else:
            if mem_occupy_by_wsg+mem_occupy_by_act_stage<tiledram_size:
                dataflow0=dataflow.OS
                sram1=store_strategy.cache
                recomputes2=recompute_strategy.none
                tiledram3=store_strategy.ACT_weight
                edgedram4=store_strategy.none

            elif mem_occupy_by_wsg<tiledram_size:
                dataflow0=dataflow.WS
                sram1=store_strategy.cache
                recomputes2=recompute_strategy.all
                tiledram3=store_strategy.weight
                edgedram4=store_strategy.ACT
            else:
                dataflow0=dataflow.IS
                sram1=store_strategy.cache
                recomputes2=recompute_strategy.all
                tiledram3=store_strategy.cache
                edgedram4=store_strategy.ACT_weight
        return  dataflow0,sram1,recomputes2,tiledram3,edgedram4

    @staticmethod
    def execute_comm_process(tile,comm_op:CommOp,wd1:wd,traffic_tpye:traffic=traffic.comm):
        comm_mbytes=0
        if comm_op.type==COMM.ALL_REDUCE:
            for gp in comm_op.device_group:
                comm_mbytes=comm_op.size*tile.comm_bytes
                yield wd1.env.process(wd1.ALL_REDUCE_process(comm_mbytes,gp,traffic_tpye))
        elif comm_op.type==COMM.ALL_2_ALL:
            for gp in comm_op.device_group:
                comm_mbytes=comm_op.size*tile.comm_bytes
                yield wd1.env.process(wd1.ALL_2_ALL_process(comm_mbytes,gp,traffic_tpye))
        elif comm_op.type==COMM.NONE:
            pass
        else:
            pass

    @staticmethod
    def execute_forward_process(tile,env,map_ana:list,device:List[int],op_list:List[OpNode],wd1:wd):
        dataflow0,sram1,recomputes2,tiledram3,edgedram4=map_ana
        for op in op_list:#@fangjh21.20230609
            if sram1==store_strategy.ACT_weight:
                #
                yield env.timeout(2*op.fd_macs_m/tile.tflops)
            elif sram1==store_strategy.ACT:
                pass
            elif sram1==store_strategy.weight:
                pass
            elif sram1==store_strategy.none:
                pass
            else:
                raise NotImplementedError


    @staticmethod 
    def execute_backward_process(tile,env,map_ana,device:List[int],op_list:List[OpNode],wd1:wd):
        yield env.timeout(10)

    @staticmethod 
    def execute_weight_update_process(tile,env,map_ana,device:List[int],op_list:List[OpNode],wd1:wd):
        yield env.timeout(10)




