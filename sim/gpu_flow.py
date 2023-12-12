import math
import simpy
import numpy as np
from typing import List,Union
from util import *
from wafer_device import Wafer_Device as wd
from ML import *
from comp_graph import CompGraph,OpNode
from op_pd import CommOp

class GPU():# for compute process
    def __init__(self,env,GPU_name='A100',
                FLOPS=[1,2,3,4],#INT8,FP16,FP32,FP64
                dram_bw_GB=12288/16/8,
                dram_capacity_GB=6/16,
                opt=OPTIMIZER.ADAM,
                ZeRO=ZeRO_strategy.ZeRO_2,
                Analytical=True
                ) -> None:
        #info
        self.GPU_name=GPU_name
        self.dram_bw=dram_bw_GB
        self.FLOPS=FLOPS
        self.tile_dram_capacity_m=dram_capacity_GB*1000 #Mb
        self.ZeRO=ZeRO
        self.opt=opt
        # define store byte
        self.act_bytes=0
        self.wsg_store_bytes=[0,0,0]
        self.buffer_bytes=0
        self.comm_bytes=0
        self.__set_bytes()

        #simpy env
        self.env=env
        self.Analytical=Analytical
        if not self.Analytical:
            self.cp_worker= simpy.Resource(env, capacity=1)
            self.cm_worker= simpy.Resource(env, capacity=1)

        #mapping op
        self.device_id=[]
        self.op_list=[]
        self.noc=None


    def __set_bytes(self):
        #TODO Mixed-precision is popular in  ML training process.
        #However,many AI archs have their float numberbprecision like TF32(Nvdia),CFP16(Dojo),etc.
        #@fangjh21.20230606
        #print('Mixed-precision')
        self.cp_bytes=BYTES['FP16']
        self.act_bytes=BYTES['FP16'] if self.opt!=OPTIMIZER.NONE else BYTES['NONE']
        w=BYTES['FP16']+BYTES['FP32']
        g=BYTES['FP16']+BYTES['FP32']
        s=BYTES['NONE'] if self.opt!=OPTIMIZER.ADAM else 2*BYTES['FP32']
        self.wsg_store_bytes=[w,s,g]
        self.buffer_bytes=BYTES['FP16']
        self.comm_bytes=BYTES['FP16']
        self.full_bytes=BYTES['FP32']

    def GPU_comp_process(self,macs_m:Union[float,List[int]]):
        exetime = 2 * macs_m / self.FLOPS / 1e3  # us to ms
        if not self.Analytical:
            with self.cp_worker.request() as req:
                    yield req
                    yield self.env.timeout(exetime)
        else:
            yield self.env.timeout(exetime)            
    def GPU_comm_process(self,comm_op:CommOp,wd1:wd,traffic_tpye:event=event.comm,overlap=False):
        '''
        this is the GPU communication process
        '''
        comm_mbytes=comm_op.size*self.comm_bytes
        #here if communication can not overlap by compute time,the 'cm_worker' should to be 'cp_worker'
        if not self.Analytical:
            with (self.cm_worker.request() if overlap else self.cp_worker.request()) as req:
                    yield req       
                    for gp in comm_op.device_group:
                        if comm_op.type==COMM.ALL_REDUCE:
                            yield wd1.env.process(wd1.ALL_2_ALL_process(comm_mbytes,gp,traffic_tpye))
                        elif comm_op.type==COMM.ALL_2_ALL:
                            yield wd1.env.process(wd1.ALL_REDUCE_process(comm_mbytes,gp,traffic_tpye))
                        elif comm_op.type==COMM.NONE:
                            pass
                        else:
                            pass
        else:
            for gp in comm_op.device_group:
                if comm_op.type==COMM.ALL_REDUCE:
                    yield wd1.env.process(wd1.ALL_2_ALL_process(comm_mbytes,gp,traffic_tpye))
                elif comm_op.type==COMM.ALL_2_ALL:
                    yield wd1.env.process(wd1.ALL_REDUCE_process(comm_mbytes,gp,traffic_tpye))
                elif comm_op.type==COMM.NONE:
                    pass
                else:
                    pass
    def mapping_analysis(self,stage_info,device:List[int],op_list:List[OpNode],wd1:wd,train:bool):
        #init 
        #device_gp=device
        assert(self.with_dram==wd1.with_dram_per_tile)
        self.device_id=device
        self.op_list=op_list
        self.noc=wd1

    
    def forward_process(self):
        def analysis_template_event(param=[None,None,None,None,None,None,None,None,None,None,None]):
            assert(len(param)==11)
            #param=[wt_load,act_fetch,wt_load,act_fetch,Zero_comm,comp,intra_act_store,out_act_store,intra_act_store,out_act_store]
            event_list=[]
            comm_event=None
            if(param[0]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[0]*self.buffer_bytes*len(self.device_id),group_id=self.device_id,task_id=event.wt_load,multicast=False))
            if(param[1]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[1]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.act_fetch,multicast=False))
            if(param[2]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[2]*self.buffer_bytes,self.device_id,event.wt_load,WRITE=False))
            if(param[3]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[3]*self.act_bytes,self.device_id,event.act_fetch,WRITE=False))
            if(param[4]!=None):
                #ZeRO communication 
                event_list.append(self.GPU_comm_process(param[4],self.noc,event.comm))
            if(param[5]!=None):
                #compute 
                event_list.append(self.GPU_comp_process(param[5]))
            if(param[6]!=None):
                #communication 
                comm_event=self.GPU_comm_process(param[6],self.noc,event.comm)
            if(param[7]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[7]*self.act_bytes,self.device_id,event.act_store,WRITE=True))
            if(param[8]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[8]*self.act_bytes,self.device_id,event.act_store,WRITE=True))
            if(param[9]!=None):
                event_list.append(self.noc.dram_write_group_process(access_size_MB=param[9]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.act_store,gather=True))
            if(param[10]!=None):
                event_list.append(self.noc.dram_write_group_process(access_size_MB=param[10]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.act_store,gather=True))
            return event_list,comm_event
        dataflow0,sram1,recomputes2,GPUdram3,edgedram4=self.map_ana
        for op in self.op_list:#@fangjh21.20230609
            access_size_m=0
            #print(op.type)
            #print(op.ZeRO_comm_d[0])
            #print(op.f_b_u_comm_d[0])
            param=[None,None,None,None,op.ZeRO_comm_d[0],op.fd_macs_m,op.f_b_u_comm_d[0] ,None,None,None,None]
            #param[0],param[1] edge dram read
            #param[2],param[3] GPU dram read
            #param[4]=op.ZeRO_comm_d[0] 
            #param[5]=op.fd_macs_m
            #param[6]=op.f_b_u_comm_d[0] 
            #param[7],param[8] GPU dram write
            #param[9],param[10] edge dram write
            if sram1==store_strategy.ACT_weight:
                #ideal situation
                pass
            elif sram1==store_strategy.ACT:
                if GPUdram3==store_strategy.weight:
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            param[2]=op.w_s_g_access_m[0]                           
                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            param[2]=op.w_s_g_access_m[0]*max(1,self.buffer_bytes*temp_input_size_m/self.sram_capacity_m)      
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            param[2]=op.w_s_g_access_m[0]                            
                        elif dataflow0==dataflow.IS:
                            param[2]=op.w_s_g_access_m[0]*max(1,self.act_bytes*op.intra_act_access_m/self.sram_capacity_m)                           
                        else:
                            raise NotImplementedError
                else:
                    raise NotImplementedError  
            elif sram1==store_strategy.weight:
                if GPUdram3==store_strategy.ACT:
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            param[3]=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[7]=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            param[3]=temp_input_size_m
                            param[7]=temp_input_size_m
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            param[3]=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[7]=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        elif dataflow0==dataflow.IS:
                            param[3]=op.intra_act_access_m
                            param[7]=op.intra_act_access_m
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        else:
                            raise NotImplementedError
                else:
                    raise NotImplementedError
            elif sram1==store_strategy.cache:
                if GPUdram3==store_strategy.ACT_weight:
                    assert(edgedram4==store_strategy.none)
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            param[2]=op.w_s_g_access_m[0]
                            param[3]=temp_input_size_m*max(1,op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[7]=temp_input_size_m*max(1,op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            param[2]=op.w_s_g_access_m[0]*max(1,self.buffer_bytes*temp_input_size_m/self.sram_capacity_m)
                            param[3]=temp_input_size_m
                            param[7]=temp_input_size_m
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        elif dataflow0==dataflow.OS:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            param[2]=op.w_s_g_access_m[0]
                            param[3]=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[7]=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        elif dataflow0==dataflow.IS:
                            param[2]=op.w_s_g_access_m[0]*max(1,self.act_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            param[3]=op.intra_act_access_m
                            param[7]=op.intra_act_access_m
                            param[8]=mulc(op.o_shape)/1000/1000 #TODO
                        elif dataflow0==dataflow.OS:
                            raise NotImplementedError
                elif GPUdram3==store_strategy.ACT:
                    assert(edgedram4==store_strategy.weight and dataflow0==dataflow.WS)
                    raise NotImplementedError
                elif GPUdram3==store_strategy.weight:
                    assert(edgedram4==store_strategy.ACT and dataflow0==dataflow.WS)
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            access_size_m=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO    
                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            access_size_m=temp_input_size_m
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO 
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO 

                        elif dataflow0==dataflow.IS:
                            access_size_m=op.intra_act_access_m
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO 
                        else:
                            raise NotImplementedError           
                elif GPUdram3==store_strategy.cache:
                    assert(edgedram4==store_strategy.ACT_weight)
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            access_size_m=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            param[0]=op.w_s_g_access_m[0]
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO 
                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            access_size_m=temp_input_size_m
                            param[0]=op.w_s_g_access_m[0]*max(1,self.act_bytes*temp_input_size_m/self.tile_dram_capacity_m)
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO 
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)  
                            param[0]=op.w_s_g_access_m[0]
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO 
                        elif dataflow0==dataflow.IS:
                            access_size_m=op.intra_act_access_m
                            param[0]=op.w_s_g_access_m[0]*max(1,self.act_bytes*temp_input_size_m/self.tile_dram_capacity_m)
                            param[1]=access_size_m
                            param[9]=access_size_m
                            param[10]=mulc(op.o_shape)/1000/1000 #TODO 
                        else:
                            raise NotImplementedError
            else:
                raise NotImplementedError
            #please notice that the bytes number is considered in execute_template_event function
            events,comm_event=analysis_template_event(param)
            execute_event=[self.env.process(event) for event in events]
            #print("forward_process start @ {:.3f} ms".format(self.env.now))
            yield simpy.AllOf(self.env,execute_event)
            yield self.env.process(comm_event)
            #print("forward_process end @ {:.3f} ms".format(self.env.now))

    def backward_process(self):
        def analysis_template_recompute_event(param=[None,None,None,None,None,None,None,None,None]):
            assert(len(param)==9)
            #param=[wt_load,act_fetch,wt_load,act_fetch,Zero_comm,comp,intra_act_store,out_act_store,intra_act_store,out_act_store]
            event_list=[]
            comm_event=None
            if(param[0]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[0]*self.buffer_bytes*len(self.device_id),group_id=self.device_id,task_id=event.wt_load,multicast=False))
            if(param[1]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[1]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.act_fetch,multicast=False))
            if(param[2]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[2]*self.buffer_bytes,self.device_id,event.wt_load,WRITE=False))
            if(param[3]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[3]*self.act_bytes,self.device_id,event.act_fetch,WRITE=False))
            if(param[4]!=None):
                #ZeRO communication 
                event_list.append(self.GPU_comm_process(param[4],self.noc,event.comm))
            if(param[5]!=None):
                #compute 
                event_list.append(self.GPU_comp_process(param[5]))
            if(param[6]!=None):
                #communication 
                comm_event=self.GPU_comm_process(param[6],self.noc,event.comm)
            if(param[7]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[7]*self.act_bytes*len(self.device_id),self.device_id,event.act_store,WRITE=True))
            if(param[8]!=None):
                event_list.append(self.noc.dram_write_group_process(access_size_MB=param[8]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.act_store,gather=True))
            if event_list==[]:
                event_list=[None]
            return event_list,comm_event
        def analysis_template_dloss_event(param=[None,None,None,None,None,None,None,None,None]):  
            assert(len(param)==9)
            event_list=[]
            comm_event=None
            if(param[0]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[0]*self.buffer_bytes*len(self.device_id),group_id=self.device_id,task_id=event.wt_load,multicast=False))
            if(param[1]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[1]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.grad_fetch,multicast=False))
            if(param[2]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[2]*self.buffer_bytes,self.device_id,event.wt_load,WRITE=False))
            if(param[3]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[3]*self.act_bytes,self.device_id,event.grad_fetch,WRITE=False))

            if(param[4]!=None):
                #ZeRO communication 
                event_list.append(self.GPU_comm_process(param[4],self.noc,event.comm))
            if(param[5]!=None):
                #compute 
                event_list.append(self.GPU_comp_process(param[5]))
            if(param[6]!=None):
                #communication 
                comm_event=self.GPU_comm_process(param[6],self.noc,event.comm)
            if(param[7]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[7]*self.act_bytes,self.device_id,event.grad_store,WRITE=True))
            if(param[8]!=None):
                event_list.append(self.noc.dram_write_group_process(access_size_MB=param[8]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.grad_store,gather=True))
            if event_list==[]:
                event_list=[None]
            return event_list,comm_event
        def analysis_template_dW_event(param=[None,None,None,None,None,None,None,None,None,None,None,None,None]):  
            assert(len(param)==13)
            event_list=[]
            comm_event=None
            if(param[0]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[0]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.act_fetch,multicast=False))
            if(param[1]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[1]*self.act_bytes*len(self.device_id),group_id=self.device_id,task_id=event.grad_fetch,multicast=False))
            if(param[2]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[2]*self.full_bytes*len(self.device_id),group_id=self.device_id,task_id=event.opt_load,multicast=False))
            if(param[3]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[3]*self.act_bytes,self.device_id,event.act_fetch,WRITE=False))
            if(param[4]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[4]*self.act_bytes,self.device_id,event.grad_fetch,WRITE=False))
            if(param[5]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[5]*self.full_bytes,self.device_id,event.opt_load,WRITE=False))
            if(param[6]!=None):
                #ZeRO communication 
                event_list.append(self.GPU_comm_process(param[6],self.noc,event.comm))
            if(param[7]!=None):
                #compute 
                event_list.append(self.GPU_comp_process(param[7]))
            if(param[8]!=None):
                #communication 
                comm_event=self.GPU_comm_process(param[8],self.noc,event.comm)
            if(param[9]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[9]*self.full_bytes,self.device_id,event.opt_store,WRITE=True))
            if(param[10]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[10]*self.full_bytes,self.device_id,event.wt_store,WRITE=True))
            if(param[11]!=None):
                event_list.append(self.noc.dram_write_group_process(access_size_MB=param[11]*self.full_bytes*len(self.device_id),group_id=self.device_id,task_id=event.opt_store,gather=True))
            if(param[12]!=None):
                event_list.append(self.noc.dram_write_group_process(access_size_MB=param[12]*self.full_bytes*len(self.device_id),group_id=self.device_id,task_id=event.wt_store,gather=True))
            if event_list==[]:
                event_list=[None]
            return event_list,comm_event 
        dataflow0,sram1,recomputes2,GPUdram3,edgedram4=self.map_ana

        for op in self.op_list:
            #9,9,13
            re_param    =[None,None,None,None,op.ZeRO_comm_d[0],op.fd_macs_m,op.f_b_u_comm_d[0] ,None,None]
            dloss_param =[None,None,None,None,op.ZeRO_comm_d[1],op.fd_macs_m,op.f_b_u_comm_d[1],None,None]
            dW_param    =[None,None,None,None,None,None,None,op.fd_macs_m,None,None,None,None,None]
            if sram1==store_strategy.ACT_weight:
                #ideal situation
                pass
            elif sram1==store_strategy.ACT:
                if GPUdram3==store_strategy.weight:
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            re_param[2]=op.w_s_g_access_m[0]   
                            dloss_param[2]=op.w_s_g_access_m[0] 
                            dW_param[5]=op.w_s_g_access_m[1]
                        elif dataflow0==dataflow.IS:   
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            total_size_m=op.w_s_g_access_m[0]*max(1,self.act_bytes*temp_input_size_m/self.sram_capacity_m) 
                            re_param[2]=total_size_m
                            total_size_m=op.w_s_g_access_m[0]*max(1,self.act_bytes*op.intra_act_access_m/self.sram_capacity_m) 
                            dloss_param[2]=total_size_m
                            dW_param[5]=op.w_s_g_access_m[1]
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            re_param[2]=op.w_s_g_access_m[0] 
                            dloss_param[2]=op.w_s_g_access_m[0]  
                            dW_param[5]=op.w_s_g_access_m[1]
                        elif dataflow0==dataflow.IS:
                            re_param[2]=op.w_s_g_access_m[0]*max(1,self.act_bytes*op.intra_act_access_m/self.sram_capacity_m) 
                            dloss_param[2]=op.w_s_g_access_m[0]*max(1,self.act_bytes*op.intra_act_access_m/self.sram_capacity_m) 
                            dW_param[5]=op.w_s_g_access_m[1]
                        else:
                            raise NotImplementedError
                else:
                    raise NotImplementedError  
            elif sram1==store_strategy.weight:
                if GPUdram3==store_strategy.ACT:
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            re_param[3]=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            re_param[7]=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            dloss_param[3]=access_size_m
                            dloss_param[7]=access_size_m
                            dW_param[3]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            dW_param[4]=op.intra_act_access_m
                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            re_param[3]=temp_input_size_m
                            re_param[7]=temp_input_size_m
                            dloss_param[3]=op.intra_act_access_m
                            dloss_param[7]=op.intra_act_access_m
                            dW_param[3]=op.intra_act_access_m
                            dW_param[4]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            temp_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            re_param[3]=temp_size_m
                            re_param[7]=temp_size_m
                            dloss_param[3]=temp_size_m
                            dloss_param[7]=temp_size_m
                            dW_param[3]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            dW_param[4]=op.intra_act_access_m
                        elif dataflow0==dataflow.IS:
                            re_param[3]=op.intra_act_access_m
                            re_param[7]=op.intra_act_access_m
                            dloss_param[3]=op.intra_act_access_m
                            dloss_param[7]=op.intra_act_access_m
                            dW_param[3]=op.intra_act_access_m
                            dW_param[4]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                        else:
                            raise NotImplementedError
                else:
                    raise NotImplementedError
            elif sram1==store_strategy.cache:
                if GPUdram3==store_strategy.ACT_weight:
                    assert(edgedram4==store_strategy.none)
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            re_param[2]=op.w_s_g_access_m[0]
                            re_param[3]=temp_input_size_m*max(1,self.act_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            re_param[7]=temp_input_size_m*max(1,self.act_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            dloss_param[2]=op.w_s_g_access_m[0]
                            dloss_param[3]=op.intra_act_access_m*max(1,self.act_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            dloss_param[7]=op.intra_act_access_m*max(1,self.act_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            dW_param[3]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            dW_param[4]=op.intra_act_access_m
                            dW_param[5]=op.w_s_g_access_m[1]
                            dW_param[9]=op.w_s_g_access_m[1]
                            dW_param[10]=op.w_s_g_access_m[2]

                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            re_param[2]=op.w_s_g_access_m[0]*max(1,self.buffer_bytes*temp_input_size_m/self.sram_capacity_m)
                            re_param[3]=temp_input_size_m
                            re_param[7]=temp_input_size_m
                            dloss_param[2]=op.w_s_g_access_m[0]*max(1,self.buffer_bytes*temp_input_size_m/self.sram_capacity_m)
                            dloss_param[3]=op.intra_act_access_m
                            dloss_param[7]=op.intra_act_access_m
                            dW_param[3]=op.intra_act_access_m
                            dW_param[4]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            dW_param[5]=op.w_s_g_access_m[1]
                            dW_param[9]=op.w_s_g_access_m[1]
                            dW_param[10]=op.w_s_g_access_m[2]
                        elif dataflow0==dataflow.OS:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            re_param[2]=op.w_s_g_access_m[0]
                            temp_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            re_param[3]=temp_size_m
                            re_param[7]=temp_size_m
                            re_param[2]=op.w_s_g_access_m[0]
                            temp_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            re_param[3]=temp_size_m
                            re_param[7]=temp_size_m
                            dW_param[3]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            dW_param[4]=op.intra_act_access_m
                            dW_param[5]=op.w_s_g_access_m[1]
                            dW_param[9]=op.w_s_g_access_m[1]
                            dW_param[10]=op.w_s_g_access_m[2]
  
                        elif dataflow0==dataflow.IS:
                            re_param[2]=op.w_s_g_access_m[0]*max(1,self.act_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            re_param[3]=op.intra_act_access_m
                            re_param[7]=op.intra_act_access_m
                            dloss_param[2]=op.w_s_g_access_m[0]
                            temp_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.sram_capacity_m)
                            dloss_param[3]=temp_size_m
                            dloss_param[7]=temp_size_m
                            dW_param[3]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.sram_capacity_m)
                            dW_param[4]=op.intra_act_access_m
                            dW_param[5]=op.w_s_g_access_m[1]
                            dW_param[9]=op.w_s_g_access_m[1]
                            dW_param[10]=op.w_s_g_access_m[2]

                        elif dataflow0==dataflow.OS:
                            raise NotImplementedError
                elif GPUdram3==store_strategy.ACT:
                    assert(edgedram4==store_strategy.weight and dataflow0==dataflow.WS)
                    raise NotImplementedError
                elif GPUdram3==store_strategy.weight:
                    assert(edgedram4==store_strategy.ACT and dataflow0==dataflow.WS)
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            access_size_m=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            re_param[1]=access_size_m
                            re_param[8]=access_size_m  
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            dloss_param[1]=access_size_m
                            dloss_param[8]=access_size_m
                            dW_param[0]=access_size_m
                            dW_param[1]=op.intra_act_access_m

                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            re_param[1]=temp_input_size_m
                            re_param[8]=temp_input_size_m
                            dloss_param[1]=op.intra_act_access_m
                            dloss_param[8]=op.intra_act_access_m
                            dW_param[0]=op.intra_act_access_m
                            dW_param[1]=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            re_param[1]=access_size_m
                            re_param[8]=access_size_m
                            dloss_param[1]=access_size_m
                            dloss_param[8]=access_size_m
                            dW_param[0]=access_size_m
                            dW_param[1]=op.intra_act_access_m
                        elif dataflow0==dataflow.IS:
                            access_size_m=op.intra_act_access_m
                            re_param[1]=access_size_m
                            re_param[8]=access_size_m
                            dloss_param[1]=access_size_m
                            dloss_param[8]=access_size_m
                            dW_param[0]=op.intra_act_access_m
                            dW_param[1]=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                        
                        else:
                            raise NotImplementedError           
                elif GPUdram3==store_strategy.cache:
                    assert(edgedram4==store_strategy.ACT_weight)
                    if recomputes2==recompute_strategy.all:
                        if dataflow0==dataflow.WS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            access_size_m=temp_input_size_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            re_param[0]=op.w_s_g_access_m[0]
                            re_param[1]=access_size_m
                            re_param[8]=access_size_m 
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            dloss_param[0]=op.w_s_g_access_m[0]
                            dloss_param[1]=access_size_m
                            dloss_param[8]=access_size_m 
                            dW_param[0]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.tile_dram_capacity_m)
                            dW_param[1]=op.intra_act_access_m
                            dW_param[2]=op.w_s_g_access_m[1]
                            dW_param[11]=op.w_s_g_access_m[1]
                            dW_param[12]=op.w_s_g_access_m[2]
                        elif dataflow0==dataflow.IS:
                            temp_input_size_m=mulc(op.i_shape)/1000/1000
                            access_size_m=temp_input_size_m
                            re_param[0]=op.w_s_g_access_m[0]*max(1,self.act_bytes*temp_input_size_m/self.tile_dram_capacity_m)
                            re_param[1]=access_size_m
                            re_param[8]=access_size_m
                            access_size_m=op.intra_act_access_m
                            dloss_param[0]=op.w_s_g_access_m[0]*max(1,self.act_bytes*temp_input_size_m/self.tile_dram_capacity_m)
                            dloss_param[1]=access_size_m
                            dloss_param[8]=access_size_m 
                            dW_param[0]=op.intra_act_access_m*max(1,self.act_bytes*op.intra_act_access_m/self.tile_dram_capacity_m)
                            dW_param[1]=op.intra_act_access_m
                            dW_param[2]=op.w_s_g_access_m[1]
                            dW_param[11]=op.w_s_g_access_m[1]
                            dW_param[12]=op.w_s_g_access_m[2]
                        else:
                            raise NotImplementedError
                    else:#without recompute strategy
                        if dataflow0==dataflow.WS:
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            re_param[0]=op.w_s_g_access_m[0]
                            re_param[1]=access_size_m
                            re_param[8]=access_size_m
                            access_size_m=op.intra_act_access_m*max(1,self.buffer_bytes*op.w_s_g_access_m[0]/self.tile_dram_capacity_m)
                            dloss_param[0]=op.w_s_g_access_m[0]
                            dloss_param[1]=access_size_m
                            dloss_param[8]=access_size_m 
                            dW_param[0]=op.intra_act_access_m*max(1,self.buffer_bytes*op.intra_act_access_m/self.tile_dram_capacity_m)
                            dW_param[1]=op.intra_act_access_m
                            dW_param[2]=op.w_s_g_access_m[1]
                            dW_param[11]=op.w_s_g_access_m[1]
                            dW_param[12]=op.w_s_g_access_m[2]
                        elif dataflow0==dataflow.IS:
                            access_size_m=op.intra_act_access_m
                            re_param[0]=op.w_s_g_access_m[0]*max(1,self.act_bytes*temp_input_size_m/self.tile_dram_capacity_m)
                            re_param[1]=access_size_m
                            re_param[8]=access_size_m
                            access_size_m=op.intra_act_access_m
                            dloss_param[0]=op.w_s_g_access_m[0]*max(1,self.act_bytes*temp_input_size_m/self.tile_dram_capacity_m)
                            dloss_param[1]=access_size_m
                            dloss_param[8]=access_size_m 
                            dW_param[0]=op.intra_act_access_m*max(1,self.act_bytes*op.intra_act_access_m/self.tile_dram_capacity_m)
                            dW_param[1]=op.intra_act_access_m
                            dW_param[2]=op.w_s_g_access_m[1]
                            dW_param[11]=op.w_s_g_access_m[1]
                            dW_param[12]=op.w_s_g_access_m[2]
                        else:
                            raise NotImplementedError
            else:
                raise NotImplementedError
            
            if recomputes2==recompute_strategy.all:
                events,comm_event=analysis_template_recompute_event(re_param)
                execute_event=[self.env.process(event) for event in events]
                yield simpy.AllOf(self.env,execute_event)
                yield self.env.process(comm_event)
            else:
                pass
            events,comm_event=analysis_template_dloss_event(dloss_param)
            execute_event=[self.env.process(event) for event in events]
            yield simpy.AllOf(self.env,execute_event)
            yield self.env.process(comm_event)
            events,comm_event=analysis_template_dW_event(dW_param)
            execute_event=[self.env.process(event) for event in events]
            yield simpy.AllOf(self.env,execute_event)
            #if comm_event!=None:
            #    yield self.env.process(comm_event)
    def update_process(self):
        def analysis_template_event(param=[None,None,None,None,None,None,None]):
            assert(len(param)==7)
            event_list=[]
            if(param[0]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[0]*self.full_bytes*len(self.device_id),group_id=self.device_id,task_id=event.wt_load,multicast=False))
            if(param[1]!=None):
                event_list.append(self.noc.dram_read_group_process(access_size_MB=param[1]*self.full_bytes*len(self.device_id),group_id=self.device_id,task_id=event.grad_fetch,multicast=False))
            if(param[2]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[2]*self.full_bytes,self.device_id,event.wt_load,WRITE=False))
            if(param[3]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[3]*self.full_bytes,self.device_id,event.grad_fetch,WRITE=False))
            if(param[4]!=None):
                #DP communication 
                event_list.append(self.GPU_comm_process(param[4],self.noc,event.comm,True))
            if(param[5]!=None):
                event_list.append(self.noc.tile_dram_group_access_process(param[5]*self.full_bytes,self.device_id,event.wt_load,WRITE=True))
            if(param[6]!=None):
                event_list.append(self.noc.dram_write_group_process(access_size_MB=param[6]*self.full_bytes*len(self.device_id),group_id=self.device_id,task_id=event.wt_load,gather=True))
            return event_list
        dataflow0,sram1,recomputes2,GPUdram3,edgedram4=self.map_ana
        for op in self.op_list:
            update_param=[None,None,None,None,op.f_b_u_comm_d[2],None,None]
            if sram1==store_strategy.ACT_weight:
                #ideal situation
                pass
            elif sram1==store_strategy.ACT:
                if GPUdram3==store_strategy.weight:
                    update_param[2]=1
                    update_param[3]=1
                else:
                    raise NotImplementedError  
            elif sram1==store_strategy.weight:
                if GPUdram3==store_strategy.ACT:
                    pass
                else:
                    raise NotImplementedError
            elif sram1==store_strategy.cache:
                if GPUdram3==store_strategy.ACT_weight:
                    assert(edgedram4==store_strategy.none)
                    update_param[2]=op.w_s_g_access_m[0]
                    update_param[3]=op.w_s_g_access_m[1]
                    update_param[5]=op.w_s_g_access_m[0]
                elif GPUdram3==store_strategy.ACT:
                    assert(edgedram4==store_strategy.weight and dataflow0==dataflow.WS)
                    update_param[0]=op.w_s_g_access_m[0]
                    update_param[1]=op.w_s_g_access_m[1]
                    update_param[6]=op.w_s_g_access_m[0]
                elif GPUdram3==store_strategy.weight:
                    assert(edgedram4==store_strategy.ACT and dataflow0==dataflow.WS)
                    update_param[2]=op.w_s_g_access_m[0]
                    update_param[3]=op.w_s_g_access_m[1]
                    update_param[5]=op.w_s_g_access_m[0]
                elif GPUdram3==store_strategy.cache:
                    assert(edgedram4==store_strategy.ACT_weight)
                    update_param[0]=op.w_s_g_access_m[0]
                    update_param[1]=op.w_s_g_access_m[1]
                    update_param[6]=op.w_s_g_access_m[0]
            else:
                raise NotImplementedError
            events=analysis_template_event(update_param)
            execute_event=[self.env.process(event) for event in events]
            yield simpy.AllOf(self.env,execute_event)
    

