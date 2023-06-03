import math
import simpy
from  util import BaseEnum as Enum
from typing import List,Optional
from util import *
from wafer_noc_mesh import Wafer_Noc_Mesh as NoC
from wafer_noc_mesh import Packet


import ml
import graph
DATAFLOW=Enum('DATAFLOW',('IS','WS','OS'))
COMP_MODEL=Enum('COMP_MODEL',('SIMPLE','SCALE_SIM'))

class Core():# for compute process
    def __init__(self,env:simpy.Environment,core_name='tx8',
                 sram_size_MB=3,macs=4000,freq_GHz=1,SR_SC_T=[64,128,16]) -> None:
        #info
        self.core_name=core_name

        # buffer & sram size definition
        self.store_byte=ml.BYTES['FP16']
        self.sram=sram_size_MB
        self.ifmap=0
        self.weight=0
        self.ofmap=0
        self.max_dim=1024


        # compute shape definition
        self.macs=macs
        self.cp_byte=ml.BYTES['FP16']
        self.array_group=[2,2]
        self.array_shape=[]
        self.kernel=SR_SC_T
        self.meta_cycles=0
        self.cp_model=COMP_MODEL.SCALE_SIM

        self.freq_GHz=freq_GHz
        self.work_dataflow=DATAFLOW.IS
        self.update_info()

        #simpy env 
        self.env=env
        self.ifmap_buffer=[]
        self.weight_buffer=[]
        self.ofmap_buffer=[]
        self.weight_sram=[]
        self.act_sram=[]

    def compute_model(self,SR,SC,T,R,C,PR,PC):
        '''
        define matrix multiply [m,n,k]: (m,k)*(k,n)=(m,n)
        SR:m SC:n T: k
        PE array num:PR*RC
        each PE macs units: R*C
        #reference: https://github.com/ARM-software/SCALE-Sim
        '''
        if self.cp_model==COMP_MODEL.SCALE_SIM:
            sr=math.ceil(SR/PR)
            sc=math.ceil(SC/PC)
            return (2*R+C+T-2)*math.ceil(sr/R)*math.ceil(sc/C)
        elif self.cp_model==COMP_MODEL.SIMPLE:
            return T*math.ceil(SR/(R*PR))*math.ceil(SC/(C*SC))
        else :
            raise NotImplementedError
         
    def allocate_buffer_size(self,SR,SC,T):
        #o=w*i+b
        if self.work_dataflow==DATAFLOW.IS:
            self.ifmap=SC*T/1000/1000*self.store_byte
            self.weight=SR*T/1000/1000*self.store_byte
            self.ofmap=self.max_dim*SC/1000/1000*self.store_byte# means output dim M<=max
        elif self.work_dataflow==DATAFLOW.OS:
            self.ifmap=SC*self.max_dim/1000/1000*self.store_byte# means input dim K<=max
            self.weight=SR*T/1000/1000*self.store_byte
            self.ofmap=SR*SC/1000/1000*self.store_byte
        elif self.work_dataflow==DATAFLOW.WS:
            self.ifmap=T*self.max_dim/1000/1000*self.store_byte# means input dim N<=max
            self.weight=SR*T/1000/1000*self.store_byte
            self.ofmap=SR*self.max_dim/1000/1000*self.store_byte# means input dim N<=max
        else:
            raise NotImplementedError
        total_buffer_size_mb=self.ifmap+self.weight+self.ofmap
        print('total_buffer_size:{:.3f} M Byte'.format(total_buffer_size_mb))
        return total_buffer_size_mb

    def update_info(self):
        #suppose meta shape [SR,SC,T]
        [SR,SC,T]=self.kernel
        self.allocate_buffer_size(SR,SC,T)
        #compute core micro arch R,C,PR,PC
        self.array_shape=shape_suppose(int(self.macs/(self.array_group[0]*self.array_group[1])))
        self.meta_cycles=self.compute_model(SR=SR,SC=SC,T=T,\
            R=self.array_shape[0],C=self.array_shape[1],PR=self.array_group[0],PC=self.array_group[1])
        
    def allocate_sram_check(self,op_list:List[ml.Op]):
        self.ifmap_buffer=simpy.Store(self.env,capacity=1)
        self.weight_buffer=simpy.Store(self.env,capacity=1)
        self.ofmap_buffer=simpy.Store(self.env,capacity=1)
        [SR,SC,T]=self.kernel
        num_op=len(op_list)
        if num_op>1:
            raise NotImplementedError
        else:
            op=op_list[0]
            if op.type==ml.OP.Linear: 
                # mxk kxn -> mxn
                [B,M,N,K]=op.para 
                op.print() 
                self.weight_sram=M*K*self.store_byte
                self.act_sram=[]
            else:
                raise NotImplementedError
            
        if self.weight_sram>self.sram:
            self.weight_sram=self.sram
            return False
        else:
            return True

    def core_cp_process_env(self,meta_cycles,DEBUG_MODE=True):
        def cp_process(DEBUG_MODE=True):
            if DEBUG_MODE:
                print('cp_process got input data @{:.3f} us ...'.format(self.env.now))
                print(self.ifmap_buffer.items)
                print(self.weight_buffer.items)
            if_data,weight_data=yield self.env.all_of([self.ifmap_buffer.get(), self.weight_buffer.get()])
            if DEBUG_MODE:
                if_data=if_data.value
                weight_data=weight_data.value
            #assert(if_data<=self.ifmap and weight_data<=self.weight)
            yield self.env.timeout(meta_cycles/self.freq_GHz/1000) #us
            of_data=Packet(id=i,shape=[self.kernel[0],self.kernel[1]])
            yield self.ofmap_buffer.put(of_data)
        while True:
            yield self.env.process(cp_process())
            if DEBUG_MODE:
                print('cp_process finish @ {:.3f} us'.format(self.env.now))
            break

    def dataflow_process(self,op_list:List[ml.Op],NoC:NoC,v_last_q:simpy.Store,v_next_q:simpy.Store,core_id_group=[1,2,3,4],DEBUG_MODE=True):
        (SR,SC,T)=self.kernel
        for index, op in enumerate(op_list):
            if op.op_type==ml.OP.Linear: 
                # mxk kxn -> mxn
                [B,M,N,K]=op.op_param 
                op.print() 
                if self.work_dataflow==DATAFLOW.WS:
                    pass
                elif self.work_dataflow==DATAFLOW.OS:
                    pass 
                elif self.work_dataflow==DATAFLOW.IS:
                    for i0 in range(B):
                        for i1  in range(math.ceil(N/SC)):
                            for i2  in range(math.ceil(K/T)):  
                                input_data=v_last_q.get().value
                                yield self.ifmap_buffer.put(input_data)
                                print('test-2')
                                for i3  in range(math.ceil(M/SR)):
                                    print(math.ceil(M/SR))
                                    print('test-1')
                                    weight_data=yield self.sram_weight[index].get()
                                    print('test-0')
                                    yield self.weight_buffer.put(weight_data)
                                    print('test-0.5')
                                    #print(if_stationary)
                                    yield self.env.process(self.core_cp_process_env(self.meta_cycles))
                                    print('test0')
                                    self.ifmap_buffer.put(input_data)
                                    #self.weight_buffer.put(weight_data)
                                    yield self.sram_weight[index].put(weight_data)
                                    print('test1')
                                    yield self.ofmap_buffer.get()
                                self.ifmap_buffer.get()
                            if op.comm_type==ml.COMM.ALL_REDUCE:
                                yield self.env.process(NoC.ALL_REDUCE_process(self.env,N/SC*M/SR,core_id_group))
                            elif op.comm_type==ml.COMM.ALL_2_ALL:
                                yield self.env.process(NoC.ALL_2_ALL_process(self.env,N/SC*M/SR,core_id_group))
                            output= self.ofmap_buffer.get().value
                            print('test2')
                            yield v_next_q.get(output)
                            print('test3')
                else:
                    NotImplementedError
            elif op.op_type==ml.OP.Embedding:
                [N,M,N]=op.op_param
            elif op.op_type==ml.OP.Conv2:
                # stride=1
                # padding=0
                [N,C,H,W,K,R,S]=op.op_param
            else:
                raise NotImplementedError
        #
if __name__ == '__main__':
    env=simpy.Environment()
    tx8=Core(env)
    noc=NoC(tile_inter_shape=[2,2],tile_intra_shape=[2,2])
    noc.create_noc_link(env)
    op1=graph.Op_Node(op_type=ml.OP.Linear,op_param=[1,128,128,512])
    op2=graph.Op_Node(op_type=ml.OP.Linear,op_param=[1,512,128,128],comm_type=ml.COMM.ALL_REDUCE)
    op_list=[op1,op2]
    tx8.allocate_sram(op_list)
    last_Q=simpy.Store(env,1)
    next_Q=simpy.Store(env,1)
    t=math.ceil(128*512/128/16)
    for i in range(t):
        pk=Packet(id=i,shape=[128,16])
        env.process(noc.dram_read_process(env,128*16/1000/1000,noc,src_id=1,task_id=i,DEBUG_MODE=False))
        last_Q.put(pk)
        #print(last_Q.get())
    env.process(tx8.dataflow_process(op_list,NoC,last_Q,next_Q,core_id_group=[1,2,3,4]))

    env.run(until=10000000) 







    


