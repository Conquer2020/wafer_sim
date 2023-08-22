from ML import *
from util import *
from typing import List,Optional,Union
import numpy as np
import math
class CompOp():
    def __init__(self,op_type:OP,op_param:List[int],p_sgy:Union[List[int],None]) -> None:
        #base info 
        self.type=op_type
        self.param_dim=op_param
        self.p_sgy=p_sgy
        self.ZeRO=ZeRO_strategy.none

        self.o_shape=[]
        self.i_shape=[]

        self.unit_m=1000*1000
        #for complex op like transformer @fangjh21.202306602
        #influenced by parallism strategy
        #capacity req
        self.intra_act_size_m=0 
        self.w_s_g_size_m=[]#[0,0,0]
        #bandwidth req
        self.intra_act_access_m=0
        self.w_s_g_access_m=[]#[0,0,0]
        #compute power req
        self.fd_macs_m=0  
        #communication req
        self.f_b_u_comm=[]#[0,0,0]
        self.ZeRO_comm=[]#[0,0]
        #self._analysis()
    def __str__(self):
        return '({},{})'.format(self.type,self.param_dim)
    def _analysis(self):
        if self.type==OP.Linear:

            assert(len(self.param_dim)==4 and ( len(self.p_sgy)==4))#B,M,N,K
            [B,M,N,K]=self.param_dim
            [Nd,Nm_M,Nm_N,Nm_K]=self.p_sgy

            self.o_shape=[B//Nd,M//Nm_M,N//Nm_N] #[B,M,N]
            self.i_shape=[B//Nd,N//Nm_N,K//Nm_K] #[B,K,N]  
            #capacity req
            self.intra_act_size_m=0 
            self.w_s_g_size_m=[(M*K+M)/Nm_M/Nm_N/Nm_K/self.unit_m,2*(M*K+M)/Nm_M/Nm_N/Nm_K/self.unit_m,(M*K+M)/Nm_M/Nm_N/Nm_K/self.unit_m]
            #bandwidth req
            self.intra_act_access_m=0
            self.w_s_g_access_m=self.w_s_g_size_m
            #compute power req
            #TODO 
            assert(self.ZeRO==ZeRO_strategy.none)
            self.fd_macs_m=B*M*N*K/Nd/Nm_M/Nm_N/Nm_K/self.unit_m
            #no sharding
            self.f_b_u_comm=[Nm_K*B*M*N/Nd/Nm_M/Nm_N/self.unit_m,Nm_M*B*K*N/Nd/Nm_K/Nm_N/self.unit_m,Nd*M*K/Nm_M/Nm_K/self.unit_m]
            self.ZeRO_comm=[0,0]


        elif self.type==OP.Conv2:
            assert(len(self.param_dim)==7 and ( len(self.p_sgy)==5))#B,C,H,W,R,S,K, 
            [B,C,H,W,R,S,K]=self.param_dim
            [Nd,Nm_C,Nm_H,Nm_W,Nm_K]=self.p_sgy
            o_h=H//Nm_H - R  + 1   
            o_w=W//Nm_W - S  + 1
            self.o_shape=[B//Nd,K//Nm_K,o_h,o_w]
            self.i_shape=[B//Nd,C//Nm_C,H//Nm_H,W//Nm_W] 

            #capacity req
            self.intra_act_size_m=0 
            self.w_s_g_size_m=[(R*S*C/Nm_C+1)*K/Nm_K/self.unit_m,2*(R*S*C/Nm_C+1)*K/Nm_K/self.unit_m,(R*S*C/Nm_C+1)*K/Nm_K/self.unit_m]
            #bandwidth req
            self.intra_act_access_m=0
            self.w_s_g_access_m=self.w_s_g_size_m
            #compute power req
            #TODO 
            assert(self.ZeRO==ZeRO_strategy.none)
            self.fd_macs_m=C//Nm_C*R*S*o_h*o_w*K//Nm_K/self.unit_m
            self.f_b_u_comm=[Nm_C*B*K*o_h*o_w/Nd/Nm_K/self.unit_m,Nm_K*B*C*H*W/Nd/Nm_C/Nm_H/Nm_W/self.unit_m,Nd*R*S*C*K/Nm_C/Nm_K/self.unit_m]
            self.ZeRO_comm=[0,0]

        elif self.type==OP.Embedding:
            #TODO
            assert(len(self.param_dim)==4 and ( len(self.p_sgy)==2))
            #dlrm emb
            #emb_size_list=[39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]
            [batch_size,input_dim,emb_dim,emb_size_list]=self.param_dim
            [Nd,Nm_emb_dim]=self.p_sgy 
            emb_size_total=sum(emb_size_list)#/Nm_emb_size
            emb_num=len(emb_size_list)
            self.o_shape=[batch_size//Nd,input_dim,emb_dim//Nm_emb_dim]
            self.i_shape=[batch_size//Nd,input_dim] 

            #capacity req
            self.intra_act_size_m=0 
            w_size_m=emb_dim/Nm_emb_dim*emb_size_total/self.unit_m
            max_grad_size_m=(batch_size/Nd*emb_dim/Nm_emb_dim)*emb_num/self.unit_m
            self.w_s_g_size_m=[w_size_m,2*max_grad_size_m,max_grad_size_m]
            #bandwidth req
            self.intra_act_access_m=0
            self.w_s_g_access_m=self.w_s_g_size_m
            #compute power req
            assert(self.ZeRO==ZeRO_strategy.none)
            self.fd_macs_m=0#batch_size*input_dim*math.log(emb_size_total)/self.unit_m
            #TODO
            #all2all,all2all,all2all(allreduce's extreme case),
            self.f_b_u_comm=[batch_size*input_dim*emb_dim/Nd/Nm_emb_dim,batch_size*input_dim*emb_dim/Nd/Nm_emb_dim,batch_size*emb_dim/Nd]
            self.ZeRO_comm=[0,0]

        elif self.type==OP.Transformer:
            #TODO for verification with hand analysis
            assert(len(self.param_dim)==4 and ( len(self.p_sgy)==2))
            [B,S,H,A]=self.param_dim
            [Nd,Nm]=self.p_sgy
            self.o_shape=[B//Nd,S,H]
            self.i_shape=[B//Nd,S,H]  
            w_s_g=np.array([12*H*H/Nm,3*12*H*H/Nm,12*H*H/Nm])/self.unit_m
            zero_w_s_g=np.array([1,1,1])
            w_s_g_access=np.array([12*H*H/Nm,3*12*H*H/Nm,12*H*H/Nm])/self.unit_m
            zero_w_s_g_access=np.array([1,1,1])
            self.ZeRO_comm=[0,0]
            #reference:Wang huizheng's
            if self.ZeRO==ZeRO_strategy.ZeRO_3:
                zero_w_s_g_access=np.array([1/Nd,1/Nd,1/Nd])
                zero_w_s_g=np.array([1/Nd,1/Nd,1/Nd])
                self.ZeRO_comm=[2*12*H*H/Nm/self.unit_m,2*12*H*H/Nm/self.unit_m] #TODO
            elif self.ZeRO==ZeRO_strategy.ZeRO_2:
                zero_w_s_g_access=np.array([1,1/Nd,1/Nd])
                zero_w_s_g=np.array([1,1/Nd,1/Nd])
                self.ZeRO_comm=[2*12*H*H/Nm/self.unit_m,2*12*H*H/Nm/self.unit_m] #TODO
            elif self.ZeRO==ZeRO_strategy.ZeRO_1:
                zero_w_s_g_access=np.array([1,1/Nd,1])
                zero_w_s_g=np.array([1,1/Nd,1])
                self.ZeRO_comm=[2*12*H*H/Nm/self.unit_m,2*12*H*H/Nm/self.unit_m] #TODO
            elif self.ZeRO==ZeRO_strategy.none:
                zero_w_s_g_access=np.array([1,1,1])
                zero_w_s_g=np.array([1,1,1])
                self.ZeRO_comm=[0,0]
            self.w_s_g_size_m=(w_s_g*zero_w_s_g).tolist()#capacity req
            self.w_s_g_access_m=(w_s_g_access*zero_w_s_g_access).tolist()#bandwidth req
            self.f_b_u_comm=[2*12*B*S*H/Nd/self.unit_m,2*12*B*S*H/Nd/self.unit_m,12*H*H/Nm/self.unit_m]
            self.intra_act_size_m=B*S*((15*H+2.5*A*S)/Nm+2*H)/Nd/self.unit_m
            self.intra_act_access_m=((34*B*S*H+7*B*A*S*S)/Nm+4*B*S*H)/Nd/self.unit_m#bandwidth req
            self.fd_macs_m=(24*B*S*H*H+4*B*S*S*H)/Nd/Nm/self.unit_m#compute power req
        else:
            #TODO
            self.o_shape=0 
            self.i_shape=0 
            self.fd_macs_m=0
            raise NotImplementedError
    def set_ZeRO(self,ZeRO):
        self.ZeRO=ZeRO
        self._analysis()

class CommOp():
    def __init__(self,device_group:Optional[List[int]]=None,comm_type:COMM=COMM.NONE,comm_size=0) -> None:
        self.type=comm_type
        self.size=comm_size
        self.device_group=device_group
        self._analysis()
    def _analysis(self):
        assert(self.type==COMM.NONE or self.type==COMM.ALL_REDUCE or self.type==COMM.ALL_2_ALL)
    def __str__(self) -> str:
        return '({},{})'.format(self.type,self.size)
    def No_comm(self):
        if self.type==COMM.NONE or self.size==0:
            return True
        else:
            return False       
class Oppd(CompOp):
    def __init__(self,op_type:OP,op_param:List[int],hint_name:str) -> None:
        super(Oppd,self).__init__(op_type,op_param,None)
        self.hint_name=hint_name
        self.device=[]
        #parallism_dim:forward(f):(comm_type,comm_size_MB),backward(b):updata_weight(u):
        #here is a fact that communication caused by data parallelism only happens on weight update phase,
        #similarly,communication caused by model parallelism only happens on forward and backward phase
        self.f_b_u_comm_d=[]
        self.ZeRO_comm_d=[] #forward all-gather,backward all-gather
        self.dpmap_flag=False
    def _comm_set(self):
        #pass
        self.f_b_u_comm_d=[]
        self.ZeRO_comm_d=[]
        if self.type==OP.Linear:
            #[Nd,Nm_M,Nm_N,Nm_K]=self.p_sgy
            [Nd_Group,Nm_M_Group,Nm_N_Group,Nm_K_Group]=split_comm_group(self.device,self.p_sgy)
            comm_info=[]
            comm_info.append(CommOp(Nm_K_Group,COMM.ALL_REDUCE,self.f_b_u_comm[0]))#forward
            comm_info.append(CommOp(Nm_M_Group,COMM.ALL_REDUCE,self.f_b_u_comm[1]))#backward
            comm_info.append(CommOp(Nd_Group,COMM.ALL_REDUCE,self.f_b_u_comm[2]))#weight update
        elif self.type==OP.Conv2:
            #[Nd,Nm_C,Nm_H,Nm_W,Nm_K]=self.p_sgy
            [Nd_Group,Nm_C_Group,_,_,Nm_K_Group]=split_comm_group(self.device,self.p_sgy)
            comm_info=[]
            comm_info.append(CommOp(Nm_C_Group,COMM.ALL_REDUCE,self.f_b_u_comm[0]))#forward
            comm_info.append(CommOp(Nm_K_Group,COMM.ALL_REDUCE,self.f_b_u_comm[1]))#backward
            comm_info.append(CommOp(Nd_Group,COMM.ALL_REDUCE,self.f_b_u_comm[2]))#weight update
        elif self.type==OP.Transformer:
            [Nd_Group,Nm_Group]=split_comm_group(self.device,self.p_sgy[0:2])
            comm_info=[]
            comm_info.append(CommOp(Nm_Group,COMM.ALL_REDUCE,self.f_b_u_comm[0]))#forward
            comm_info.append(CommOp(Nm_Group,COMM.ALL_REDUCE,self.f_b_u_comm[1]))#backward
            comm_info.append(CommOp(Nd_Group,COMM.ALL_REDUCE,self.f_b_u_comm[2]))#weight update
            self.f_b_u_comm_d=comm_info  
            self.ZeRO_comm_d.append(CommOp(Nd_Group,COMM.ALL_2_ALL,self.ZeRO_comm[0]))
            self.ZeRO_comm_d.append(CommOp(Nd_Group,COMM.ALL_2_ALL,self.ZeRO_comm[1]))
        elif self.type==OP.Embedding:
            #[Nd,Nm_emb_dim]=self.p_sgy 
            [Nd_Group,Nm_emb_dim_Group]=split_comm_group(self.device,self.p_sgy[0:2])
            comm_info=[]
            comm_info.append(CommOp(Nm_emb_dim_Group,COMM.ALL_2_ALL,self.f_b_u_comm[0]))#forward
            comm_info.append(CommOp(Nm_emb_dim_Group,COMM.ALL_2_ALL,self.f_b_u_comm[1]))#backward
            comm_info.append(CommOp(Nd_Group,COMM.ALL_2_ALL,self.f_b_u_comm[2]))#weight update
        else:
            raise NotImplementedError
    def update(self):
        self._analysis()
        self._comm_set()
    def set_ZeRO(self,ZeRO):
        super().set_ZeRO(ZeRO)
        self._comm_set()

    def dpmap(self,device_id:List[int],p_sgy:Union[List[int],None]=None):
        if (p_sgy==None or p_sgy==[]):
            if self.type==OP.Linear:
                self.p_sgy=[1,len(device_id),1,1]
            elif self.type==OP.Conv2:
                self.p_sgy=[1,len(device_id),1,1,1]
            elif self.type==OP.Transformer:
                self.p_sgy=[1,len(device_id)]
            self.device=device_id
        else:
            assert(mulc(p_sgy)==len(device_id))
            self.p_sgy=p_sgy
            self.device=device_id
        # TODO 完成并行通信算子的生成
        self.update()
        self.dpmap_flag=True
        return True
    def __str__(self):
        if self.dpmap_flag:
            return '{}:(({},{}),p_sgy={},device={})'.format(self.hint_name,self.type,self.param_dim,self.p_sgy,self.device)
        else:
            return '{}:({},{})'.format(self.hint_name,self.type,self.param_dim)
if __name__ == '__main__':
    op1=Oppd(op_type=OP.Transformer,op_param=[1,128,128,512],hint_name='s0')
    op1.dpmap(p_sgy=[1,2],device_id=[0,1])
    print(op1)