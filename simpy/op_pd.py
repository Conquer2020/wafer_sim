import ML
from typing import List,Optional
class CompOp():
    def __init__(self,op_type:ML.OP,op_param:List[int]) -> None:
        self.type=op_type
        self.param_dim=op_param
        self.param_size_m=0
        self.intra_act_size_m=0 #only for complex op like transformer or conv_block @fangjh21.202306602
        self.o_shape=[]
        self.i_shape=[]
        self.fd_macs=0 
        self.__check()
    def __check(self):
        if self.type==ML.OP.Linear:
            assert(len(self.param_dim)==4)#B,M,N,K
            [B,M,N,K]=self.param_dim
            self.param_size_mbyte=(M*K+M)/1000/1000
            self.intra_act_size_m=0
            self.o_shape=[B,M,N] #[B,M,N]
            self.i_shape=[B,N,K] #[B,K,N]
            self.fd_macs=B*M*N*K

        elif self.type==ML.OP.Conv2:
            assert(len(self.param_dim)==7)#B,C,H,W,R,S,K
            [B,C,H,W,R,S,K]=self.param_dim
            self.param_size_mbyte=(R*S*C+1)*K/1000/1000  #without bias @fangjh21.202306602
            self.intra_act_size_m=0
            o_h=H - R  + 1
            o_w=W - S  + 1
            self.o_shape=[B,K,o_h,o_w]
            self.i_shape=[B,C,H,W]
            self.fd_macs=C*R*S*o_h*o_w*K
        elif self.type==ML.OP.Transformer:
            #TODO for verification with hand analysis
            assert(len(self.param_dim)==4)
            [B,S,H,A]=self.param_dim
            self.param_size_mbyte=9*H*H+H*S*B*2
            # @fangjh21.20230602 edited for recompute detail analysis
            self.intra_act_size_m=[1,2,3]#38*B*S*H+7*B*A*S*S  
            self.o_shape=[B,S,H] #[B,M,N]
            self.i_shape=[B,S,H]
            self.fd_macs=24*B*S*H*H+4*B*S*S*H
        elif self.type==ML.OP.Embedding:
            assert(len(self.param_dim)==4)
            self.param_size_mbyte=0
            self.o_shape=0 
            self.i_shape=0 
            self.fd_macs=0
        else:
            self.param_size_mbyte=0
            self.o_shape=0 
            self.i_shape=0 
            self.fd_macs=0
            raise NotImplementedError
    def __str__(self):
        return '({},{})'.format(self.type,self.param_dim)
class CommOp():
    def __init__(self,comm_type:ML.COMM=ML.COMM.NONE,comm_size=0) -> None:
        self.type=comm_type
        self.size=comm_size
        self.__check()
    def __check(self):
        assert(self.type==ML.COMM.NONE or self.type==ML.COMM.ALL_REDUCE or self.type==ML.COMM.ALL_2_ALL)
    def __str__(self) -> str:
        return '({},{})'.format(self.type,self.size)
    def No_comm(self):
        if self.type==ML.COMM.NONE or self.size==0:
            return True
        else:
            return False       
class Oppd(CompOp):
    def __init__(self,op_type:ML.OP,op_param:List[int],hint_name:str) -> None:
        super(Oppd,self).__init__(op_type,op_param)
        self.hint_name=hint_name
        self.parallel_dim=[]
        self.device=[]
        self.comm_full=[]
        self.dpmap_flag=False
    def dpmap(self,device_id:List[int],parallel_dim:Optional[List[int]]=None):
        assert parallel_dim==None or len(parallel_dim)<=len(self.param_dim),'The number of parallel dimensions exceeds the op dim space!'
        if (parallel_dim==None or []) and len(device_id)>1:
            print('Warning:parallel dimension not specified as the number of device  more than one!')
            self.parallel_dim=[0]
            self.device=device_id
        elif len(parallel_dim)>=len(device_id):
            print('Warning:The number of parallel dimensions exceeds the device space!')
            self.parallel_dim=parallel_dim[0:len(device_id)-1]
            self.device=device_id
        else:
            self.parallel_dim=parallel_dim
            self.device=device_id
        # TODO 完成并行通信算子的生成
        self.comm_insert()
        self.dpmap_flag=True
        return True
    def comm_insert(self):
        #pass
        self.comm_full.append(CommOp())#forward
        self.comm_full.append(CommOp(ML.COMM.ALL_REDUCE,128))#backward
        self.comm_full.append(CommOp(ML.COMM.ALL_2_ALL,128))#weight syn
        #for comm in self.comm_full:
        #    print(comm)
    def __str__(self):
        if self.dpmap_flag:
            return '{}:(({},{}),parallel_dim={},device={})'.format(self.hint_name,self.type,self.param_dim,self.parallel_dim,self.device)
        else:
            return '{}:({},{})'.format(self.hint_name,self.type,self.param_dim)
if __name__ == '__main__':
    op1=Oppd(op_type=ML.OP.Linear,op_param=[1,128,128,512],hint_name='s0')
    op1.dpmap(parallel_dim=[0,1],device_id=[0,1])
    print(op1)