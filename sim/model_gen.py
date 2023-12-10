from comp_graph import CompGraph,OpNode
from ML import *

def GPT3_Gen(path='model',L=96,B=1564,S=2048,H=12288,A=96):
    ops=[]
    gp=CompGraph(name='GPT3')
    for i in range(L):
        hint='t'+str(i)
        ops.append(OpNode(op_type=OP.Transformer,op_param=[B,S,H,A],hint_name=hint))
        if i==0:
            gp.AddEdge(ops[i])
        else:
            gp.AddEdge(ops[i],ops[i-1])
    CompGraph.gwrite(gp,path=path,name='GPT3')
def Tranformer_Gen(path='model'):
    #L,B,S,H,A=L,B,S,H,A
    Model_size_L=['18B','39B','76B','145B','310B','530B','1T',]
    for Model_size  in Model_size_L:
        if Model_size=='18B':
            L,B,S,H,A=40,1024,2048,6144,48
        elif Model_size=='39B':
            L,B,S,H,A=48,1536,2048,8192,64
        elif Model_size=='76B':
            L,B,S,H,A=60,1792,2048,10240,80
        elif Model_size=='145B':
            L,B,S,H,A=80,2304,2048,12288,96
        elif Model_size=='310B':
            L,B,S,H,A=96,2160,2048,16384,128
        elif Model_size=='530B':
            L,B,S,H,A=105,2520,2048,20480,128
        elif Model_size=='1T':
            L,B,S,H,A=160,3072,2048,25600,160
        else:
            raise NotImplementedError
        ops=[]
        name='T_'+Model_size
        gp=CompGraph(name=name)
        for i in range(L):
            hint='t'+str(i)
            ops.append(OpNode(op_type=OP.Transformer,op_param=[B,S,H,A],hint_name=hint))
            if i==0:
                gp.AddEdge(ops[i])
            else:
                gp.AddEdge(ops[i],ops[i-1])
        CompGraph.gwrite(gp,path=path,name=name)

def BERT_Gen(path='model'):
    #L,B,S,H,A=L,B,S,H,A
    
    [V,L,B,S,H,A]=[30522,24,512,512,1024,16]
    ops=[]
    gp=CompGraph(name='BERT_LARGE')
    emb=OpNode(op_type=OP.Embedding,op_param=[B,S,H,V,2,S],hint_name='emb_3')
    gp.AddEdge(emb)
    for i in range(L):
        hint='L'+str(i)
        ops.append(OpNode(op_type=OP.Transformer,op_param=[B,S,H,A],hint_name=hint))
        #print(i)
        if i==0:
            gp.AddEdge(ops[i],emb)
        else:
            gp.AddEdge(ops[i],ops[i-1])
    pooler=OpNode(op_type=OP.Linear,op_param=[B,H,S,H],hint_name='pooler')
    #pred_linear=OpNode(op_type=OP.Linear,op_param=[B,V,S,H],hint_name='pred_linear')
    gp.AddEdge(pooler,ops[-1])
    CompGraph.gwrite(gp,path=path,name='BERT_LARGE')
    
    ops=[]
    [V,L,B,S,H,A]=[30522,12,512,128,768,12]
    gp=CompGraph(name='BERT_BASE')
    emb=OpNode(op_type=OP.Embedding,op_param=[B,S,H,V,2,S],hint_name='emb_3')
    gp.AddEdge(emb)
    for i in range(L):
        hint='L'+str(i)
        ops.append(OpNode(op_type=OP.Transformer,op_param=[B,S,H,A],hint_name=hint))
        #print(i)
        if i==0:
            gp.AddEdge(ops[i],emb)
        else:
            gp.AddEdge(ops[i],ops[i-1])
    pooler=OpNode(op_type=OP.Linear,op_param=[B,H,S,H],hint_name='pooler')
    #pred_linear=OpNode(op_type=OP.Linear,op_param=[B,V,S,H],hint_name='pred_linear')
    gp.AddEdge(pooler,ops[-1])
    CompGraph.gwrite(gp,path=path,name='BERT_BASE')

def ResNet50_Gen(path='model',B=64,H=224,W=224,C=3):
    def BTNK1_Gen(gp,op_start,hint,B,C,W,C1,S):#S=stride
        #op_param=[B,C,H,W,R,S,K]
        ops=[]
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,W,W,1,S,C1],hint_name=hint+'_left0'))
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,W,W,3,1,C1],hint_name=hint+'_left1'))
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,W,W,1,1,C1*4],hint_name=hint+'_left2'))
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,W,W,1,S,C1*4],hint_name=hint+'_right0'))
        gp.AddEdge(ops[0],op_start)
        gp.AddEdge(ops[3],op_start)
        gp.AddEdge(ops[1],ops[0])
        gp.AddEdge(ops[2],ops[1])
        return ops[2]#,ops[3] 
    def BTNK2_Gen(gp,op_start,hint,B,C,W):
        ops=[]
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,W,W,1,1,C//4],hint_name=hint+'_left0'))
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,W,W,3,1,C//4],hint_name=hint+'_left1'))
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,W,W,1,1,C],hint_name=hint+'_left2'))
        gp.AddEdge(ops[0],op_start)
        gp.AddEdge(ops[1],ops[0])
        gp.AddEdge(ops[2],ops[1])
        return ops[2]#,op_start
    def STAGE0_Gen(gp,B=64,H=224,W=224,C=3):
        hint='STAGE0_'
        ops=[]
        ops.append(OpNode(op_type=OP.Conv2,op_param=[B,C,H,W,7,2,64],hint_name=hint+'0'))
        ops.append(OpNode(op_type=OP.Pool,op_param=[B,C,H,W,3,2],hint_name=hint+'1'))
        gp.AddEdge(ops[0])
        gp.AddEdge(ops[1])
        return ops[1]
    def STAGE1_Gen(gp,op_start,B=64,H=56,W=56,C=256):
        hint='STAGE1_'
        op0=BTNK1_Gen(gp,op_start,hint+'BTNK1',B,64,56,64,1)
        op1=BTNK2_Gen(gp,op0,hint+'BTNK2_0',B,256,56)
        op2=BTNK2_Gen(gp,op1,hint+'BTNK2_1',B,256,56)
        return op2
    def STAGE2_Gen(gp,op_start,B=64,H=56,W=56,C=256):
        hint='STAGE2_'
        op0=BTNK1_Gen(gp,op_start,hint+'BTNK1',B,256,56,128,2)
        op1=BTNK2_Gen(gp,op0,hint+'BTNK2_0',B,512,28)
        op2=BTNK2_Gen(gp,op1,hint+'BTNK2_1',B,512,28)
        op3=BTNK2_Gen(gp,op2,hint+'BTNK2_2',B,512,28)
        return op3
    def STAGE3_Gen(gp,op_start,B=64,H=28,W=28,C=512):
        hint='STAGE3_'
        op0=BTNK1_Gen(gp,op_start,hint+'BTNK1',B,256,56,128,2)
        op1=BTNK2_Gen(gp,op0,hint+'BTNK2_0',B,512,28)
        op2=BTNK2_Gen(gp,op1,hint+'BTNK2_1',B,512,28)
        op3=BTNK2_Gen(gp,op2,hint+'BTNK2_2',B,512,28)
        op4=BTNK2_Gen(gp,op3,hint+'BTNK2_3',B,512,28)
        op5=BTNK2_Gen(gp,op4,hint+'BTNK2_4',B,512,28)
        return op5
    
    def STAGE4_Gen(gp,op_start,B=64,H=14,W=14,C=1024):
        hint='STAGE4_'
        op0=BTNK1_Gen(gp,op_start,hint+'BTNK1',B,1024,14,512,2)
        op1=BTNK2_Gen(gp,op0,hint+'BTNK2_0',B,2048,7)
        op2=BTNK2_Gen(gp,op1,hint+'BTNK2_1',B,2048,7)
        return op2
    gp=CompGraph(name='ResNet50')
    op0=STAGE0_Gen(gp,B=64,H=224,W=224,C=3)
    op1=STAGE1_Gen(gp,op0,B=64,H=56,W=56,C=256)
    op2=STAGE2_Gen(gp,op1,B=64,H=56,W=56,C=256)
    op3=STAGE3_Gen(gp,op2,B=64,H=28,W=28,C=512)
    op4=STAGE4_Gen(gp,op3,B=64,H=14,W=14,C=1024)
    CompGraph.gwrite(gp,path='model',name='ResNet50')
if __name__ == '__main__':
    BERT_Gen()
    GPT3_Gen()
    ResNet50_Gen()
    Tranformer_Gen()