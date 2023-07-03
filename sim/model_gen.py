from comp_graph import CompGraph,OpNode
from ML import *

#here is a sample Tranformer model for test
#
def Transformer_Gen(L=96,B=1564,S=2048,H=12288,A=96):
    #L,B,S,H,A=L,B,S,H,A
    ops=[]
    gp=CompGraph()
    for i in range(L):
        hint='t'+str(i)
        ops.append(OpNode(op_type=OP.Transformer,op_param=[B,S,H,A],hint_name=hint))
        #print(i)
        if i==0:
            gp.AddEdge(ops[i])
        else:
            gp.AddEdge(ops[i],ops[i-1])
    CompGraph.gwrite(gp,path='mljson',name='gpt-3.json')