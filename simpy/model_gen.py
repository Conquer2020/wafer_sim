from comp_graph import CompGraph,OpNode
from ML import *

#here is a sample Tranformer model for test
batch_size=4
L,B,S,H,A=96,3200000,2048,12288,96
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