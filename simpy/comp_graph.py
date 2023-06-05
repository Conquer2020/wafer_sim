from ML import *
import os
import json
from typing import List,Optional
from op_pd import Oppd
from util import *


class OpNode(Oppd):
    def __init__(self, op_type:OP, op_param: List[int], hint_name: str) -> None:
        super().__init__(op_type, op_param, hint_name)
        self.nxt_lt=[]
        self.isTraversed=False
    def __str__(self):
        nxt_lt_id=[it.hint_name for it in self.nxt_lt]
        if self.dpmap_flag:
            return '{}:({},{}),parallel_dim={},device={},child_nodes:{}\n'.\
                format(self.hint_name,self.type,self.param_dim,self.parallel_dim,self.device,nxt_lt_id)
        else:
            return '{}:({},{}),child_nodes:{}\n'.\
                format(self.hint_name,self.type,self.param_dim,nxt_lt_id)
        
    @staticmethod
    def _op2dict(op):
        nxt_lt_id=[it.hint_name for it in op.nxt_lt]
        op_dict={}
        op_dict['type']=str(op.type.name)
        op_dict['param_dim']=str(op.param_dim)
        op_dict['child_nodes']=str(nxt_lt_id)
        if op.dpmap_flag:
            op_dict['parallel_dim']=str(op.parallel_dim)
            op_dict['device']=str(op.device)
        return op_dict
    
    @staticmethod
    def _json2op(json):
        pass   
class CompGraph():
    def __init__(self,root:Optional[OpNode]=None,name='t_Compute_Graph',meta='20230424') -> None:
        self.name=name
        self.meta=meta
        self.root=root
        self.cur=root
        self.op_dict={}
        self.iter=self.next_op(self.root)
    def next_op(self,cur:OpNode):
        #one op node in ML compute graph my have more than one parent node.
        #so when traverse graph,we have to avoid visit one node twice with 'isTraversed' flag
        #However, 'isTraversed' should be clear before next graph traverse
        #TODO 不灵活，遍历后isTraversed flag拉高,下次遍历前需要清除flag
        cur.isTraversed=True
        yield cur
        if cur.nxt_lt!=[]:
            for nxt_op in cur.nxt_lt: 
                if not nxt_op.isTraversed:
                    for op in self.next_op(nxt_op):
                        yield op
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.iter)
    def __str__(self):
        Compute_Graph_str='CompGraph:{}'.format(self.name)
        Compute_Graph_str+=',Root:{}\n'.format(self.root.hint_name)
        return Compute_Graph_str
    @staticmethod
    def _graph2dict(CompGraph):
        graph_dict={}
        graph_dict['graph_name']=CompGraph.name
        graph_dict['root_name']=CompGraph.root.hint_name
        for op in CompGraph:
             graph_dict[op.hint_name]=OpNode._op2dict(op)
        return graph_dict

    def AddEdge(self,son_Op_Node:OpNode,prt_Op_Node:Optional[OpNode]=None):
        if prt_Op_Node==None and self.root==None:
            self.root=son_Op_Node
            self.cur=self.root
            self.iter=self.next_op(self.root)
        elif prt_Op_Node==None:   
            self.cur.nxt_lt.append(son_Op_Node)
        else:
            prt_Op_Node.nxt_lt.append(son_Op_Node)
            self.cur=son_Op_Node
        self.op_dict[son_Op_Node.hint_name]=son_Op_Node
    def AddSubGraph(self,prt_Op_Node:OpNode,SubGraph):
        #TODO
        pass
    def CheckGraph(self):
        #TODO 输出形状与输入形状对齐
        #TODO 计算图的输出节点检查
        pass
    def SplitGraph(self):
        #TODO 
        pass

    @staticmethod
    def gread(path='test',name='gh.json'):
        if os.path.exists(path) is False:
            assert False,'No {} in ./{}/'.format(name,path)
        whole_path_filename = os.path.join(path, name)
        with open(whole_path_filename, mode="r", encoding='utf-8') as f:
            gpdict=json.load(f)
        print(gpdict)
        gp=CompGraph()
        root_index=''
        i=0
        ops_dict={}
        op_next_dict={}
        for items in gpdict:
            if i==0:
                gp.name=gpdict[items]
            elif i==1:
                root_index=gpdict[items] 
            else:
                op_hint_name=items
                op_dict=gpdict[items]
                op_type=None
                op_param=None
                op_next_op=None
                op_plm_dim=None
                op_device=None
                for op_key in op_dict:
                    if op_key=='type':
                        op_type=str2openum(op_dict[op_key])
                    elif op_key=='param_dim':
                        op_param=str2list(op_dict[op_key])
                    elif op_key=='child_nodes':
                        op_next_op=str2strlist(op_dict[op_key])
                    elif op_key=='parallel_dim':
                        op_plm_dim=str2list(op_dict[op_key])
                    elif op_key=='device':
                        op_device=str2list(op_dict[op_key])
                ops_dict[op_hint_name]=OpNode(op_type=op_type,op_param=op_param,hint_name=op_hint_name)
                op_next_dict[op_hint_name]=op_next_op
                if op_plm_dim!=None and op_device !=None:
                    ops_dict[op_hint_name].dpmap(op_device,op_plm_dim)
            i+=1
        gp.AddEdge(ops_dict[root_index])
        for op_father in ops_dict:
            for op_son in op_next_dict[op_father]:
                if op_son!=[] or op_son!=None:
                    gp.AddEdge(ops_dict[op_son],ops_dict[op_father])

        return gp

    @staticmethod
    def gwrite(gp,path='test',name='gh.json'):
        if os.path.exists(path) is False:
            os.mkdir(path)
        whole_path_filename = os.path.join(path, name)
        with open(whole_path_filename, mode="w", encoding='utf-8') as f:
            gpdict=CompGraph._graph2dict(gp)
            #print(gpdict)
            json.dump(gpdict,f,indent=1,separators=(',',':'))

if __name__ == '__main__':

    #define compute graph 
    op1=OpNode(op_type=OP.Linear,op_param=[1,128,128,512],hint_name='s1')
    op2=OpNode(op_type=OP.Linear,op_param=[1,64,64,512],hint_name='s2')
    op3=OpNode(op_type=OP.Linear,op_param=[1,128,64,512],hint_name='s3')
    op4=OpNode(op_type=OP.Linear,op_param=[1,64,64,512],hint_name='s4')
    gp=CompGraph()
    gp.AddEdge(op1)
    gp.AddEdge(op2)
    gp.AddEdge(op3,op2)
    gp.AddEdge(op4,op1)
    gp.AddEdge(op3,op4)


    #mapping by hand
    op1.dpmap(device_id=[0,1,2,3])
    op2.dpmap(device_id=[4,5])
    op3.dpmap(device_id=[6,7,10,11,14,15])
    op4.dpmap(device_id=[12,13])
    CompGraph.gwrite(gp)
    
    gp1=CompGraph.gread(path='test',name='gh.json')
    #with stage info
    CompGraph.gwrite(gp1,path='test',name='gh1.json')

