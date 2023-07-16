from ML import *
import os
import json
from typing import List,Optional
from op_pd import Oppd
from util import *
class OpNode(Oppd):
    def __init__(self, op_type:OP, op_param: List[int], hint_name: str) -> None:
        super().__init__(op_type, op_param, hint_name)
        self.next_nodes=[]
        self.last_nodes=[]
        #self.isTraversed=False
    def __str__(self):
        if self.dpmap_flag:
            return '{}:({},{}),p_sgy={},device={},parent_nodes:{},child_nodes:{}\n'.\
                format(self.hint_name,self.type,self.param_dim,self.p_sgy,self.device,self.last_nodes,self.next_nodes)
        else:
            return '{}:({},{}),parent_nodes:{},child_nodes:{}\n'.\
                format(self.hint_name,self.type,self.param_dim,self.last_nodes,self.next_nodes)
        
    @staticmethod
    def _op2dict(op):
        op_dict={}
        op_dict['type']=str(op.type.name)
        op_dict['param_dim']=str(op.param_dim)
        op_dict['child_nodes']=str(op.next_nodes)
        if op.dpmap_flag:
            op_dict['p_sgy']=str(op.p_sgy)
            op_dict['device']=str(op.device)
        return op_dict
    
    @staticmethod
    def _json2op(json):
        pass   
class CompGraph():
    def __init__(self,root:str=None,name='t_Compute_Graph',meta='20230424') -> None:
        self.name=name
        self.meta=meta
        self.root=root
        self.cur=root
        self.op_dict={}
        self.__iter_index=0
        self.__iter_items=None
    def __iter__(self):
        return self
    def __next__(self):
        if self.__iter_index==0:
            self.__iter_items=iter(self.op_dict.items())
        elif self.__iter_index==len(self.op_dict):
             self.__iter_index=0
             raise StopIteration
        self.__iter_index+=1
        return next(self.__iter_items)
    def __str__(self):
        Compute_Graph_str='CompGraph:{}'.format(self.name)
        Compute_Graph_str+=',Root:{}\n'.format(self.root.hint_name)
        return Compute_Graph_str
    def __len__(self):
        return len(self.op_dict)
    @staticmethod
    def _graph2dict(CompGraph):
        graph_dict={}
        graph_dict['graph_name']=CompGraph.name
        graph_dict['root_name']=CompGraph.root
        for op in CompGraph:
             print(op)
             graph_dict[op.hint_name]=OpNode._op2dict(op)
        return graph_dict

    def AddEdge(self,son_Op_Node:OpNode,prt_Op_Node:Optional[OpNode]=None):
        son_op_name=son_Op_Node.hint_name
        if prt_Op_Node==None and self.root==None:
            self.root=son_op_name
            self.cur=self.root
        elif prt_Op_Node==None:   
            son_Op_Node.last_nodes.append(self.op_dict[self.cur].hint_name)
            self.op_dict[self.cur].next_nodes.append(son_op_name)
        else:
            prt_Op_Node.next_nodes.append(son_op_name)
            son_Op_Node.last_nodes.append(prt_Op_Node.hint_name)
            self.cur=son_op_name
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
        #print(gpdict)
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
                    elif op_key=='p_sgy':
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
    for op in gp:
        print(op)
    for op in gp:
        print(op)
    '''
    CompGraph.gwrite(gp,path='mljson',name='test.json')
    gp1=CompGraph.gread(path='mljson',name='test.json')
    CompGraph.gwrite(gp1,path='mljson',name='test1.json')
    '''

