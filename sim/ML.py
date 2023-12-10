
from enum import Enum

class Enum(Enum):
    def __str__(self):
        return self.name
    def __repr__(self):
        return f'{self.name}'
OP = Enum('OP', ('Linear', 'Conv2', 'Embedding', 'Softmax','LayerNorm','Transformer','Pool','Concat','Sum'))
COMM=Enum('COMM',('NONE','ALL_REDUCE','ALL_2_ALL'))
OPTIMIZER=Enum('OPTIMIZER',('NONE','SGD','ADAM'))
BYTES={'NONE':0,'INT8':1,'FP16':2,'TF32':2.375,'FP32':4,'FP64':5}
ML_STATE=Enum('ML_STATE',('FORWARD','BACKWARD','PARAM_SYNC'))
dataflow=Enum('dataflow',('IS','WS','OS'))
comp_model=Enum('comp_model',('simple','SCALE_SIM','abrupt_curve'))

store_strategy=Enum('store_strategy',('cache','weight','ACT','ACT_weight','none'))
recompute_strategy=Enum('recompute_strategy',('none','half','all'))

pipe_strategy=Enum('pipe_strategy',('GPipe','Megatron1F1B','Interleaved1F1B','Cerebras'))
ZeRO_strategy=Enum('ZeRO_strategy',('none','ZeRO_1','ZeRO_2','ZeRO_3'))

event=Enum('event',('act_store','act_fetch','comm','act_fd','grad_fetch','grad_store','wt_load','wt_store','opt_load','opt_store','dloss_load','dloss_store'))

def str2openum(op_str):
    if op_str=='Linear':
        return OP.Linear
    elif op_str=='Conv2':
        return OP.Conv2
    elif op_str=='Embedding':
        return OP.Embedding
    elif op_str=='Softmax':
        return OP.Softmax
    elif op_str=='LayerNorm':
        return OP.LayerNorm
    elif op_str=='Transformer':
        return OP.Transformer
    elif op_str=='Pool':
        return OP.Pool
    return NotImplementedError