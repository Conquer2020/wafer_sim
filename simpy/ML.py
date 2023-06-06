
from  util import BaseEnum as Enum
#定义枚举类
OP = Enum('OP', ('Linear', 'Conv2', 'Embedding', 'Softmax','LayerNorm','Transformer'))
COMM=Enum('COMM',('NONE','ALL_REDUCE','ALL_2_ALL'))
OPTIMIZER=Enum('OPTIMIZER',('NONE','SGD','ADAM'))
BYTES={'NONE':0,'INT8':1,'FP16':2,'TF32':2.375,'FP32':4}

dataflow=Enum('dataflow',('IS','WS','OS'))
comp_model=Enum('comp_model',('simple','SCALE_SIM'))

sram_strategy=Enum('sram_strategy',('cache','weight','ACT','ACT_weight'))
recompute_strategy=Enum('recompute_strategy',('none','half','all'))

pipe_strategy=Enum('pipe_strategy',('GPipe','Megatron1F1B','Interleaved1F1B','Cerebras'))
ZeRO_strategy=Enum('ZeRO_strategy',('none','ZeRO_1','ZeRO_2','ZeRO_3'))

traffic=Enum('traffic',('act_store','act_fetch','comm','act_fd','grad_bd','wt_load','wt_store'))

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
    return NotImplementedError