
from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import pipeline_copy as pipe
from ML import *
import simpy
import math
from util import *
def mapping_GPT3(env:simpy.Environment,model:CompGraph,tile_config:dict,wd:Wafer_Device):
    tiles_id=wd.device() 
    STG_NUM=32
    DATA_PARALLELISM=2
    tiles=[]
    for i in range(STG_NUM):
        tiles.append(tiles_id[i::STG_NUM])
    #print(tiles)
    Layers_num=len(model)
    nums_per_stg=math.ceil(Layers_num/STG_NUM)
    j=0
    ops=[]
    ops_per_stg=[]
    for i,op_name in enumerate(model.op_dict):
        d_size=len(tiles[j])
        dp=DATA_PARALLELISM
        mp=d_size//dp
        assert mp*dp==d_size,'make sure that mp*dp=d_size'
        op=model.op_dict[op_name]
        op.dpmap(device_id=tiles[j],p_sgy=[dp,mp])
        ops.append(op)
        if i % nums_per_stg==nums_per_stg-1:
            j+=1
            ops_per_stg.append(ops)
            ops=[]
    if ops!=[]:
        ops_per_stg[-1].append(op)
    #write graph with device to file
    CompGraph.gwrite(model,path='model',name=model.name+'_map')
    stgs=[]
    for i in range(STG_NUM):
        last_core_id=None if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=None if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,tile_config,ops_per_stg[i],last_core_id,cur_core_id,next_core_id,noc=wd))
    return stgs
def mapping_BERT_LARGE(env:simpy.Environment,model:CompGraph,tile_config:dict,wd:Wafer_Device):
    STG_NUM=26
    DATA_PARALLELISM=1
    tiles=[]
    tiles=[[] for _ in range(STG_NUM)]
    '''
    for i in range(10):
        temp[0]+=[0+32*i,1+32*i,16+32*i,17+32*i]
        temp[1]+=[2+32*i]
        temp[2]+=[18+32*i]
        temp[3]+=[19+32*i]
        temp[4]+=[3+32*i]
        temp[5]+=[4+32*i]
        temp[6]+=[20+32*i]
        temp[7]+=[21+32*i]
        temp[8]+=[5+32*i]
        temp[9]+=[6+32*i]
        temp[10]+=[22+32*i]
        temp[11]+=[23+32*i]
        temp[12]+=[7+32*i]
        temp[13]+=[8+32*i]
        temp[14]+=[24+32*i]
        temp[15]+=[25+32*i]
        temp[16]+=[9+32*i]
        temp[17]+=[10+32*i]
        temp[18]+=[26+32*i]
        temp[19]+=[27+32*i]
        temp[20]+=[11+32*i]
        temp[21]+=[12+32*i]
        temp[22]+=[28+32*i]
        temp[23]+=[29+32*i]
        temp[24]+=[13+32*i]
        temp[25]+=[14+32*i,15+32*i,30+32*i,31+32*i]
    
    #temp[0]=[0]
    for i in range(26):
        tiles.append(temp[i])
    '''
    #nums_per_stg=math.ceil(Layers_num/STG_NUM)
    #tiles[0]=wd.pos_trans([0,0],[0,3])
    tiles[0]=wd.pos_trans([0,0])
    tiles[1]=wd.pos_trans([1,0],[3,3])
    tiles[2]=wd.pos_trans([4,0],[6,3])
    tiles[3]=wd.pos_trans([7,0],[9,3])
    tiles[4]=wd.pos_trans([10,0],[12,3])
    tiles[5]=wd.pos_trans([13,0],[15,3])
    tiles[6]=wd.pos_trans([16,0],[19,2])
    tiles[7]=wd.pos_trans([16,3],[19,5])
    tiles[8]=wd.pos_trans([16,6],[19,8])
    tiles[9]=wd.pos_trans([16,9],[19,11])
    tiles[10]=wd.pos_trans([17,12],[19,15])
    tiles[11]=wd.pos_trans([14,12],[16,15])
    tiles[12]=wd.pos_trans([13,8],[15,11])
    tiles[13]=wd.pos_trans([13,4],[15,7])
    tiles[14]=wd.pos_trans([10,4],[12,7])
    tiles[15]=wd.pos_trans([7,4],[9,7])
    tiles[16]=wd.pos_trans([4,4],[6,7])
    tiles[17]=wd.pos_trans([0,4],[3,6])
    tiles[18]=wd.pos_trans([0,7],[3,9])
    tiles[19]=wd.pos_trans([0,10],[3,12])
    tiles[20]=wd.pos_trans([0,13],[3,15])
    tiles[21]=wd.pos_trans([4,12],[6,15])
    tiles[22]=wd.pos_trans([4,8],[6,11])
    tiles[23]=wd.pos_trans([7,8],[9,11])
    tiles[24]=wd.pos_trans([10,8],[12,11])
    tiles[25]=wd.pos_trans([7,12],[13,15])
    draw_mapping(wd,model.name,tiles)
    ops_per_stg=[]
    for i,op_name in enumerate(model.op_dict):
        #print(op_name)
        d_size=len(tiles[i])
        dp=DATA_PARALLELISM
        mp=d_size//dp
        #assert mp*dp==d_size,'make sure that mp({})*dp({})=d_size({})'.format(mp,dp,d_size)
        op=model.op_dict[op_name]
        if op.type==OP.Transformer:
            #print([dp,mp,d_size])
            op.dpmap(device_id=tiles[i],p_sgy=[dp,mp])
        elif op.type==OP.Embedding:
            op.dpmap(device_id=tiles[i],p_sgy=[1,d_size])
            #op.dpmap(device_id=tiles[i],p_sgy=[dp,mp])
        elif op.type==OP.Linear:
            op.dpmap(device_id=tiles[i],p_sgy=[dp,mp,1,1])
        ops_per_stg.append([op])
    #write graph with device to file
    CompGraph.gwrite(model,path='model',name=model.name+'_map')
    draw_mapping(wd,model.name,tiles)
    stgs=[]
    for i in range(STG_NUM):
        last_core_id=None if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=None if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,tile_config,ops_per_stg[i],last_core_id,cur_core_id,next_core_id,noc=wd))
    return stgs


def mapping_ResNet50(env:simpy.Environment,model:CompGraph,tile_config:dict,wd:Wafer_Device):
    STG_NUM=17
    DATA_PARALLELISM=1
    #tiles=[]
    #Layers_num=len(model)
    stgs=[]
    #total_macs_m=0
    #total_tile=4*4*4*5
    tiles=[[] for _ in range(STG_NUM)]
    split=[2,6,18,18,18,18,18,18,46,18,18,18,18,18,18,18,32,12,12]
    #print(len(split))
    #tiles_id=wd.device() 
    tiles[0]=wd.pos_trans([0,0,0,0],[0,0,1,0])
    #print(tiles[0])
    #tiles[1]=wd.pos_trans([0,0,0,1],[0,1,1,0])
    tiles[1]=wd.pos_trans([0,1],[1,3])
    #print(tiles[1])
    tiles[2]=wd.pos_trans([2,0],[3,3])
    tiles[2]+=wd.pos_trans([0,4],[3,5])
    tiles[2]+=(wd.pos_trans([0,6],[1,6]))
    tiles[3]=wd.pos_trans([2,6],[3,6])
    tiles[3]+=(wd.pos_trans([0,7],[3,11]))
    tiles[4]=(wd.pos_trans([0,12],[3,15]))
    tiles[4]+=(wd.pos_trans([4,14],[4,15]))
    tiles[5]=(wd.pos_trans([5,11],[7,15]))
    tiles[5]+=(wd.pos_trans([4,11],[4,13]))  
    tiles[6]=(wd.pos_trans([4,6],[5,10]))
    tiles[6]+=(wd.pos_trans([6,7],[7,10]))  
    tiles[7]=(wd.pos_trans([4,3],[5,5]))
    tiles[7]+=(wd.pos_trans([6,3],[8,6])) 
    tiles[8]=(wd.pos_trans([4,0],[12,2]))
    tiles[8]+=(wd.pos_trans([9,3],[12,6]))
    tiles[8]+=(wd.pos_trans([12,7]))
    #print(len(tiles[8])) 
    tiles[9]=(wd.pos_trans([8,7],[8,10]))
    tiles[9]+=(wd.pos_trans([9,8],[12,10]))  
    tiles[9]+=(wd.pos_trans([9,7],[11,7])) 

    tiles[10]=(wd.pos_trans([8,11],[10,15]))
    tiles[10]+=(wd.pos_trans([11,11],[11,13])) 
    tiles[11]=(wd.pos_trans([11,14],[11,15]))
    tiles[11]+=(wd.pos_trans([12,11],[14,15]))  
    tiles[12]=(wd.pos_trans([13,6],[15,10]))
    tiles[12]+=(wd.pos_trans([16,8],[16,10])) 
    tiles[13]=(wd.pos_trans([13,0],[15,5]))
    tiles[14]=(wd.pos_trans([16,0],[19,7]))

    tiles[15]=(wd.pos_trans([17,8],[19,13]))
    tiles[16]=(wd.pos_trans([15,11],[16,15]))
    tiles[16]+=(wd.pos_trans([17,14],[19,15]))
    '''
    #print(tiles_id)
    start_id=0
    for i in range(STG_NUM):
        tiles[i]=tiles_id[start_id:start_id+split[i]]
        start_id=start_id+split[i]    
    '''

    ops_per_stg=[]
    ops=[]
    a=[[0,1],[2,3,4,5],[6,7,8],[9,10,11],[12,13,14,15],[16,17,18],[19,20,21],[22,23,24],\
       [25,26,27,28],[29,30,31],[32,33,34],[35,36,37],[38,39,40],[41,42,43],[44,45,46,47],[48,49,50],[51,52,53]]
    j=0
    draw_mapping(wd,'ResNet50',tiles)
    for i,op_name in enumerate(model.op_dict):
        d_size=len(tiles[j])
        #print(j,tiles[j],d_size)
        #dp=DATA_PARALLELISM
        dp=d_size
        mp=d_size//dp
        assert mp*dp==d_size,'make sure that mp*dp=d_size'
        op=model.op_dict[op_name]
        if op.type==OP.Conv2:
            #print([dp,mp,d_size])
            if i>=44:
                if(len(tiles[j])==18):
                    op.dpmap(device_id=tiles[j],p_sgy=[3,2,1,1,3])
                    #print(1111)
                elif(len(tiles[j])==16):
                    #print(2222)
                    op.dpmap(device_id=tiles[j],p_sgy=[2,4,1,1,2])
                elif(len(tiles[j])==32):
                    #print(2222)
                    op.dpmap(device_id=tiles[j],p_sgy=[4,4,1,1,2])
                else:
                    #print(3333)
                    op.dpmap(device_id=tiles[j],p_sgy=[dp,mp,1,1,1])
            elif i>=25 and i<=28:
                #print(len(tiles[j]))
                op.dpmap(device_id=tiles[j],p_sgy=[d_size//2,2,1,1,1])
            else:
                op.dpmap(device_id=tiles[j],p_sgy=[dp,mp,1,1,1])
            
            #op.dpmap(device_id=tiles[j],p_sgy=[dp,mp,1,1,1])
        elif op.type==OP.Pool:
            #print('1',[dp,mp,d_size])
            op.dpmap(device_id=tiles[j],p_sgy=[dp,mp,1,1])
        if i not in a[j]:
            #print(i,j)
            ops_per_stg.append(ops)
            ops=[]
            j+=1
            #print(j)
        ops.append(op)
    if ops!=[]:
        ops_per_stg.append(ops)
    CompGraph.gwrite(model,path='model',name=model.name+'_map')
    stgs=[]
    #print(len(ops_per_stg))
    assert(len(tiles)==STG_NUM)
    for i in range(STG_NUM):
        last_core_id=None if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=None if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,tile_config,ops_per_stg[i],last_core_id,cur_core_id,next_core_id,noc=wd))
    return stgs