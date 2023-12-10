from wafer_device import Wafer_Device 
from comp_graph import CompGraph
import pipeline_copy as pipe
from ML import *
import simpy
import math
from util import *

def mapping_ResNet50_tenstorrent(env:simpy.Environment,model:CompGraph,tile_config:dict,wd:Wafer_Device):
    STG_NUM=17
    DATA_PARALLELISM=1
    #tiles=[]
    #Layers_num=len(model)
    op_comp = []
    stage_comp = []
    stage_comp_temp = 0
    for op in model:
        op._analysis()
        # print(op.fd_macs_m)
        op_comp.append(op.fd_macs_m)
    a=[[0,1],[2,3,4,5],[6,7,8],[9,10,11],[12,13,14,15],[16,17,18],[19,20,21],[22,23,24],\
       [25,26,27,28],[29,30,31],[32,33,34],[35,36,37],[38,39,40],[41,42,43],[44,45,46,47],[48,49,50],[51,52,53]]
    for i in range (0,STG_NUM):
        index_len = len(a[i])
        for k in range(0,index_len):
            stage_comp_temp += op_comp[a[i][k]]
        stage_comp.append(stage_comp_temp)
        stage_comp_temp = 0
    
    # test
    # for i in range(0,STG_NUM):
    #    print(stage_comp[i])

    sum_comp = sum(stage_comp)
    tile_alloc = []
    for i in range(0,STG_NUM):
        tile_alloc.append(max(1,round(stage_comp[i]*120/sum_comp)))
    
    # for i in range(0,STG_NUM):
    #     print(tile_alloc[i])

    stgs=[]
    #total_macs_m=0
    #total_tile=4*4*4*5
    tiles=[[] for _ in range(STG_NUM)]
    split=[2,6,18,18,18,18,18,18,46,18,18,18,18,18,18,18,32,12,12]
    #print(len(split))
    #tiles_id=wd.device() 

    # 1tile
    # map 1 & 2
    tiles[0]=wd.pos_trans([0,0,0,0])

    # 2 tile
    # map 1 & 2
    tiles[1]=wd.pos_trans([1,0],[2,0])
    #print(tiles[1])

    # 7 tile map 1
    # tiles[2]=wd.pos_trans([3,0],[9,0])
    # 6 tile map 2
    tiles[2]=wd.pos_trans([0,1],[2,2])

    # 7tile
    # tiles[3]=wd.pos_trans([3,1],[9,1])
    # 6 tile map 2
    tiles[3]=wd.pos_trans([3,0],[5,1])

    # 14tile
    #tiles[4]=wd.pos_trans([0,1],[1,7])
    # 16 tile map 2
    tiles[4]=wd.pos_trans([6,0],[9,3])

    # 7tile
    # tiles[5]=wd.pos_trans([3,2],[9,2])
    # 6 tile map 2
    tiles[5]=wd.pos_trans([3,2],[5,3])
    
    # 7tile
    # tiles[6]=(wd.pos_trans([3,3],[9,3])).
    # 6 tile map 2
    tiles[6]=wd.pos_trans([0,3],[2,4])
    
    # 7tile
    # tiles[7]=(wd.pos_trans([2,1],[2,7]))
    # 6 tile map 2
    tiles[7]=wd.pos_trans([3,4],[5,5])
    
    # 14tile
    # tiles[8]=(wd.pos_trans([3,4],[9,5]))
    # 16 tile map 2
    tiles[8]=wd.pos_trans([6,4],[9,7])

    #print(len(tiles[8])) 
    # 6tile
    # tiles[9]=(wd.pos_trans([0,8],[2,9]))
    # 6 tile map 2
    tiles[9]=wd.pos_trans([3,6],[5,7])

    # 6tile
    # tiles[10]=(wd.pos_trans([0,10],[2,11]))
    # 6 tile map 2
    tiles[10]=wd.pos_trans([0,5],[2,6])
    
    # 6tile
    # tiles[11]=(wd.pos_trans([3,10],[5,11]))
    # 6 tile map 2
    tiles[11]=wd.pos_trans([0,7],[2,8])

    # 7tile
    # tiles[12]=(wd.pos_trans([3,6],[9,6]))
    # 6 tile map 2
    tiles[12]=wd.pos_trans([3,8],[5,9])
    
    # 7tile
    # tiles[13]=(wd.pos_trans([3,7],[9,7]))
    # 6 tile map 2
    tiles[13]=wd.pos_trans([3,10],[5,11])
   
    # 12 tile
    # tiles[14]=(wd.pos_trans([3,8],[7,8]))
    # 16 tile map 2
    tiles[14]=wd.pos_trans([6,8],[9,11])

    # 5tile
    # tiles[15]=(wd.pos_trans([3,9],[7,9]))
    # 4 tile map 2
    tiles[15]=wd.pos_trans([1,9],[2,10])
    
    # 5tile
    #tiles[16]=(wd.pos_trans([8,8],[9,9]))
    #tiles[16]+=(wd.pos_trans([6,10],[9,11]))
    # 5 tile map 2
    tiles[16]=wd.pos_trans([0,9],[0,11])
    tiles[16]+=(wd.pos_trans([1,11],[2,11]))
    
    '''
    #print(tiles_id)
    start_id=0
    for i in range(STG_NUM):
        tiles[i]=tiles_id[start_id:start_id+split[i]]
        start_id=start_id+split[i]    
    '''
    
    ops_per_stg=[]
    ops=[]
    j=0
    draw_mapping(wd,'ResNet50_tenstorrent',tiles)
    for i,op_name in enumerate(model.op_dict):
        d_size=len(tiles[j])
        #print(j,tiles[j],d_size)
        #dp=DATA_PARALLELISM
        dp=d_size 
        mp=d_size//dp
        assert mp*dp==d_size,'make sure that mp*dp=d_size'
        op=model.op_dict[op_name]
        if op.type==OP.Conv2:
            print([dp,mp,d_size])
            if i>=44:
                if(len(tiles[j])==18):
                    op.dpmap(device_id=tiles[j],p_sgy=[3,2,1,1,3])
                    #print(1111)
                elif(len(tiles[j])==16):
                    #print(2222)
                    op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,1,16])
                    # op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,1,16])
                elif(len(tiles[j])==32):
                    #print(2222)
                    op.dpmap(device_id=tiles[j],p_sgy=[4,4,1,1,2])
                elif(len(tiles[j])==12):
                    op.dpmap(device_id=tiles[j],p_sgy=[4,1,1,1,3])
                elif(len(tiles[j])==6):
                    #op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,2,3])
                    op.dpmap(device_id=tiles[j],p_sgy=[6,1,1,1,1])
                elif(len(tiles[j])==4):
                    op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,1,4])
                else:
                    #print(3333)
                    # op.dpmap(device_id=tiles[j],p_sgy=[dp,mp,1,1,1])
                    op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,mp,dp])
            # elif i>=25 and i<=28:
                #print(len(tiles[j]))
                # op.dpmap(device_id=tiles[j],p_sgy=[d_size//2,2,1,1,1])
            else:
                if(len(tiles[j])==12):
                    op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,1,12])
                elif(len(tiles[j])==6):
                    # op.dpmap(device_id=tiles[j],p_sgy=[2,1,1,1,3])
                    op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,1,6])
                elif(len(tiles[j])==4):
                    op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,1,4])
                else:
                    #op.dpmap(device_id=tiles[j],p_sgy=[dp,mp,1,1,1])
                    op.dpmap(device_id=tiles[j],p_sgy=[1,1,1,mp,dp])
                    #op.dpmap(device_id=tiles[j],p_sgy=[1,dp,1,mp,1])
            
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
    print(len(ops_per_stg))
    assert(len(tiles)==STG_NUM)
    for i in range(STG_NUM):
        last_core_id=None if i==0 else tiles[i-1]
        cur_core_id=tiles[i]
        next_core_id=None if i==STG_NUM-1 else tiles[i+1]
        stgs.append(pipe.Stage(env,tile_config,ops_per_stg[i],last_core_id,cur_core_id,next_core_id,noc=wd))
    return stgs

def mapping_BERT_BASE_tenstorrent(env:simpy.Environment,model:CompGraph,tile_config:dict,wd:Wafer_Device):
    STG_NUM=14
    DATA_PARALLELISM=1
    #tiles=[]
    #Layers_num=len(model)
    op_comp = []
    stage_comp = []
    stage_comp_temp = 0

    stgs=[]
    #total_macs_m=0
    #total_tile=4*4*4*5
    tiles=[[] for _ in range(STG_NUM)]
    split=[2,6,18,18,18,18,18,18,46,18,18,18,18,18,18,18,32,12,12]
    #print(len(split))
    #tiles_id=wd.device() 
    op_comp = []
    stage_comp = []
    stage_comp_temp = 0
    for op in model:
        op._analysis()
        print(op.w_s_g_size_m)
        op_comp.append(op.fd_macs_m)
        # print(op.fd_macs_m)
    # test
    # for i in range(0,STG_NUM):
    #    print(stage_comp[i])

        # print(max(1,round(op_comp[i]*120/sum_comp)))
    '''
    #print(tiles_id)
    start_id=0
    for i in range(STG_NUM):
        tiles[i]=tiles_id[start_id:start_id+split[i]]
        start_id=start_id+split[i]    
    '''
    tiles[0] = wd.pos_trans([0,0])
    tiles[1] = wd.pos_trans([1,0],[4,0])
    tiles[1] += wd.pos_trans([0,1],[4,1])

    for i in range(2,7):
        tiles[i]=wd.pos_trans([0,2*(i-1)],[4,2*(i-1)+1])
    for i in range(7,12):
        tiles[i]=wd.pos_trans([5,-2*i+24],[9,-i*2+24+1])

    tiles[12]=wd.pos_trans([9,0])
    tiles[12]+=wd.pos_trans([5,1],[9,1])
    tiles[13]=wd.pos_trans([5,0],[8,0])

    # tiles[0] = wd.pos_trans([0,0],[3,0])
    # tiles[1] = wd.pos_trans([4,0],[9,0])

    # for i in range(2,12):
    #     tiles[i]=wd.pos_trans([0,i-1],[9,i-1])

    # tiles[12] = wd.pos_trans([0,11],[5,11])
    # tiles[13] = wd.pos_trans([6,11],[9,11])

    # a = []
    # tiles[0] = wd.pos_trans([0,0])
    # tiles[1] = wd.pos_trans([1,0],[4,0])
    # tiles[2] = wd.pos_trans([3,1],[4,2])
    # tiles[3] = wd.pos_trans([0,1],[2,2])
    # tiles[4] = wd.pos_trans([0,3],[1,4])
    # tiles[5] = wd.pos_trans([2,3],[4,4])
    # tiles[6] = wd.pos_trans([3,5],[4,6])
    # tiles[7] = wd.pos_trans([0,5],[2,6])
    # tiles[8] = wd.pos_trans([0,7],[2,8])
    # tiles[9] = wd.pos_trans([3,7],[4,8])
    # tiles[10] = wd.pos_trans([3,9],[4,10])
    # tiles[11] = wd.pos_trans([0,9],[2,10])
    # tiles[12] = wd.pos_trans([0,11],[4,11])
    # tiles[13] = wd.pos_trans([5,11],[9,11])
    # tiles[14] = wd.pos_trans([5,9],[6,10])
    # tiles[15] = wd.pos_trans([7,9],[9,10])
    # tiles[16] = wd.pos_trans([8,7],[9,8])
    # tiles[17] = wd.pos_trans([5,7],[7,8])
    # tiles[18] = wd.pos_trans([5,5],[6,6])
    # tiles[19] = wd.pos_trans([7,5],[9,6])
    # tiles[20] = wd.pos_trans([8,3],[9,4])
    # tiles[21] = wd.pos_trans([5,3],[7,4])
    # tiles[22] = wd.pos_trans([5,1],[7,2])
    # tiles[23] = wd.pos_trans([8,1],[9,2])
    # tiles[24]=wd.pos_trans([6,0],[9,0])
    # tiles[25]=wd.pos_trans([5,0])

    draw_mapping(wd,'Bert_BASE_tenstorrent',tiles)
    
    ops_per_stg=[]
    for i,op_name in enumerate(model.op_dict):
        #print(op_name)
        d_size=len(tiles[i])
        # print(d_size)
        dp=DATA_PARALLELISM
        mp=d_size//dp
        #assert mp*dp==d_size,'make sure that mp({})*dp({})=d_size({})'.format(mp,dp,d_size)
        op=model.op_dict[op_name]
        if op.type==OP.Transformer:
            #print([dp,mp,d_size])
            # op.dpmap(device_id=tiles[i],p_sgy=[dp,mp])
            if d_size == 4:
                op.dpmap(device_id=tiles[i],p_sgy=[4,1])
            if d_size == 6:
                op.dpmap(device_id=tiles[i],p_sgy=[6,1])
            if d_size == 10:
                op.dpmap(device_id=tiles[i],p_sgy=[2,5])   
            op.dpmap(device_id=tiles[i],p_sgy=[mp,dp])
        elif op.type==OP.Embedding:
            op.dpmap(device_id=tiles[i],p_sgy=[d_size,1])
            #op.dpmap(device_id=tiles[i],p_sgy=[dp,mp])
        elif op.type==OP.Linear:
            # op.dpmap(device_id=tiles[i],p_sgy=[dp,mp,1,1])
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