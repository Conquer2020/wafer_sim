import matplotlib.pyplot as plt
import os
from enum import Enum
from typing import List
from queue import Queue
from operator import mul
from functools import reduce
class BaseEnum(Enum):
    def __str__(self):
        return self.name
    
def mulc(a:List[int]):
    return reduce(mul,a)
def shape_suppose(size):
    R=0
    C=0
    if size==8000 or size==8192:
        R=128
        C=64
    if size==4000 or size==4096:
        R=64
        C=64
    if size==1000 or size== 1024:
        R=32
        C=32
    return [R,C]
def str2list(string):
    ls=[]
    string=string.split('[')[1].split(']')[0]
    string=string.split(',')

    for num_str in string:
        #print(int(num_str))
        num_str.split(',')
        ls.append(int(num_str))
    return ls

def str2strlist(string):
    ls=[]
    string=string.split('[')[1].split(']')[0]
    string=string.split(',')
    for str_ in string:
        if str_!='':
            str_str=str_.split('\'')
            ls.append(str_str[1])
    return ls

def split_comm_group(Group_Id,parall_dims):
    ''' 
    Here is an example :
    suppose Group_Id=[0,1,2,3,...,15],len=16
    1.if parall_dims=[16,1,1,1],group=[[0:15],[],[],[]]
    2.if parall_dims=[1,16,1,1],group=[[],[0:15],[],[]]
    3.if parall_dims=[8,2,1,1],group=
    [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    []
    []
    '''
    Group_Size=len(Group_Id)
    total_dims=1
    split_group=[]
    for dim in parall_dims:
        split_group.append(total_dims)
        total_dims*=dim
    assert Group_Size==total_dims,'Group_Size={},but total_dims={} '.format(Group_Size,total_dims)
    num_dims=len(parall_dims)
    groups=[]
    offset=Group_Size
    #print(split_group)
    for k in range(num_dims):
        temp_group_size=parall_dims[k]
        #print(temp_group_size)
        temp_group=[]
        if temp_group_size!=1:
            offset//=parall_dims[k]
            #print("offset",offset)
            for j in range(split_group[k]):
                #print(k,offset,j)
                for i in range(offset):
                    #print(i+j*(Group_Size//split_group[k]),(j+1)*Group_Size//split_group[k],offset)
                    temp_group.append(Group_Id[i+j*(Group_Size//split_group[k]):(j+1)*Group_Size//split_group[k]:offset])
        groups.append(temp_group)
    return groups
def draw_pipeline(trace_list,path,title,endtime,name='pipeline'):
    #print(trace_list)
    #[[(s,e),(s,e),(s,e)],[],[]], []=stages,s=micro_start_time,e=micro_end_time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #color = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#10c020', '#D20F99','#FFFFFF','#000000']
    color=['#63b2ee','#76da91','#f8cb7f','#f89588','#7cd6cf','#9192ab','#7898e1','#efa666','#eddd86','#9987ce','#63b2ee','#76da91','#100000']#maktalong
    num=len(trace_list)
    leng=len(trace_list[0])
    width_scale=max(trace_list[0][-1][1],trace_list[-1][-1][1])
    height_scale=4
    single_height=1
    start_height=single_height/2
    for j in range(num):
        k=0
        m=0
        for i in range(leng):
            x=trace_list[j][i]
            #facecolor=color[0] if x[2]==0 else color[5]
            if x[2]==0:
                facecolor=color[k % len(color)]
                k+=1
            else:
                facecolor=color[m % len(color)]
                m+=1
            edgecolor=color[-1]
            rect = plt.Rectangle((x[0],start_height+single_height*j),(x[1]-x[0]),single_height,fill=False,facecolor=facecolor,linewidth=0.5)
            ax.add_patch(rect)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.gcf().subplots_adjust(left=0.04,right=0.05)
    #plt.tight_layout()
    plt.title('{} stages ML {} pipeline [{:.1f} days]'.format(num,title,endtime))
    ax.set_ylim(0, num+1)
    ax.set_yticks([num-i for i in range(num)])
    ax.set_xlim(0, width_scale)
    #ax.set_aspect(20)
    plt.xlabel("Time")
    plt.ylabel("Stage")
    #plt.show()
    plt.savefig(os.path.join(path,name+'.png'))

def data_average(data,ave_size=1000):
    #data type[[start_time,end_time,max_resource],...]
    assert(ave_size>1)
    pass
def max_ave_1F1B_time(trace):
    max_1F1B_time=0
    tp_time=0
    stage_num=len(trace)
    fb_num=len(trace[0])
    #print(stage_num,fb_num)
    for i in range(stage_num):
        tp_time=0
        for j in range(fb_num):
            tp_time+=(trace[i][j][1]-trace[i][j][0])
        tp_time/=(fb_num /2)
        #print(i,tp_time)
        if max_1F1B_time< tp_time:
            max_1F1B_time=tp_time
            #print('max_index=',i)
    return max_1F1B_time

def visualize_resource(data:List,path,name,clear_redundance=True,max_resource=256,ave_unit_ms=1):
    #[(req_flag,req_time,len_q),(res_flag,res_time,len_q)]
    #print(data)
    if data==[]:
        return None
    q_req=Queue()
    occupy_list=[]
    for item in data:
        if item[0]=='req':
            q_req.put(item)
        elif item[0]=='res':
            req_item=q_req.get()
            res_item=item
            occupy_list.append([req_item[1],res_item[1],max_resource])
    if clear_redundance:
        leng=len(occupy_list)
        del_list=[]
        for i in range(1,leng):
            if occupy_list[i][0]==occupy_list[i-1][0]:
                del_list.append(i-1)
            elif occupy_list[i][0]<occupy_list[i-1][1]:
                del_list.append(i-1)
                occupy_list[i][0]=occupy_list[i-1][0]
        new_list=[]
        for i in range(leng):
            if i not in del_list:
                new_list.append(occupy_list[i])
    #print(data_list)  
    if ave_unit_ms!=1:
        list_ave=[]
        occupy_time=0
        time=ave_unit_ms
        for data in new_list:
            if data[1]<time and data!=new_list[-1]:
                occupy_time+=data[1]-data[0]
            else:
                if data==new_list[-1]:
                    occupy_time+=data[1]-data[0]
                ave_resource=max_resource*occupy_time/ave_unit_ms
                list_ave.append((time-ave_unit_ms,time,ave_resource))
                time+=ave_resource
                occupy_time=data[1]-data[0]
    data_list=list_ave if ave_unit_ms!=1 else new_list
    #[(start_time,end_time,resource_occupy)]
    #print(data_list)  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    data0=0
    for data in data_list:
        #plt.plot([data[0],data[1]],[data[2],data[2]],color='r',linewidth=2)
        plt.scatter(data[0],data[2],color='b')
        plt.scatter(data[1],data[2],color='r')
        #print(data[1],data[0])
        if data[0]>data0:
            plt.plot([data0,data[0]],[0,0],color='black',linewidth=1)
            data0=data[0]
    plt.xlabel("Time(ms)")
    plt.ylabel("Bandwidth(GB/s)")
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,name+'.png'))
    plt.close()
    return data_list
        
