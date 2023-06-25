import matplotlib.pyplot as plt
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
def draw_pipeline(trace_list,path,title,name='pipeline.png'):
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
            rect = plt.Rectangle((x[0],start_height+single_height*j),(x[1]-x[0]),single_height,fill=True,facecolor=facecolor,linewidth=0.5)
            ax.add_patch(rect)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('{} stages ML {} pipeline'.format(num,title))
    ax.set_ylim(0, num+1)
    ax.set_yticks([num-i for i in range(num)])
    ax.set_xlim(0, width_scale)
    #ax.set_aspect(20)
    plt.xlabel("Time")
    plt.ylabel("Stage")
    plt.savefig(path+name)

def visualize_resource(data:List,name,clear_redundance=True,max_resource=256,ave_unit_ms=1):
    #[(req_flag,req_time,len_q),(res_flag,res_time,len_q)]
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
    print(data_list)  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    data0=0
    for data in data_list:
        #plt.plot([data[0],data[1]],[data[2],data[2]],color='r',linewidth=4)
        plt.scatter(data[0],data[2],color='r')
        plt.scatter(data[1],data[2],color='r')
        if data[0]>data0:
            plt.plot([data0,data[0]],[0,0],color='black',linewidth=2)
            data0=data[0]
    plt.xlabel("Time(ms)")
    plt.ylabel("Bandwidth(GB/s)")
    #plt.show()
    plt.savefig(name+'.png')     
    return data_list
        
