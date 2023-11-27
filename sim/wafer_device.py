import simpy
from monitored_resource import MonitoredResource as Resource
from typing import List,Union
import random
from util import *
from functools import wraps
import shutil
import numpy as np
DIRECT=Enum('DIRECT',('LEFT','RIGHT','UP','DOWN'))
class Packet():
    def __init__(self,id,shape:List[int],meta_data='test') -> None:
        self.id=id
        self.shape=shape
        self.size=self._size_MB()
        self.meta_data=meta_data
    def _size_MB(self):
        temp=mulc(self.shape)
        return temp/1000/1000
    def __str__(self): 
        return 'Packet:(id:{},shape:{},size:{} MByte,meta:{})'.format(self.id,self.shape,self.size,self.meta_data)
    @staticmethod
    def random_gen():
        id=random.randint(0,10000)
        shape=[]
        shape_dim=random.randint(1,2)
        for i in range(shape_dim):
            shape.append(random.randint(1,128))
        return Packet(id=id,shape=shape)
'''
class DDR_model():
    def __init__(self,name,env,transfer_rate_M,channel_num,die_num,per_die_cap_GB,bit_width=32) -> None:
        self.name=name
        self.transfer_rate_M=transfer_rate_M
        self.channel_num=channel_num
        self.die_num=die_num
        self.die_cap_GB=per_die_cap_GB
        self.bit_width=bit_width
        self.capacity=die_num*per_die_cap_GB
        self.env=env
        self.access_resource=Resource(self.env,capacity=1)
        self.bandwidth=transfer_rate_M*bit_width/8*channel_num #GB/s
    def access_process(self,data_MB,task_id=1,write=True,DEBUG_MODE=False):
        with self.access_resource.request() as req:
            yield req 
            latency=data_MB/self.bandwidth
            #latency+=self.write_latency if write else self.read_latency
            yield self.env.timeout(latency)
    def gen_latency(self,data_MB):
        return data_MB/self.bandwidth
'''
class dram_model():
    def __init__(self,name,env,attach_tile_id,bw_GB=256,capacity_GB=16*100,read_latency_ms=0,write_latency_ms=0) -> None:
        self.name=name
        self.bw_GB=bw_GB
        self.read_latency=read_latency_ms
        self.write_latency=write_latency_ms
        self.attach_tile_id=attach_tile_id
        #TODO consider the dram capacity influence for total ml network
        #@fangjh21.20230602: related to embedding op when ml network is recommoned system like DLRM ,etc.
        self.capacity=capacity_GB 
        self.env=env
        self.access_resource=Resource(self.env,capacity=1)
    def access_process(self,data_MB,task_id=1,write=True,DEBUG_MODE=False):
        with self.access_resource.request() as req:
            yield req 
            latency=data_MB/self.bw_GB
            latency+=self.write_latency if write else self.read_latency
            yield self.env.timeout(latency)
    def analytical_latency(self,data_MB):
        return data_MB/self.bw_GB
class Router_Link():
    def __init__(self,env,attach_tile_id,bw_GB:List[float],link_latency_ns) -> None:
        self.env=env
        self.attach_tile_id=attach_tile_id
        self.bw_GB={
            DIRECT.LEFT:bw_GB[0],
            DIRECT.RIGHT:bw_GB[1],
            DIRECT.UP:bw_GB[2],
            DIRECT.DOWN:bw_GB[3]
                  } 
        self.link_latency_ns=link_latency_ns
        self.res={
            DIRECT.LEFT:Resource(self.env,capacity=1),
            DIRECT.RIGHT:Resource(self.env,capacity=1),
            DIRECT.UP:Resource(self.env,capacity=1),
            DIRECT.DOWN:Resource(self.env,capacity=1)
                  }      
    def transfer_process(self,data_MB,direct=DIRECT.LEFT):
        with self.res[direct].request() as req:
            yield req 
            latency=data_MB/self.bw_GB[direct]
            latency+=self.link_latency_ns 
            yield self.env.timeout(latency)
    def analytical_latency(self,data_MB,direct=DIRECT.LEFT):
        return data_MB/self.bw_GB[direct]
    
class Wafer_Device():
    def __init__(self,env,wafer_name='test_wafer',
                tile_shape=[4,4],die_shape=[2,2],
                tile_noc_bw_GB=256,
                die_noc_bw_GB=256*0.6,
                die_dram_bw_GB=12288/16/8,
                die_dram_cap_GB=6/16,
                edge_die_dram_bw_GB=256,
                clk_freq_GHz=1,
                with_dram_per_die=True,
                Analytical=True
                ) -> None:
        self.wafer_name=wafer_name

        self.tile_shape=tile_shape
        self.die_shape=die_shape
        self.tile_noc_bw_GB=tile_noc_bw_GB
        self.die_noc_bw_GB=die_noc_bw_GB
        self.with_dram_per_die=with_dram_per_die
        self.die_dram_bw_GB=die_dram_bw_GB
        self.die_dram_cap_GB=die_dram_cap_GB

        self.edge_die_dram_bw_GB=edge_die_dram_bw_GB
        self.noc_response_latency_ms=0
        self.dram_response_latency_ms=0
        self.route_XY='X'
        self.clk_freq_GHz=clk_freq_GHz
        self.env=env
        self.Analytical=Analytical
        self.adjacency_matrix=0

        if not Analytical:
            self.link_resource=[]
            self.ddr_per_die_resource={}
            #maybe in real system,there is one dram per die 
            #self.dram_per_die_resource=[]
            self.edge_dram_resource={}
            self.__create_resource()
    def wafer_info(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            print('----------wafer-scale infomation----------')
            print('2D mesh {}:{}x{},{}x{}'.format(self.wafer_name,self.die_shape[0],self.die_shape[1],self.tile_shape[0],self.tile_shape[1]))
            return func(self, *args, **kwargs)
        return wrapper
    def device_list(self):
        x0=self.tile_shape[0]
        x1=self.die_shape[0]
        y0=self.tile_shape[1]
        y1=self.die_shape[1]
        return [i for i in range(x1*x0*y1*y0)]
    @wafer_info
    def __create_resource(self):
        x0=self.tile_shape[0]
        x1=self.die_shape[0]
        y0=self.tile_shape[1]
        y1=self.die_shape[1]
        #TODO
        #self.adjacency_matrix=np.zeros(x1*x0*y1*y0,x1*x0*y1*y0)
        for yy1 in range(y1):
            for xx1 in range(x1):
                for yy0 in range(y0):
                    for xx0 in range(x0):
                        i=xx0+yy0*x0+xx1*y0*x0+x1*y0*x0*yy1 
                        bw_left=   self.tile_noc_bw_GB
                        bw_right=   self.tile_noc_bw_GB
                        bw_up=   self.tile_noc_bw_GB
                        bw_down=   self.tile_noc_bw_GB 

                        if xx0==0:
                            if yy0==0:
                                bw_left=self.die_noc_bw_GB
                                bw_up=self.die_noc_bw_GB
                            elif yy0==y0-1:
                                bw_right=self.die_noc_bw_GB
                                bw_up=self.die_noc_bw_GB
                            else:
                                bw_up=0
                        elif xx0==x0-1:
                            if yy0==0:
                                bw_left=self.die_noc_bw_GB
                                bw_down=self.die_noc_bw_GB
                            elif yy0==y0-1:
                                bw_right=self.die_noc_bw_GB
                                bw_down=self.die_noc_bw_GB
                            else:
                                bw_down=0
                        else:
                            if yy0==0:
                                if self.with_dram_per_die:
                                    bw_left=0# to ddr on chip left
                                    self.ddr_per_die_resource.append(dram_model('LPDDR',self.env,i,self.die_dram_bw_GB,self.die_dram_cap_GB))
                            elif yy0==y0-1:
                                if self.with_dram_per_die:
                                    bw_right=0 # to ddr on chip right
                                    self.ddr_per_die_resource.append(dram_model('LPDDR',self.env,i,self.die_dram_bw_GB,self.die_dram_cap_GB))
                            else:
                                pass
                        self.link_resource.append(Router_Link(self.env,i,[bw_left,bw_right,bw_up,bw_down],0))
                      
        for _ in range(x1):
            self.edge_dram_resource.append(dram_model('DDR',self.env,self.edge_die_dram_bw_GB))#left dram
        for _ in range(x1):
            self.edge_dram_resource.append(dram_model('DDR',self.env,self.edge_die_dram_bw_GB))#right dram
        print('edge dram resource is created...')

    def Manhattan_hops(self,src_id,dis_id):
        x=self.tile_shape[0]*self.die_shape[0]
        y=self.die_shape[1]*self.tile_shape[1]
        min_id=min(src_id,dis_id)
        max_id=max(src_id,dis_id)
        res=max_id-min_id
        if (max_id% y)>=(min_id % y):
            return (res%y)+(res//y)
        else:
            return (min_id% y)-(max_id % y)+(max_id//y)


    def route_gen(self,src_id,des_id,DEBUG_MODE=True):
        x=self.tile_shape[0]*self.die_shape[0]
        y=self.die_shape[1]*self.tile_shape[1]
        assert src_id != des_id, "Source and destination IDs must be different"
        assert 0 <= des_id < (x * y), "Destination ID out of range"
        route_list =[src_id]
        if self.route_XY=='X':
            while(src_id!=des_id):
                if((src_id % y) >(des_id % y)):
                    src_id-=1
                elif ((src_id % y) <(des_id % y)):
                    src_id+=1
                else:
                    src_id += y if src_id < des_id else -y
                route_list .append(src_id)
        else:
            pass
        #if DEBUG_MODE:
        #    print('Router_List:{}'.format(list))
        return route_list 
    def link_gen(self,src_id,des_id,DEBUG_MODE=False):
        x0=self.tile_shape[0]
        x1=self.die_shape[0]
        y0=self.tile_shape[1]
        y1=self.die_shape[1]
        Y_OFFSET=(y0*y1-1)*x0*x1
        route_list=self.route_gen(src_id,des_id,DEBUG_MODE=DEBUG_MODE)
        distence=len(route_list)
        link_list=[]
        for i in range(distence-1):
            if abs(route_list[i+1]-route_list[i])==1:
                temp=min(route_list[i],route_list[i+1])
                t1=(temp //( y0*y1))*(y0*y1-1)
                t2=(temp % (y0*y1))
                X_INDEX=t1+t2
                link_list.append(X_INDEX)
            elif abs(route_list[i+1]-route_list[i])==y0*y1:
                Y_INDEX=min(route_list[i],route_list[i+1])
                link_list.append(Y_OFFSET+Y_INDEX)
            else:
                raise NotImplemented
        return link_list
    def is_inter_link(self,link_id):
        x0=self.tile_shape[0]
        x1=self.die_shape[0]
        y0=self.tile_shape[1]
        y1=self.die_shape[1]
        Y_OFFSET=(x0*x1-1)*y0*y1
        if link_id<Y_OFFSET:
            if (link_id +1) % x0==0:
                return True
            else:
                return False
        else:
            offset_link_id=link_id-Y_OFFSET
            if (int(offset_link_id /(x0*x1))+1) ==y0:
                return True
            else:
                return False
    def noc_process(self,comm_size_MB,src_id,des_id,task_id=1,DEBUG_MODE=False):
        assert(src_id!=des_id)
        ListID=self.link_gen(src_id,des_id,DEBUG_MODE)
        while(True):
            for i in ListID:
                if not self.Analytical:
                    with self.link_resource[i].request() as req: 
                        yield req
                        if self.is_inter_link(i):
                            yield self.env.timeout(self.noc_response_latency_ms+comm_size_MB/self.die_noc_bw_GB)
                        else:
                            yield self.env.timeout(self.noc_response_latency_ms+comm_size_MB/self.tile_noc_bw_GB)
                else:
                    if self.is_inter_link(i):
                            yield self.env.timeout(self.noc_response_latency_ms+comm_size_MB/self.die_noc_bw_GB)
                    else:
                            yield self.env.timeout(self.noc_response_latency_ms+comm_size_MB/self.tile_noc_bw_GB)
            break
    def edge_dram_write_process(self,access_size_MB,src_id,task_id='DDR_READ_TEST',DEBUG_MODE=False):
        #TODO 
        x1=self.die_shape[0]
        y=self.tile_shape[1]*self.die_shape[1]
        row_line=int(src_id /y)+1
        des_id=row_line*y-1 if (row_line*y-1-src_id)<(y/2) else (row_line-1)*y
        while(True):
            #if DEBUG_MODE:
            #    print("task {} start dram wrtie  @ {:.3f} ms".format(task_id,self.envenv.now))
            if des_id!=src_id:
                yield self.env.process(self.noc_process(access_size_MB,src_id,des_id,task_id=task_id,DEBUG_MODE=DEBUG_MODE))
            if not self.Analytical:
                dram_index=int(des_id/y) if (des_id % y)  ==0 else int(des_id/ y)+x1
                yield self.env.process(self.edge_dram_resource[dram_index].access_process(access_size_MB,task_id=task_id,write=True))
            else:
                yield self.env.timeout(self.dram_response_latency_ms+access_size_MB/self.die_dram_bw_GB)    
            #if DEBUG_MODE:
            #print("task {} end dram wrtie  @ {:.3f} ms".format(task_id,self.env.now))
            break
    def edge_dram_read_process(self,access_size_MB,src_id,task_id='DDR_READ_TEST',DEBUG_MODE=True):
        x1=self.die_shape[0]
        x0=self.tile_shape[0]
        y=self.tile_shape[1]*self.die_shape[1]
        row_line=int(src_id /y)+1
        des_id=row_line*y-1 if (row_line*y-1-src_id)<(y/2) else (row_line-1)*y
        while(True):
            #if DEBUG_MODE:
            #    print("task {} start dram read  @ {:.3f} ms".format(task_id,self.env.now))
            dram_index=int(des_id/y) if (des_id % y)  ==0 else int(des_id/ y)+x1
            '''
            print('int(des_id/ y)',int(des_id/ y))
            print('x1',x1)
            print('int(des_id/ y)+x1',int(des_id/ y)+x1)
            print('dram_index',dram_index)
            print(len(self.edge_dram_resource))
            '''
            if not self.Analytical:
                yield self.env.process(self.edge_dram_resource[dram_index].access_process(access_size_MB,task_id=task_id,write=False))
            else:
                yield self.env.timeout(self.dram_response_latency_ms+access_size_MB/self.edge_die_dram_bw_GB)            
            if des_id!=src_id:
                yield self.env.process(self.noc_process(access_size_MB,des_id,src_id,task_id=task_id,DEBUG_MODE=DEBUG_MODE))
            #if DEBUG_MODE:
            #    print("task {} end dram read @ {:.3f} ms".format(task_id,self.env.now))
            break
    def tile_dram_access_process(self,access_size_MB,src_id,task_id='3DDRAM-TEST',WRITE=True,DEBUG_MODE=False):
        while(True):
            assert(self.with_dram_per_die)
            if not self.Analytical:
                yield self.env.process(self.ddr_per_die_resource[src_id].access_process\
                                    (access_size_MB,task_id=task_id,write=WRITE,DEBUG_MODE=DEBUG_MODE))
            else:
                yield self.env.timeout(self.dram_response_latency_ms+access_size_MB/self.die_dram_bw_GB)
            break
    def tile_dram_group_access_process(self,access_size_MB,group_id:List[int],task_id='3DDRAM-TEST',WRITE=True,DEBUG_MODE=False):
        for id in group_id:
            yield self.env.process(self.tile_dram_access_process(access_size_MB,id,task_id,WRITE,DEBUG_MODE))

    def dram_read_group_process(self,access_size_MB:Union[int,List[int]],group_id:List[int],task_id,multicast=True):
        #TODO 优化
        if type(access_size_MB) is list:
            temp=mulc(access_size_MB)
            access_size_MB=temp/1000/1000*2  
            #print(access_size_MB) 
        while(True):
            #print("task {} start dram_read_group_process @ {:.3f} ms".format(task_id,self.env.now))
            #print(group_id[0])
            yield self.env.process(self.edge_dram_read_process(access_size_MB,group_id[0],task_id))
            g_size=len(group_id)  
            for i in range(1,g_size):
                comm_size=access_size_MB/g_size if not multicast else access_size_MB
                yield self.env.process(\
                   self.noc_process(comm_size,group_id[i-1],group_id[i],task_id))
            #print("task {} end dram_read_group_process @ {:.3f} ms".format(task_id,self.env.now))
            break
    def dram_write_group_process(self,access_size_MB:Union[int,List[int]],group_id:List[int],task_id,gather=True):
        #TODO 优化
        if type(access_size_MB) is list:
            temp=mulc(access_size_MB)
            access_size_MB=temp/1000/1000*2
        while(True):
            g_size=len(group_id)
            if gather:  
                for i in range(g_size-1,0,-1):
                    comm_size=access_size_MB/g_size
                    yield self.env.process(\
                        self.noc_process(comm_size,group_id[i],group_id[0],task_id))
            yield self.env.process(self.edge_dram_write_process(access_size_MB,group_id[0],task_id))
            break

    def ALL_REDUCE_process(self,comm_size,group_id:List[int],task_id,DEBUG_MODE=False):
        # TODO 完成通信原语及其优化
        #yield self.env.timeout(5)
        group_size=len(group_id)
        chunk_size=comm_size/group_size
        #if DEBUG_MODE:
        #        print("ALL_REDUCE task {} start @ {:.3f} ms".format(task_id,self.env.now))
        #t_last=self.env.now
        for i in range(group_size-1):
            event_list=[]
            for id_idx in range(group_size-1):
                event_list.append(self.env.process(self.noc_process(chunk_size,group_id[id_idx],group_id[id_idx+1])))
            event_list.append(self.env.process(self.noc_process(chunk_size,group_id[-1],group_id[0])))
            yield simpy.AllOf(self.env, event_list)
            #if DEBUG_MODE:
            #    print('Reduce-Scatter {}/{} phase'.format(i+1,group_size-1))
        for i in range(group_size-1):
            event_list=[]
            for id_idx in range(group_size-1):
                event_list.append(self.env.process(self.noc_process(chunk_size,group_id[id_idx],group_id[id_idx+1])))
            event_list.append(self.env.process(self.noc_process(chunk_size,group_id[-1],group_id[0])))
            yield simpy.AllOf(self.env, event_list)
            #if DEBUG_MODE:
            #    print('All-Gather {}/{} phase'.format(i+1,group_size-1))
        #if DEBUG_MODE:
            #    print("ALL_REDUCE task {} end @ {:.3f} ms".format(task_id,self.env.now))
        #print("ALL_REDUCE task {} end with {:.3f} ms".format(task_id,self.env.now-t_last))
        
    def ALL_2_ALL_process(self,comm_size,group_id:List[int],task_id,DEBUG_MODE=False):
        # TODO 完成通信原语及其优化
        group_size=len(group_id)
        #print(group_size)
        chunk_size=comm_size/group_size
        for i in range(group_size-1):
            event_list=[]
            for id_idx in range(group_size):
                des_id=(id_idx+i+1 )% group_size
                event_list.append(self.env.process(self.noc_process(chunk_size,group_id[id_idx],group_id[des_id])))
            yield simpy.AllOf(self.env, event_list)
    def STAGE_PASS_process(self,comm_size:Union[int,Packet],group_a:List[int],group_b:List[int],task_id,DEBUG_MODE=False):
        # TODO 完成通信原语
        if type(comm_size) is Packet:
            comm_size=comm_size.size
        distance=[]
        for i in group_a:
            for j in group_b:
                distance.append(self.Manhattan_hops(i,j))
        index=distance.index(min(distance))
        src=group_a[index // len(group_b)]
        des= group_b[index % len(group_b)]
        #if DEBUG_MODE:
            #print('Group_A {} to Group_B stage pass {}'.format(src,des))
        while(True):
            #if DEBUG_MODE:
            #    print("STAGE_PASS task {} start @ {:.3f} ms".format(task_id,self.env.now))
            for i in group_a:
                if i!=src:
                    yield self.env.process(self.noc_process(comm_size,i,src,task_id,DEBUG_MODE))
            all_comm_size=comm_size*len(group_a)
            yield self.env.process(self.noc_process(all_comm_size,src,des,task_id,DEBUG_MODE))
            for j in group_b:
                if j!=des:
                    yield self.env.process(self.noc_process(all_comm_size/len(group_b),des,j,task_id,DEBUG_MODE))
            #if DEBUG_MODE:
            #    print("STAGE_PASS task {} start @ {:.3f} ms".format(task_id,self.env.now))
            break

    def resource_visualize(self,res_type:str='edge_dram',path='./status/resource/',clear=True):
        if clear:
            ls = os.listdir(path)
            for i in ls:
                f_path = os.path.join(path, i)
                #print(f_path)
                shutil.rmtree(f_path)
        if res_type=='all':
            for index,res in enumerate(self.edge_dram_resource):
                visualize_resource(res.access_resource.data,path+'edge_dram',str(index),max_resource=self.edge_die_dram_bw_GB)
            for index,res in enumerate(self.ddr_per_die_resource):
                visualize_resource(res.access_resource.data,path+'3ddram',str(index),max_resource=self.die_dram_bw_GB)
            path1=path+'inter_noc'
            path2=path+'intra_noc'
            for index,res in enumerate(self.link_resource):
                if self.is_inter_link(index):
                    visualize_resource(res.data,path1,str(index),max_resource=self.die_noc_bw_GB)
                else:
                    visualize_resource(res.data,path2,str(index),max_resource=self.tile_noc_bw_GB)
        elif res_type=='edge_dram':
            for index,res in enumerate(self.edge_dram_resource):
                visualize_resource(res.access_resource.data,path+'edge_dram',str(index),max_resource=self.edge_die_dram_bw_GB)
        elif res_type=='3ddram':
            for index,res in enumerate(self.ddr_per_die_resource):
                visualize_resource(res.access_resource.data,path+'3ddram',str(index),max_resource=self.die_dram_bw_GB)
        elif res_type=='noc' :
            path1=path+'inter_noc'
            path2=path+'intra_noc'
            for index,res in enumerate(self.link_resource):
                if self.is_inter_link(index):
                    visualize_resource(res.data,path1,str(index),max_resource=self.die_noc_bw_GB)
                else:
                    visualize_resource(res.data,path2,str(index),max_resource=self.tile_noc_bw_GB)
        else:
            raise NotImplementedError
if __name__ == '__main__':
    Debug=True
    env = simpy.Environment()
    wd=Wafer_Device(env,die_shape=[4,4],tile_shape=[4,4],with_dram_per_die=True,Analytical=True)
    '''
    env.process(wd.noc_process(10,src_id=0,des_id=3,task_id=1,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=3,des_id=0,task_id=2,DEBUG_MODE=Debug))
    env.process(wd.edge_dram_read_process(10,src_id=1,DEBUG_MODE=Debug))
    env.process(wd.edge_dram_read_process(10,src_id=1,task_id=4,DEBUG_MODE=Debug))
    env.process(wd.edge_dram_write_process(16,src_id=1,task_id=5,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=13,des_id=15,task_id=6,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=13,des_id=15,task_id=7,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=13,des_id=15,task_id=8,DEBUG_MODE=Debug))
    env.process(wd.STAGE_PASS_process(10,[0,1,2,3,5],[8,9],'TEST'))
    '''
    #env.process(wd.tile_dram_access_process(0,63,'TEST_3DDRAM',DEBUG_MODE=Debug))
    env.process(wd.edge_dram_read_process(100,7))
    env.process(wd.edge_dram_read_process(100,8))
    env.run(until=10000)
 