import simpy
from monitored_resource import MonitoredResource as Resource
from typing import List,Union
import random
from util import *

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
    
class dram_model():
    def __init__(self,name,env,bw_GB=256,capacity_GB=16*100,read_latency=0,write_latency=0) -> None:
        self.name=name
        self.bw_GB=bw_GB
        self.read_latency=read_latency
        self.write_latency=write_latency
        
        #TODO consider the dram capacity influence for total ml network
        #@fangjh21.20230602: related to embedding op when ml network is recommoned system like DLRM ,etc.
        self.capacity=capacity_GB 
        self.env=env
        self.access_resource=Resource(self.env,capacity=1)
    def access_process(self,data_size_MB,task_id=1,write=True,DEBUG_MODE=True):
        with self.access_resource.request() as req:
            yield req 
            if DEBUG_MODE:
                print("{}:{} access  processing...  @ {:.3f} us".format(task_id,self.name,self.env.now))
            latency=1000*data_size_MB/self.bw_GB
            latency+=self.write_latency if write else self.read_latency
            yield self.env.timeout(latency)

class Wafer_Device():
    def __init__(self,env,wafer_name='fangjh21.20230331',tile_intra_shape=[4,4],tile_inter_shape=[2,2],\
                    tile_intra_noc_bw_GB=256,tile_inter_noc_bw_GB=256*0.6,\
                    tile_dram_bw_GB=12288/16/8,tile_dram_capacity_GB=6/16,
                        edge_die_dram_bw_GB=256,clk_freq_Ghz=1,with_3ddram_per_tile=True) -> None:
        #@3ddram data from wanghuizheng
        self.wafer_name=wafer_name

        self.tile_intra_shape=tile_intra_shape
        self.tile_inter_shape=tile_inter_shape
        self.tile_intra_noc_bw_GB=tile_intra_noc_bw_GB
        self.tile_inter_noc_bw_GB=tile_inter_noc_bw_GB

        self.with_3ddram_per_tile=with_3ddram_per_tile
        self.tile_dram_bw_GB=tile_dram_bw_GB
        self.tile_dram_capacity_GB=tile_dram_capacity_GB


        self.edge_die_dram_bw_GB=edge_die_dram_bw_GB
        self.clk_freq_Ghz=clk_freq_Ghz
        self.noc_response_latency=0
        self.dram_response_latency=0
        self.route_XY='X'

        #simpy env and resource define @fangjh21.20230602
        self.env=env
        self.link_resource=[]
        self.dram_per_tile_resource=[]
        #maybe in real system,there is one dram per die 
        self.dram_per_die_resource=[]
        self.edge_dram_resource=[]
        self.__create_resource()

    def device_list(self):
        x0=self.tile_intra_shape[0]
        x1=self.tile_inter_shape[0]
        y0=self.tile_intra_shape[1]
        y1=self.tile_inter_shape[1]
        return [i for i in range(x1*x0*y1*y0)]
    def __create_resource(self):
        
        x0=self.tile_intra_shape[0]
        x1=self.tile_inter_shape[0]
        y0=self.tile_intra_shape[1]
        y1=self.tile_inter_shape[1]
        #here I define the noc link is occupied by only one process until the process release it.
        for _ in range(y0*y1-1):
            for _ in range(x0*x1):
                self.link_resource.append(Resource(self.env,capacity=1))
        for _ in range(y0*y1):
            for _ in range(x0*x1-1):
                self.link_resource.append(Resource(self.env,capacity=1))
        print('noc link resource is created...')
        
        for _ in range(x1):
            self.edge_dram_resource.append(dram_model('DDR',self.env,self.edge_die_dram_bw_GB))#left dram
        for _ in range(x1):
            self.edge_dram_resource.append(dram_model('DDR',self.env,self.edge_die_dram_bw_GB))#right dram
        print('edge dram resource is created...')

        if self.with_3ddram_per_tile:
            tile_dram_num=x1*x0*y1*y0
            for _ in range(tile_dram_num):
                self.dram_per_tile_resource.append(dram_model('3DDRAM',self.env,self.tile_dram_bw_GB,self.tile_dram_capacity_GB))
            print('tile dram resource is created...')



    def Manhattan_hops(self,src_id,dis_id):
        x=self.tile_intra_shape[0]*self.tile_inter_shape[0]
        y=self.tile_inter_shape[1]*self.tile_intra_shape[1]
        min_id=min(src_id,dis_id)
        max_id=max(src_id,dis_id)
        res=max_id-min_id
        if (max_id% y)>=(min_id % y):
            return (res%y)+(res//y)
        else:
            return (min_id% y)-(max_id % y)+(max_id//y)


    def route_gen(self,src_id,des_id,DEBUG_MODE=True):
        x=self.tile_intra_shape[0]*self.tile_inter_shape[0]
        y=self.tile_inter_shape[1]*self.tile_intra_shape[1]
        assert (src_id!=des_id)
        assert (0<=des_id and des_id<(x*y))
        list=[src_id]
        if self.route_XY=='X':
            while(src_id!=des_id):
                if((src_id % y) >(des_id % y)):
                    src_id-=1
                    list.append(src_id)
                elif ((src_id % y) <(des_id % y)):
                    src_id+=1
                    list.append(src_id)
                else:
                    if(src_id<des_id):
                        src_id+=y
                    else:
                        src_id-=y
                    list.append(src_id)
        else:
            pass
        if DEBUG_MODE:
            print('Router_List:{}'.format(list))
        return list
    def link_gen(self,src_id,des_id,DEBUG_MODE=False):
        x0=self.tile_intra_shape[0]
        x1=self.tile_inter_shape[0]
        y0=self.tile_intra_shape[1]
        y1=self.tile_inter_shape[1]
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
        x0=self.tile_intra_shape[0]
        x1=self.tile_inter_shape[0]
        y0=self.tile_intra_shape[1]
        y1=self.tile_inter_shape[1]
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
        if DEBUG_MODE:
            print('Link_id_List:{}'.format(ListID))
        while(True):
            if DEBUG_MODE:
                print("task {} start noc @ {:.3f} us".format(task_id,self.env.now))
            for i in ListID:
                #print(i)
                with self.link_resource[i].request() as req: 
                    yield req
                    if self.is_inter_link(i):
                        if DEBUG_MODE:
                            print('task {} cross the inter noc, link_id={}'.format(task_id,i))
                        yield self.env.timeout(self.noc_response_latency+1000*comm_size_MB/self.tile_inter_noc_bw_GB)
                    else:
                        yield self.env.timeout(self.noc_response_latency+1000*comm_size_MB/self.tile_intra_noc_bw_GB)
            if DEBUG_MODE:
                print("task {} end noc @ {:.3f} us".format(task_id,self.env.now))
            break
    def edge_dram_write_process(self,access_size_MB,src_id,task_id='DDR_READ_TEST',DEBUG_MODE=False):
        #TODO 此处访存id存在错误 @fangjh21.20230602
        x0=self.tile_intra_shape[0]
        x1=self.tile_inter_shape[0]
        #y0=self.tile_intra_shape[1]
        #y1=self.tile_inter_shape[1]
        row_line=int(src_id /(x0*x1))+1
        des_id=row_line*x0*x1-1 if row_line*x0*x1-1-src_id<x0*x1/2 else (row_line-1)*x0*x1
        while(True):
            if DEBUG_MODE:
                print("task {} start dram wrtie  @ {:.3f} us".format(task_id,self.envenv.now))
            if des_id!=src_id:
                yield self.env.process(self.noc_process(access_size_MB,src_id,des_id,task_id=task_id,DEBUG_MODE=DEBUG_MODE))
            dram_index=int(des_id/ (x1*x0))if des_id %( x1*x0) ==0 else int(des_id/ (x1*x0))+x1*x0
            yield self.env.process(self.edge_dram_resource[dram_index].access_process(access_size_MB,task_id=task_id,write=True))
            if DEBUG_MODE:
                print("task {} end dram wrtie  @ {:.3f} us".format(task_id,self.env.now))
            break
    def edge_dram_read_process(self,access_size_MB,src_id,task_id='DDR_READ_TEST',DEBUG_MODE=False):
        x0=self.tile_intra_shape[0]
        x1=self.tile_inter_shape[0]
        row_line=int(src_id /(x0*x1))+1
        des_id=row_line*x0*x1-1 if row_line*x0*x1-1-src_id<x0*x1/2 else (row_line-1)*x0*x1
        while(True):
            if DEBUG_MODE:
                print("task {} start dram read  @ {:.3f} us".format(task_id,self.env.now))
            dram_index=int(des_id/(x1*x0)) if des_id % (x1*x0) ==0 else int(des_id/ x1*x0)+x1*x0
            yield self.env.process(self.edge_dram_resource[dram_index].access_process(access_size_MB,task_id=task_id,write=False))
            if des_id!=src_id:
                yield self.env.process(self.noc_process(access_size_MB,des_id,src_id,task_id=task_id,DEBUG_MODE=DEBUG_MODE))
            if DEBUG_MODE:
                print("task {} end dram read @ {:.3f} us".format(task_id,self.env.now))
            break
    def tile_dram_access_process(self,access_size_MB,src_id,task_id='3DDRAM-TEST',WRITE=True,DEBUG_MODE=False):
        while(True):
            assert(self.with_3ddram_per_tile)
            yield self.env.process(self.dram_per_tile_resource[src_id].access_process\
                                   (access_size_MB,task_id=task_id,write=WRITE,DEBUG_MODE=DEBUG_MODE))
            break
    def tile_dram_group_access_process(self,access_size_MB,group_id:List[int],task_id='3DDRAM-TEST',WRITE=True,DEBUG_MODE=False):
        for id in group_id:
            yield self.env.process(self.tile_dram_access_process(access_size_MB,id,task_id,WRITE,DEBUG_MODE))

    def dram_read_group_process(self,access_size_MB:Union[int,List[int]],group_id:List[int],task_id,multicast=True):
        #TODO 优化
        if type(access_size_MB) is list:
            temp=mulc(access_size_MB)
            access_size_MB=temp/1000/1000   
        while(True):
            yield self.env.process(self.edge_dram_read_process(access_size_MB,group_id[0],task_id))
            g_size=len(group_id)  
            for i in range(1,g_size):
                comm_size=access_size_MB/g_size if not multicast else access_size_MB
                yield self.env.process(\
                   self.noc_process(comm_size,group_id[i-1],group_id[i],task_id))
            break
    def dram_write_group_process(self,access_size_MB:Union[int,List[int]],group_id:List[int],task_id,gather=True):
        #TODO 优化
        if type(access_size_MB) is list:
            temp=mulc(access_size_MB)
            access_size_MB=temp/1000/1000 
        while(True):
            g_size=len(group_id)
            if gather:  
                for i in range(g_size-1,0,-1):
                    comm_size=access_size_MB/g_size
                    yield self.env.process(\
                        self.noc_process(comm_size,group_id[i],group_id[0],task_id))
            yield self.env.process(self.edge_dram_write_process(access_size_MB,group_id[0],task_id))
            break
    def visualize_resource(self,res_data:List,name,res_type='edge_dram',path='./pic/'):
        max_resource=None
        if res_type=='edge_dram':
            max_resource=self.edge_die_dram_bw_GB
        if res_type=='3dram':
            max_resource=self.tile_dram_bw_GB
        elif res_type=='inter_noc':
            max_resource=self.tile_inter_noc_bw_GB
        elif res_type=='intra_noc':
            max_resource=self.tile_intra_noc_bw_GB
        else :
            return NotImplemented
        visualize_resource(res_data.data,name=path+res_type+name,max_resource=max_resource)

    def ALL_REDUCE_process(self,comm_size,group_id:List[int],task_id,DEBUG_MODE=True):
        # TODO 完成通信原语及其优化
        #yield self.env.timeout(5)
        group_size=len(group_id)
        chunk_size=comm_size/group_size
        if DEBUG_MODE:
                print("ALL_REDUCE task {} start @ {:.3f} us".format(task_id,self.env.now))
        for i in range(group_size-1):
            event_list=[]
            for id_idx in range(group_size-1):
                event_list.append(self.env.process(self.noc_process(chunk_size,group_id[id_idx],group_id[id_idx+1])))
            event_list.append(self.env.process(self.noc_process(chunk_size,group_id[-1],group_id[0])))
            yield simpy.AllOf(self.env, event_list)
            if DEBUG_MODE:
                print('Reduce-Scatter {}/{} phase'.format(i+1,group_size-1))
        for i in range(group_size-1):
            event_list=[]
            for id_idx in range(group_size-1):
                event_list.append(self.env.process(self.noc_process(chunk_size,group_id[id_idx],group_id[id_idx+1])))
            event_list.append(self.env.process(self.noc_process(chunk_size,group_id[-1],group_id[0])))
            yield simpy.AllOf(self.env, event_list)
            if DEBUG_MODE:
                print('All-Gather {}/{} phase'.format(i+1,group_size-1))
        if DEBUG_MODE:
                print("ALL_REDUCE task {} end @ {:.3f} us".format(task_id,self.env.now))
    def ALL_2_ALL_process(self,comm_size,group_id:List[int]):
        # TODO 完成通信原语
        yield self.env.timeout(5)
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
        if DEBUG_MODE:
            print('Group_A {} to Group_B stage pass {}'.format(src,des))
        while(True):
            if DEBUG_MODE:
                print("STAGE_PASS task {} start @ {:.3f} us".format(task_id,self.env.now))
            for i in group_a:
                if i!=src:
                    yield self.env.process(self.noc_process(comm_size/len(group_a),i,src,task_id,DEBUG_MODE))
            yield self.env.process(self.noc_process(comm_size,src,des,task_id,DEBUG_MODE))
            for j in group_b:
                if j!=des:
                    yield self.env.process(self.noc_process(comm_size/len(group_b),des,j,task_id,DEBUG_MODE))
            if DEBUG_MODE:
                print("STAGE_PASS task {} start @ {:.3f} us".format(task_id,self.env.now))
            break


if __name__ == '__main__':
    Debug=True
    env = simpy.Environment()
    wd=Wafer_Device(env,tile_inter_shape=[4,4],tile_intra_shape=[4,4],with_3ddram_per_tile=True)
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
    env.process(wd.tile_dram_access_process(0,63,'TEST_3DDRAM',DEBUG_MODE=Debug))
    env.run(until=10000)
 