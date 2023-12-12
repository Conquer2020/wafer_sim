import simpy
from monitored_resource import MonitoredResource as Resource
from typing import List, Union
import random
from util import *
from functools import wraps
import shutil
from ML import *


class Packet:
    def __init__(self, id, shape: List[int], meta_data="test") -> None:
        self.id = id
        self.shape = shape
        self.size = self._size_MB()
        self.meta_data = meta_data

    def _size_MB(self):
        temp = mulc(self.shape)
        return temp / 1000 / 1000

    def __str__(self):
        return "Packet:(id:{},shape:{},size:{} MByte,meta:{})".format(
            self.id, self.shape, self.size, self.meta_data
        )

    @staticmethod
    def random_gen():
        id = random.randint(0, 10000)
        shape = []
        shape_dim = random.randint(1, 2)
        for i in range(shape_dim):
            shape.append(random.randint(1, 128))
        return Packet(id=id, shape=shape)


class DDR_model:
    def __init__(
        self,
        name,
        env,
        transfer_rate_M,
        channel_num,
        die_num,
        per_die_cap_GB,
        bit_width=32,
    ) -> None:
        self.name = name
        self.transfer_rate_M = transfer_rate_M
        self.channel_num = channel_num
        self.die_num = die_num
        self.die_cap_GB = per_die_cap_GB
        self.bit_width = bit_width
        self.capacity = die_num * per_die_cap_GB
        self.env = env
        self.access_resource = Resource(self.env, capacity=1)
        self.bandwidth = transfer_rate_M * bit_width / 8 * channel_num

    def access_process(self, data_size_MB, task_id=1, write=True, DEBUG_MODE=False):
        with self.access_resource.request() as req:
            yield req
            latency = data_size_MB / self.bandwidth
            # latency+=self.write_latency if write else self.read_latency
            yield self.env.timeout(latency)


class dram_model:
    def __init__(
        self,
        name,
        env,
        bw_GB=256,
        capacity_GB=16 * 100,
        read_latency_ms=0,
        write_latency_ms=0,
    ) -> None:
        self.name = name
        self.bw_GB = bw_GB
        self.read_latency = read_latency_ms
        self.write_latency = write_latency_ms

        # TODO consider the dram capacity influence for total ml network
        # @fangjh21.20230602: related to embedding op when ml network is recommoned system like DLRM ,etc.
        self.capacity = capacity_GB
        self.env = env
        self.access_resource = Resource(self.env, capacity=1)

    def access_process(self, data_size_MB, task_id=1, write=True, DEBUG_MODE=False):
        with self.access_resource.request() as req:
            yield req
            latency = data_size_MB / self.bw_GB
            latency += self.write_latency if write else self.read_latency
            yield self.env.timeout(latency)


class Wafer_Device:
    def __init__(
        self,
        env,
        wafer_name="test_wafer",
        tile_intra_shape=[4, 4],
        tile_inter_shape=[2, 2],
        tile_intra_noc_bw_GB=256,
        tile_inter_noc_bw_GB=256 * 0.6,
        tile_dram_bw_GB=12288 / 16 / 8,
        tile_dram_capacity_GB=6 / 16,
        edge_die_dram_bw_GB=256,
        clk_freq_GHz=1,
        with_dram_per_tile=True,
        Analytical=True,
    ) -> None:
        self.wafer_name = wafer_name

        self.tile_intra_shape = tile_intra_shape
        self.tile_inter_shape = tile_inter_shape
        self.tile_intra_noc_bw_GB = tile_intra_noc_bw_GB
        self.tile_inter_noc_bw_GB = tile_inter_noc_bw_GB

        self.with_dram_per_tile = with_dram_per_tile
        self.tile_dram_bw_GB = tile_dram_bw_GB
        self.tile_dram_capacity_GB = tile_dram_capacity_GB

        self.edge_die_dram_bw_GB = edge_die_dram_bw_GB
        self.clk_freq_GHz = clk_freq_GHz
        self.noc_response_latency_ms = 0.001
        self.dram_response_latency_ms = 0.001
        self.route_XY = "X"
        self.device_dist = {}
        self.device()
        # simpy env and resource define @fangjh21.20230602
        self.env = env
        self.Analytical = Analytical
        if not Analytical:
            self.link_resource = []
            self.dram_per_tile_resource = []
            # maybe in real system,there is one dram per die
            self.dram_per_die_resource = []
            self.edge_dram_resource = []
            self.__create_resource()

    def wafer_info(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            print("----------wafer-scale infomation----------")
            print(
                "2D mesh {}:{}x{},{}x{}".format(
                    self.wafer_name,
                    self.tile_inter_shape[0],
                    self.tile_inter_shape[1],
                    self.tile_intra_shape[0],
                    self.tile_intra_shape[1],
                )
            )
            return func(self, *args, **kwargs)

        return wrapper

    def dpos_trans(self, device_id):
        x0 = self.tile_intra_shape[0]
        x1 = self.tile_inter_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        # i=yi+yj*y0+xi*y1*y0+xj*y0*y1*x0
        # print(x0,x1,y0,y1,device_id)
        xx1 = device_id // (y0 * y1 * x0)
        tp = device_id - xx1 * y0 * y1 * x0
        xx0 = tp // (y1 * y0)
        tp = tp - xx0 * y0 * y1
        yy1 = tp // y0
        yy0 = tp % y0
        return [xx0, xx1, yy0, yy1]

    def pos_trans(self, pos_1, pos_2=None):
        # (x1,y1,x0,y0)
        # (x,y)
        pos4p = len(pos_1) == 4
        pos2p = len(pos_1) == 2
        tiles_id = []
        assert pos4p or pos2p
        x0 = self.tile_intra_shape[0]
        x1 = self.tile_inter_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        if pos4p:
            x_pos_1 = x0 * pos_1[0] + pos_1[2]
            y_pos_1 = y0 * pos_1[1] + pos_1[3]
            # print(x_pos_1,y_pos_1)
            if pos_2 != None:
                x_pos_2 = x0 * pos_2[0] + pos_2[2]
                y_pos_2 = y0 * pos_2[1] + pos_2[3]
                # print(x_pos_2,y_pos_2)
                assert x_pos_1 <= x_pos_2 and y_pos_1 <= y_pos_2
                for x in range(x_pos_1, x_pos_2 + 1):
                    for y in range(y_pos_1, y_pos_2 + 1):
                        tiles_id.append(x * y0 * y1 + y)
            else:
                assert x_pos_1 < x1 * x0 and y_pos_1 < y1 * y0
                tiles_id.append(x_pos_1 * y0 * y1 + y_pos_1)
        elif pos2p:
            if pos_2 != None:
                assert pos_1[0] <= pos_2[0] and pos_1[1] <= pos_2[1]
                for x in range(pos_1[0], pos_2[0] + 1):
                    for y in range(pos_1[1], pos_2[1] + 1):
                        tiles_id.append(x * y0 * y1 + y)
            else:
                assert pos_1[0] < x1 * x0 and pos_1[1] < y1 * y0
                tiles_id.append(pos_1[0] * y0 * y1 + pos_1[1])
        return tiles_id

    def device(self):
        x0 = self.tile_intra_shape[0]
        x1 = self.tile_inter_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        if self.device_dist == {}:
            for xj in range(x1):
                for xi in range(x0):
                    for yj in range(y1):
                        for yi in range(y0):
                            z = yj + xj * y1
                            if z not in self.device_dist:
                                self.device_dist[z] = []
                            i = yi + yj * y0 + xi * y1 * y0 + xj * y0 * y1 * x0
                            self.device_dist[z].append(i)
        return [i for i in range(x0 * x1 * y0 * y1)]

    @wafer_info
    def __create_resource(self):
        x0 = self.tile_intra_shape[0]
        x1 = self.tile_inter_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        # here I define the noc link is occupied by only one process until the process release it.
        for _ in range(y0 * y1 - 1):
            for _ in range(x0 * x1):
                self.link_resource.append(Resource(self.env, capacity=1))
        for _ in range(y0 * y1):
            for _ in range(x0 * x1 - 1):
                self.link_resource.append(Resource(self.env, capacity=1))
        print("noc link resource is created...")
        for _ in range(x1):
            self.edge_dram_resource.append(
                dram_model("DDR", self.env, self.edge_die_dram_bw_GB)
            )  # left dram
        for _ in range(x1):
            self.edge_dram_resource.append(
                dram_model("DDR", self.env, self.edge_die_dram_bw_GB)
            )  # right dram
        print("edge dram resource is created...")

        if self.with_dram_per_tile:
            tile_dram_num = x1 * x0 * y1 * y0
            for _ in range(tile_dram_num):
                self.dram_per_tile_resource.append(
                    dram_model(
                        "3DDRAM",
                        self.env,
                        self.tile_dram_bw_GB,
                        self.tile_dram_capacity_GB,
                    )
                )
            print("tile dram resource is created...")

    def Manhattan_hops(self, src_id, dis_id):
        x = self.tile_intra_shape[0] * self.tile_inter_shape[0]
        y = self.tile_inter_shape[1] * self.tile_intra_shape[1]
        min_id = min(src_id, dis_id)
        max_id = max(src_id, dis_id)
        res = max_id - min_id
        if (max_id % y) >= (min_id % y):
            return (res % y) + (res // y)
        else:
            return (min_id % y) - (max_id % y) + (max_id // y)

    def route_gen(self, src_id, des_id, DEBUG_MODE=True):
        x = self.tile_intra_shape[0] * self.tile_inter_shape[0]
        y = self.tile_inter_shape[1] * self.tile_intra_shape[1]
        assert src_id != des_id, "Source and destination IDs must be different"
        assert 0 <= des_id < (x * y), "Destination ID {} out of range".format(des_id)
        route_list = [src_id]
        if self.route_XY == "X":
            while src_id != des_id:
                if (src_id % y) > (des_id % y):
                    src_id -= 1
                elif (src_id % y) < (des_id % y):
                    src_id += 1
                else:
                    src_id += y if src_id < des_id else -y
                route_list.append(src_id)
        else:
            pass
        # if DEBUG_MODE:
        #    print('Router_List:{}'.format(list))
        return route_list

    def link_gen(self, src_id, des_id, DEBUG_MODE=False):
        x0 = self.tile_intra_shape[0]
        x1 = self.tile_inter_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        Y_OFFSET = (y0 * y1 - 1) * x0 * x1
        route_list = self.route_gen(src_id, des_id, DEBUG_MODE=DEBUG_MODE)
        distence = len(route_list)
        link_list = []
        for i in range(distence - 1):
            if abs(route_list[i + 1] - route_list[i]) == 1:
                temp = min(route_list[i], route_list[i + 1])
                t1 = (temp // (y0 * y1)) * (y0 * y1 - 1)
                t2 = temp % (y0 * y1)
                X_INDEX = t1 + t2
                link_list.append(X_INDEX)
            elif abs(route_list[i + 1] - route_list[i]) == y0 * y1:
                Y_INDEX = min(route_list[i], route_list[i + 1])
                link_list.append(Y_OFFSET + Y_INDEX)
            else:
                raise NotImplemented
        return link_list

    def is_inter_link(self, link_id):
        x0 = self.tile_intra_shape[0]
        x1 = self.tile_inter_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        Y_OFFSET = (x0 * x1 - 1) * y0 * y1
        if link_id < Y_OFFSET:
            if (link_id + 1) % x0 == 0:
                return True
            else:
                return False
        else:
            offset_link_id = link_id - Y_OFFSET
            if (int(offset_link_id / (x0 * x1)) + 1) == y0:
                return True
            else:
                return False

    def noc_process(self, comm_size_MB, src_id, des_id, task_id=1, DEBUG_MODE=False):
        assert src_id != des_id, "src_id({})!=des_id({})".format(src_id, des_id)
        ListID = self.link_gen(src_id, des_id, DEBUG_MODE)
        # print(src_id,des_id,ListID)
        first_hop = True
        while True:
            for i in ListID:
                time_ms = self.noc_response_latency_ms
                if self.is_inter_link(i):
                    if first_hop:
                        time_ms += comm_size_MB / self.tile_inter_noc_bw_GB
                else:
                    if first_hop:
                        time_ms += comm_size_MB / self.tile_intra_noc_bw_GB
                if not self.Analytical:
                    with self.link_resource[i].request() as req:
                        yield req
                        yield self.env.timeout(time_ms)
                else:
                    yield self.env.timeout(time_ms)
                first_hop = False
            break

    def edge_dram_write_process(
        self, access_size_MB, src_id, task_id="DDR_READ_TEST", DEBUG_MODE=False
    ):
        # TODO
        x1 = self.tile_inter_shape[0]
        x0 = self.tile_intra_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        y = self.tile_intra_shape[1] * self.tile_inter_shape[1]
        row_line = int(src_id / y) + 1
        des_id = (
            row_line * y - 1
            if (row_line * y - 1 - src_id) < (y / 2)
            else (row_line - 1) * y
        )
        while True:
            # if DEBUG_MODE:
            #    print("task {} start dram wrtie  @ {:.3f} ms".format(task_id,self.envenv.now))
            if des_id != src_id:
                yield self.env.process(
                    self.noc_process(
                        access_size_MB,
                        src_id,
                        des_id,
                        task_id=task_id,
                        DEBUG_MODE=DEBUG_MODE,
                    )
                )
            if not self.Analytical:
                pos = self.dpos_trans(des_id)
                dram_index = 2 * pos[1] - 1 if pos[3] > y1 / 2 else pos[1]
                # print(dram_index)
                yield self.env.process(
                    self.edge_dram_resource[dram_index].access_process(
                        access_size_MB, task_id=task_id, write=True
                    )
                )
            else:
                yield self.env.timeout(
                    self.dram_response_latency_ms
                    + access_size_MB / self.tile_dram_bw_GB
                )
            # if DEBUG_MODE:
            # print("task {} end dram wrtie  @ {:.3f} ms".format(task_id,self.env.now))
            break

    def edge_dram_read_process(
        self, access_size_MB, src_id, task_id="DDR_READ_TEST", DEBUG_MODE=True
    ):
        x1 = self.tile_inter_shape[0]
        x0 = self.tile_intra_shape[0]
        y0 = self.tile_intra_shape[1]
        y1 = self.tile_inter_shape[1]
        y = self.tile_intra_shape[1] * self.tile_inter_shape[1]
        row_line = int(src_id / y) + 1
        des_id = (
            row_line * y - 1
            if (row_line * y - 1 - src_id) < (y / 2)
            else (row_line - 1) * y
        )
        while True:
            # if DEBUG_MODE:
            #    print("task {} start dram read  @ {:.3f} ms".format(task_id,self.env.now))
            pos = self.dpos_trans(des_id)
            # print(pos)
            dram_index = 2 * pos[1] - 1 if pos[3] > y1 / 2 else pos[1]
            # print(dram_index)
            if not self.Analytical:
                yield self.env.process(
                    self.edge_dram_resource[dram_index].access_process(
                        access_size_MB, task_id=task_id, write=False
                    )
                )
            else:
                yield self.env.timeout(
                    self.dram_response_latency_ms
                    + access_size_MB / self.edge_die_dram_bw_GB
                )
            if des_id != src_id:
                yield self.env.process(
                    self.noc_process(
                        access_size_MB,
                        des_id,
                        src_id,
                        task_id=task_id,
                        DEBUG_MODE=DEBUG_MODE,
                    )
                )
            # if DEBUG_MODE:
            #    print("task {} end dram read @ {:.3f} ms".format(task_id,self.env.now))
            break

    def tile_dram_access_process(
        self,
        access_size_MB,
        src_id,
        task_id="3DDRAM-TEST",
        WRITE=True,
        DEBUG_MODE=False,
    ):
        while True:
            assert self.with_dram_per_tile
            if not self.Analytical:
                yield self.env.process(
                    self.dram_per_tile_resource[src_id].access_process(
                        access_size_MB,
                        task_id=task_id,
                        write=WRITE,
                        DEBUG_MODE=DEBUG_MODE,
                    )
                )
            else:
                yield self.env.timeout(
                    self.dram_response_latency_ms
                    + access_size_MB / self.tile_dram_bw_GB
                )
            break

    def tile_dram_group_access_process(
        self,
        access_size_MB,
        group_id: List[int],
        task_id="3DDRAM-TEST",
        WRITE=True,
        DEBUG_MODE=False,
    ):
        for id in group_id:
            yield self.env.process(
                self.tile_dram_access_process(
                    access_size_MB, id, task_id, WRITE, DEBUG_MODE
                )
            )

    def dram_read_group_process(
        self,
        access_size_MB: Union[int, List[int]],
        group_id: List[int],
        task_id,
        multicast=True,
    ):
        # TODO 优化
        if type(access_size_MB) is list:
            temp = mulc(access_size_MB)
            access_size_MB = temp / 1000 / 1000 * 2
            # print(access_size_MB)
        while True:
            # print("task {} start dram_read_group_process @ {:.3f} ms".format(task_id,self.env.now))
            # print(group_id[0])
            yield self.env.process(
                self.edge_dram_read_process(access_size_MB, group_id[0], task_id)
            )
            g_size = len(group_id)
            for i in range(1, g_size):
                comm_size = access_size_MB / g_size if not multicast else access_size_MB
                yield self.env.process(
                    self.noc_process(comm_size, group_id[i - 1], group_id[i], task_id)
                )
            # print("task {} end dram_read_group_process @ {:.3f} ms".format(task_id,self.env.now))
            break

    def dram_write_group_process(
        self,
        access_size_MB: Union[int, List[int]],
        group_id: List[int],
        task_id,
        gather=True,
    ):
        # TODO 优化
        if type(access_size_MB) is list:
            temp = mulc(access_size_MB)
            access_size_MB = temp / 1000 / 1000 * 2
        while True:
            g_size = len(group_id)
            if gather:
                for i in range(g_size - 1, 0, -1):
                    comm_size = access_size_MB / g_size
                    yield self.env.process(
                        self.noc_process(comm_size, group_id[i], group_id[0], task_id)
                    )
            yield self.env.process(
                self.edge_dram_write_process(access_size_MB, group_id[0], task_id)
            )
            break

    def ALL_REDUCE_process(
        self, comm_size, group_id: List[int], task_id, DEBUG_MODE=False
    ):
        # TODO 完成通信原语及其优化
        # yield self.env.timeout(5)
        group_size = len(group_id)
        chunk_size = comm_size / group_size
        # if DEBUG_MODE:
        #        print("ALL_REDUCE task {} start @ {:.3f} ms".format(task_id,self.env.now))
        t_last = self.env.now
        for i in range(group_size - 1):
            event_list = []
            for id_idx in range(group_size - 1):
                event_list.append(
                    self.env.process(
                        self.noc_process(
                            chunk_size, group_id[id_idx], group_id[id_idx + 1]
                        )
                    )
                )
            event_list.append(
                self.env.process(
                    self.noc_process(chunk_size, group_id[-1], group_id[0])
                )
            )
            yield simpy.AllOf(self.env, event_list)
            # if DEBUG_MODE:
            #    print('Reduce-Scatter {}/{} phase'.format(i+1,group_size-1))
        for i in range(group_size - 1):
            event_list = []
            for id_idx in range(group_size - 1):
                event_list.append(
                    self.env.process(
                        self.noc_process(
                            chunk_size, group_id[id_idx], group_id[id_idx + 1]
                        )
                    )
                )
            event_list.append(
                self.env.process(
                    self.noc_process(chunk_size, group_id[-1], group_id[0])
                )
            )
            yield simpy.AllOf(self.env, event_list)
            # if DEBUG_MODE:
            #    print('All-Gather {}/{} phase'.format(i+1,group_size-1))
        # if DEBUG_MODE:
        #    print("ALL_REDUCE task {} end @ {:.3f} ms".format(task_id,self.env.now))
        print(
            "ALL_REDUCE task {} end with {:.3f} ms".format(
                task_id, self.env.now - t_last
            )
        )

    def ALL_2_ALL_process(
        self, comm_size, group_id: List[int], task_id, DEBUG_MODE=False
    ):
        # TODO 完成通信原语及其优化
        group_size = len(group_id)
        # print(group_size)
        chunk_size = comm_size / group_size
        t_last = self.env.now
        for i in range(group_size - 1):
            event_list = []
            for id_idx in range(group_size):
                des_id = (id_idx + i + 1) % group_size
                event_list.append(
                    self.env.process(
                        self.noc_process(chunk_size, group_id[id_idx], group_id[des_id])
                    )
                )
            yield simpy.AllOf(self.env, event_list)
        print(
            "ALL_2_ALL task {} end with {:.3f} ms".format(
                task_id, self.env.now - t_last
            )
        )

    def STAGE_PASS_process(
        self,
        comm_size: Union[int, Packet],
        group_a: List[int],
        group_b: List[int],
        task_id,
        DEBUG_MODE=False,
    ):
        # TODO 完成通信原语
        if type(comm_size) is Packet:
            comm_size = comm_size.size
        distance = []
        for i in group_a:
            for j in group_b:
                distance.append(self.Manhattan_hops(i, j))
        index = distance.index(min(distance))
        src = group_a[index // len(group_b)]
        des = group_b[index % len(group_b)]
        # if DEBUG_MODE:
        # print('Group_A {} to Group_B stage pass {}'.format(src,des))
        while True:
            # if DEBUG_MODE:
            #    print("STAGE_PASS task {} start @ {:.3f} ms".format(task_id,self.env.now))
            for i in group_a:
                if i != src:
                    yield self.env.process(
                        self.noc_process(comm_size, i, src, task_id, DEBUG_MODE)
                    )
            all_comm_size = comm_size * len(group_a)
            yield self.env.process(
                self.noc_process(all_comm_size, src, des, task_id, DEBUG_MODE)
            )
            for j in group_b:
                if j != des:
                    yield self.env.process(
                        self.noc_process(
                            all_comm_size / len(group_b), des, j, task_id, DEBUG_MODE
                        )
                    )
            # if DEBUG_MODE:
            #    print("STAGE_PASS task {} start @ {:.3f} ms".format(task_id,self.env.now))
            break

    def resource_visualize(
        self, res_type: str = "edge_dram", path="./status/resource/", clear=True
    ):
        if clear:
            ls = os.listdir(path)
            for i in ls:
                f_path = os.path.join(path, i)
                # print(f_path)
                shutil.rmtree(f_path)
        if res_type == "all":
            for index, res in enumerate(self.edge_dram_resource):
                visualize_resource(
                    res.access_resource.data,
                    path + "edge_dram",
                    str(index),
                    max_resource=self.edge_die_dram_bw_GB,
                )
            for index, res in enumerate(self.dram_per_tile_resource):
                visualize_resource(
                    res.access_resource.data,
                    path + "3ddram",
                    str(index),
                    max_resource=self.tile_dram_bw_GB,
                )
            path1 = path + "inter_noc"
            path2 = path + "intra_noc"
            for index, res in enumerate(self.link_resource):
                if self.is_inter_link(index):
                    visualize_resource(
                        res.data,
                        path1,
                        str(index),
                        max_resource=self.tile_inter_noc_bw_GB,
                    )
                else:
                    visualize_resource(
                        res.data,
                        path2,
                        str(index),
                        max_resource=self.tile_intra_noc_bw_GB,
                    )
        elif res_type == "edge_dram":
            for index, res in enumerate(self.edge_dram_resource):
                visualize_resource(
                    res.access_resource.data,
                    path + "edge_dram",
                    str(index),
                    max_resource=self.edge_die_dram_bw_GB,
                )
        elif res_type == "3ddram":
            for index, res in enumerate(self.dram_per_tile_resource):
                visualize_resource(
                    res.access_resource.data,
                    path + "3ddram",
                    str(index),
                    max_resource=self.tile_dram_bw_GB,
                )
        elif res_type == "noc":
            path1 = path + "inter_noc"
            path2 = path + "intra_noc"
            for index, res in enumerate(self.link_resource):
                if self.is_inter_link(index):
                    visualize_resource(
                        res.data,
                        path1,
                        str(index),
                        max_resource=self.tile_inter_noc_bw_GB,
                    )
                else:
                    visualize_resource(
                        res.data,
                        path2,
                        str(index),
                        max_resource=self.tile_intra_noc_bw_GB,
                    )
        else:
            raise NotImplementedError


def validate_allreduce():
    # NOTE: below is 4 NPUs
    # env.process(wd.tile_dram_access_process(0,63,'TEST_3DDRAM',DEBUG_MODE=Debug))
    comm_sizes = [64, 96, 128, 192, 768, 1536]
    print(f"&&&&&&&&&&&&&&&--4 GPUs--&&&&&&&&&&&&&&&&&")
    for comm_size in comm_sizes:
        Debug = True
        env = simpy.Environment()
        wd = Wafer_Device(
            env,
            tile_inter_shape=[1, 1],
            tile_intra_shape=[2, 2],
            tile_intra_noc_bw_GB=150,
            tile_inter_noc_bw_GB=120,
            with_dram_per_tile=True,
            Analytical=True,
        )
        env.process(
            wd.ALL_REDUCE_process(
                comm_size=comm_size, group_id=[0, 1, 3, 2], task_id="ALL_REDUCE_process"
            )
        )
        env.run(until=10000)

    print(f"&&&&&&&&&&&&&&&--16 GPUs--&&&&&&&&&&&&&&&&&")
    # NOTE: below is 16 NPUs
    comm_sizes = [64, 96, 128, 192, 768, 1536]
    for comm_size in comm_sizes:
        Debug = True
        env = simpy.Environment()
        wd = Wafer_Device(
            env,
            tile_inter_shape=[1, 2],
            tile_intra_shape=[2, 4],
            tile_intra_noc_bw_GB=150,
            tile_inter_noc_bw_GB=120,
            with_dram_per_tile=True,
            Analytical=True,
        )
        env.process(
            wd.ALL_REDUCE_process(
                comm_size=comm_size,
                group_id=[0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8],
                task_id="ALL_REDUCE_process",
            )
        )
        env.run(until=10000)


def validate_congestion():
    # NOTE: below is 16 NPUs
    # comm_sizes = [64, 96, 128, 192, 768, 1536]
    print(f"$$$$$$$$$$$$$$$$Analytical Model$$$$$$$$$$$$$$$$")
    comm_sizes = [128]
    for comm_size in comm_sizes:
        Debug = True
        env = simpy.Environment()
        wd = Wafer_Device(
            env,
            tile_inter_shape=[1, 2],
            tile_intra_shape=[2, 4],
            tile_intra_noc_bw_GB=150,
            tile_inter_noc_bw_GB=120,
            with_dram_per_tile=True,
            Analytical=True,
        )
        env.process(
            wd.ALL_REDUCE_process(
                comm_size=comm_size,
                group_id=[0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8],
                task_id="ALL_REDUCE_process",
            )
        )
        env.process(
            wd.ALL_2_ALL_process(
                comm_size=comm_size,
                group_id=[0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8],
                task_id="ALL_REDUCE_process",
            )
        )
        env.run(until=10000)

    # NOTE: below is 16 NPUs
    # comm_sizes = [64, 96, 128, 192, 768, 1536]
    print(f"$$$$$$$$$$$$$$$$Simulator Model$$$$$$$$$$$$$$$$")
    comm_sizes = [128]
    for comm_size in comm_sizes:
        Debug = True
        env = simpy.Environment()
        wd = Wafer_Device(
            env,
            tile_inter_shape=[1, 2],
            tile_intra_shape=[2, 4],
            tile_intra_noc_bw_GB=150,
            tile_inter_noc_bw_GB=120,
            with_dram_per_tile=True,
            Analytical=False,
        )
        env.process(
            wd.ALL_REDUCE_process(
                comm_size=comm_size,
                group_id=[0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8],
                task_id="ALL_REDUCE_process",
            )
        )
        env.run(until=10000)


if __name__ == "__main__":
    # Debug=True
    # env = simpy.Environment()
    # wd=Wafer_Device(env,tile_inter_shape=[2,2],tile_intra_shape=[1,1],tile_intra_noc_bw_GB=150,tile_inter_noc_bw_GB=150*0.8,with_dram_per_tile=True,Analytical=True)
    """
    env.process(wd.noc_process(10,src_id=0,des_id=3,task_id=1,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=3,des_id=0,task_id=2,DEBUG_MODE=Debug))
    env.process(wd.edge_dram_read_process(10,src_id=1,DEBUG_MODE=Debug))
    env.process(wd.edge_dram_read_process(10,src_id=1,task_id=4,DEBUG_MODE=Debug))
    env.process(wd.edge_dram_write_process(16,src_id=1,task_id=5,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=13,des_id=15,task_id=6,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=13,des_id=15,task_id=7,DEBUG_MODE=Debug))
    env.process(wd.noc_process(10,src_id=13,des_id=15,task_id=8,DEBUG_MODE=Debug))
    env.process(wd.STAGE_PASS_process(10,[0,1,2,3,5],[8,9],'TEST'))
    """
    validate_allreduce()
    # validate_congestion()

    # # NOTE: below is 4 NPUs
    # # env.process(wd.tile_dram_access_process(0,63,'TEST_3DDRAM',DEBUG_MODE=Debug))
    # comm_sizes = [64, 96, 128, 192, 768, 1536]
    # for comm_size in comm_sizes:
    #     Debug = True
    #     env = simpy.Environment()
    #     wd = Wafer_Device(
    #         env,
    #         tile_inter_shape=[2, 2],
    #         tile_intra_shape=[1, 1],
    #         tile_intra_noc_bw_GB=150,
    #         tile_inter_noc_bw_GB=120,
    #         with_dram_per_tile=True,
    #         Analytical=True,
    #     )
    #     env.process(
    #         wd.ALL_REDUCE_process(
    #             comm_size=comm_size, group_id=[0, 1, 3, 2], task_id="ALL_REDUCE_process"
    #         )
    #     )
    #     env.run(until=10000)

    # # NOTE: below is 16 NPUs
    # comm_sizes = [64, 96, 128, 192, 768, 1536]
    # for comm_size in comm_sizes:
    #     Debug = True
    #     env = simpy.Environment()
    #     wd = Wafer_Device(
    #         env,
    #         tile_inter_shape=[1, 2],
    #         tile_intra_shape=[2, 4],
    #         tile_intra_noc_bw_GB=150,
    #         tile_inter_noc_bw_GB=120,
    #         with_dram_per_tile=True,
    #         Analytical=True,
    #     )
    #     env.process(
    #         wd.ALL_REDUCE_process(
    #             comm_size=comm_size,
    #             group_id=[0, 1, 2, 3,4, 5,6, 7, 15,14,13, 12, 11, 10,9, 8],
    #             task_id="ALL_REDUCE_process",
    #         )
    #     )
    #     env.run(until=10000)
