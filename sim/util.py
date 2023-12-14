import matplotlib.pyplot as plt
import os
from typing import List
from queue import Queue
from operator import mul
from functools import reduce
from ML import *

MY_COLOR = [
    "#63b2ee",
    "#76da91",
    "#f8cb7f",
    "#f89588",
    "#7cd6cf",
    "#9192ab",
    "#7898e1",
    "#efa666",
    "#eddd86",
    "#9987ce",
    "#63b2ee",
    "#76da91",
]  # maktalong


def mulc(a: List[int]):
    return reduce(mul, a)


def shape_suppose(size):
    R = 0
    C = 0
    if size == 8000 or size == 8192:
        R = 128
        C = 64
    if size == 4000 or size == 4096:
        R = 64
        C = 64
    if size == 1000 or size == 1024:
        R = 32
        C = 32
    return [R, C]


def str2list(string):
    ls = []
    string = string.split("[")[1].split("]")[0]
    string = string.split(",")

    for num_str in string:
        # print(int(num_str))
        num_str.split(",")
        ls.append(int(num_str))
    return ls


def str2strlist(string):
    ls = []
    string = string.split("[")[1].split("]")[0]
    string = string.split(",")
    for str_ in string:
        if str_ != "":
            str_str = str_.split("'")
            ls.append(str_str[1])
    return ls


def split_comm_group(Group_Id, parall_dims):
    """
    Here is an example :
    suppose Group_Id=[0,1,2,3,...,15],len=16
    1.if parall_dims=[16,1,1,1],group=[[0:15],[],[],[]]
    2.if parall_dims=[1,16,1,1],group=[[],[0:15],[],[]]
    3.if parall_dims=[8,2,1,1],group=
    [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    []
    []
    """
    Group_Size = len(Group_Id)
    total_dims = 1
    split_group = []
    for dim in parall_dims:
        split_group.append(total_dims)
        total_dims *= dim
    assert Group_Size == total_dims, "Group_Size={},but total_dims={} ".format(
        Group_Size, total_dims
    )
    num_dims = len(parall_dims)
    groups = []
    offset = Group_Size
    # print(split_group)
    for k in range(num_dims):
        temp_group_size = parall_dims[k]
        # print(temp_group_size)
        temp_group = []
        if temp_group_size != 1:
            offset //= parall_dims[k]
            # print("offset",offset)
            for j in range(split_group[k]):
                # print(k,offset,j)
                for i in range(offset):
                    # print(i+j*(Group_Size//split_group[k]),(j+1)*Group_Size//split_group[k],offset)
                    temp_group.append(
                        Group_Id[
                            i
                            + j
                            * (Group_Size // split_group[k]) : (j + 1)
                            * Group_Size
                            // split_group[k] : offset
                        ]
                    )
        groups.append(temp_group)
    return groups


def draw_pipeline(trace_list, path, title, throughout, name="pipeline"):
    # print(trace_list)
    # [[(s,e),(s,e),(s,e)],[],[]], []=stages,s=micro_start_time,e=micro_end_time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # COLOR = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#10c020', '#D20F99','#FFFFFF','#000000']
    # COLOR=['#63b2ee','#76da91','#f8cb7f','#f89588','#7cd6cf','#9192ab','#7898e1','#efa666','#eddd86','#9987ce','#63b2ee','#76da91','#100000']#maktalong
    num = len(trace_list)
    leng = len(trace_list[0])
    width_scale = 0
    for trace in trace_list:
        if trace[-1][1] > width_scale:
            width_scale = trace[-1][1]
    # width_scale=max(trace_list[0][-1][1],trace_list[-1][-1][1])
    height_scale = 4
    single_height = 1
    start_height = single_height / 2
    for j in range(num):
        k = 0
        m = 0
        leng = len(trace_list[j])
        for i in range(leng):
            x = trace_list[j][i]
            # facecolor=color[0] if x[2]==0 else color[5]
            if x[2] == ML_STATE.FORWARD:
                facecolor = MY_COLOR[k % len(MY_COLOR)]
                k += 1
            elif x[2] == ML_STATE.BACKWARD:
                facecolor = MY_COLOR[m % len(MY_COLOR)]
                m += 1
            else:
                facecolor = MY_COLOR[0]
            edgecolor = MY_COLOR[-1]
            rect = plt.Rectangle(
                (x[0], start_height + single_height * j),
                (x[1] - x[0]),
                single_height,
                fill=True,
                facecolor=facecolor,
                linewidth=0.5,
            )
            ax.add_patch(rect)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title(
        "{} stages ML {} pipeline [{:.3f} sample/s]".format(num, title, throughout)
    )
    ax.set_ylim(0, num + 1)
    ax.set_yticks([num - i for i in range(num)])
    ax.set_xlim(0, width_scale)
    # ax.set_aspect(20)
    plt.xlabel("Time")
    plt.ylabel("Stage")
    plt.savefig(os.path.join(path, name + ".png"))


def data_average(data, ave_size=1000):
    # data type[[start_time,end_time,max_resource],...]
    assert ave_size > 1
    pass


def max_ave_1F_1B_time(trace, train):
    max_1F1B_time = 0
    tp_time = 0
    stage_num = len(trace)
    fb_num = len(trace[0])
    print(stage_num, fb_num)
    for i in range(stage_num):
        tp_time = 0
        fb_num = len(trace[i])
        for j in range(fb_num):
            tp_time += trace[i][j][1] - trace[i][j][0]
        if train:
            tp_time /= fb_num / 2
        else:
            tp_time /= fb_num
        # print(i,tp_time)
        if max_1F1B_time < tp_time:
            max_1F1B_time = tp_time
            # print('max_index=',i)
    return max_1F1B_time


def visualize_resource(
    data: List, path, name, clear_redundance=True, max_resource=256, ave_unit_ms=1
):
    # [(req_flag,req_time,len_q),(res_flag,res_time,len_q)]
    # print(data)
    if data == []:
        return None
    q_req = Queue()
    occupy_list = []
    for item in data:
        if item[0] == "req":
            q_req.put(item)
        elif item[0] == "res":
            req_item = q_req.get()
            res_item = item
            occupy_list.append([req_item[1], res_item[1], max_resource])
    if clear_redundance:
        leng = len(occupy_list)
        del_list = []
        for i in range(1, leng):
            if occupy_list[i][0] == occupy_list[i - 1][0]:
                del_list.append(i - 1)
            elif occupy_list[i][0] < occupy_list[i - 1][1]:
                del_list.append(i - 1)
                occupy_list[i][0] = occupy_list[i - 1][0]
        new_list = []
        for i in range(leng):
            if i not in del_list:
                new_list.append(occupy_list[i])
    # print(data_list)
    if ave_unit_ms != 1:
        list_ave = []
        occupy_time = 0
        time = ave_unit_ms
        for data in new_list:
            if data[1] < time and data != new_list[-1]:
                occupy_time += data[1] - data[0]
            else:
                if data == new_list[-1]:
                    occupy_time += data[1] - data[0]
                ave_resource = max_resource * occupy_time / ave_unit_ms
                list_ave.append((time - ave_unit_ms, time, ave_resource))
                time += ave_resource
                occupy_time = data[1] - data[0]
    data_list = list_ave if ave_unit_ms != 1 else new_list
    # [(start_time,end_time,resource_occupy)]
    # print(data_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    data0 = 0
    for data in data_list:
        # plt.plot([data[0],data[1]],[data[2],data[2]],color='r',linewidth=2)
        plt.scatter(data[0], data[2], color="b")
        plt.scatter(data[1], data[2], color="r")
        # print(data[1],data[0])
        if data[0] > data0:
            plt.plot([data0, data[0]], [0, 0], color="black", linewidth=1)
            data0 = data[0]
    plt.xlabel("Time(ms)")
    plt.ylabel("Bandwidth(GB/s)")
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, name + ".png"))
    plt.close()
    return data_list


def draw_mapping(wd, ml_name, tiles=[], path="status", ori=False):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    x0 = wd.tile_intra_shape[0]
    x1 = wd.tile_inter_shape[0]
    y0 = wd.tile_intra_shape[1]
    y1 = wd.tile_inter_shape[1]
    max_s = max(y0 * y1, x0 * x1)
    core_weight = 0.8
    core_high = 0.8

    # print(x1,x0)
    for xj in range(x1):
        for yj in range(y1):
            facecolor = MY_COLOR[(yj + xj * y1) % len(MY_COLOR)]
            for xi in range(x0):
                for yi in range(y0):
                    xx = xi + xj * x0 - 0.4
                    yy = (yi + yj * y0) - 0.4
                    rect = plt.Rectangle(
                        (yy, xx),
                        core_weight,
                        core_high,
                        fill=ori,
                        facecolor=facecolor,
                        linewidth=0.1,
                    )
                    ax.add_patch(rect)
    # yi+yj*y0+xi*y1*y0+xj*y0*y1*x0
    if tiles != []:
        for ids, tile in enumerate(tiles):
            for id in tile:
                xj = id // (y0 * y1 * x0)
                id -= xj * y0 * y1 * x0
                xi = id // (y1 * y0)
                id -= xi * y1 * y0
                yj = id // y0
                id -= yj * y0
                yi = id
                # print(xi,yi,xj,yj)
                xx = xi + xj * x0 - 0.4
                yy = (yi + yj * y0) - 0.4
                # print(yy,xx)
                plt.text(
                    x=yy + 0.4,
                    y=xx + 0.5,
                    s=ids,
                    rotation=1,
                    ha="center",
                )
                facecolor = MY_COLOR[ids % len(MY_COLOR)]
                rect = plt.Rectangle(
                    (yy, xx),
                    core_weight,
                    core_high,
                    fill=not ori,
                    facecolor=facecolor,
                    linewidth=0.1,
                )
                ax.add_patch(rect)
    ax.set_ylim(y0 * y1 - 1, 0)
    ax.set_xlim(0, x0 * x1 - 1)
    # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部,默认是none
    ax.xaxis.set_ticks_position("top")
    ax.yaxis.set_ticks_position("left")
    name = ml_name + "_map"
    plt.axis("equal")
    plt.xlabel("x_direct")
    plt.ylabel("y_scale")
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, name + ".png"))
    plt.close()
    # plt.show()
