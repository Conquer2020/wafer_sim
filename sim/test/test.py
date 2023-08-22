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
    print(split_group)
    for k in range(num_dims):
        temp_group_size=parall_dims[k]
        #print(temp_group_size)
        temp_group=[]
        if temp_group_size!=1:
            offset//=parall_dims[k]
            print("offset",offset)
            for j in range(split_group[k]):
                #print(k,offset,j)
                for i in range(offset):
                    print(i+j*(Group_Size//split_group[k]),(j+1)*Group_Size//split_group[k],offset)
                    temp_group.append(Group_Id[i+j*(Group_Size//split_group[k]):(j+1)*Group_Size//split_group[k]:offset])
        groups.append(temp_group)
    return groups
L=[x for x in range(16)]
parall_dims=[8,2,1,1]#[2,2,2,2]#[8,2,1,1]#[16,1,1,1]#
groups=split_comm_group(L,parall_dims)
for temp_group in groups:
    print(temp_group)
