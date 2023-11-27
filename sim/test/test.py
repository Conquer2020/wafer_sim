x0=5
x1=4
y0=3
y1=7
#here I define the noc link is occupied by only one process until the process release it.
for yy1 in range(y1):
    for xx1 in range(x1):
        for yy0 in range(y0):
            for xx0 in range(x0):
                print(((xx0+yy0*x0)+xx1*y0*x0)+x1*y0*x0*yy1)