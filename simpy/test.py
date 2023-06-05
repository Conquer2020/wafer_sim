Nd=2
Nm=3
L=[0,1,2,3,4,5]
Nd_g=[L[i::Nm] for i in range(Nm)]
Nm_g= [L[i*Nm:(i+1)*Nm:] for i in range(Nd)]
print(Nd_g)
print(Nm_g)