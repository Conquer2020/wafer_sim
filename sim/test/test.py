
from enum import Enum

class MyEnum(Enum):
    def __str__(self):
        return self.name

    def __repr__(self):
        return f'{self.name}'

ML_STATE = MyEnum('ML_STATE', ('FORWARD', 'BACKWARD', 'PARAM_SYNC'))
a = [(1, 2, ML_STATE.FORWARD)]

print(a[0][2])
print(a[0])
