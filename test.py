
import diffractix as dfx
from diffractix.graph import Node, Parameter
from diffractix.graph import ops
from diffractix.elements import *


if __name__ == "__main__":
    L1 = ThinLens(f=0.1, label="L1").variable()
    L2 = ThinLens(f=0.1, label="L2").variable()
    S1 = Space(d=L1.f + L2.f, label="S1")
    # S4 = Space(d=L1.f + L2.f, label="S2")
    # S2 = Space(d=L2.f + L1.f, label="S3")
    # S4 = Space(d=S2.d * 9 + 828 * L1.f * 2983, label="S3")
    

    # # S2 = Space(d=2 * S1.d, label="S2")
    # # S3 = Space(d= S2.d * L1.f, label="S3")

    # for k, v in Node._cache.items():
    #     print(k, v)

    # print(S1.d.value)
    # print(S2.d.value)
    # print(S4.d.value)

    # print(S1.length)

    # S5 = Space(d = 2)
    # S5.d = 5
    # print(S5.length)
    # print(type(S5.d))
    # print(S5.d)

    # print(S5.has_variable_length)
    # S5.variable()
    # print(S5.has_variable_length)

    print(S1.has_variable_length)
    L1.fixed()
    L2.fixed()
    print(S1.has_variable_length)

    S1.variable("d")

    print(S1.has_variable_length)

