import sys
import HW5_mod
import HW5_mod7
st=sys.argv[1]
t=st.index('/')
n=st.index('.')
inp2=st[t+1:n]
c,r=HW5_mod.main(sys.argv[1],inp2)
b,m=HW5_mod7.main(sys.argv[1],inp2)


if b >= c:
    print r
else:
    print m
