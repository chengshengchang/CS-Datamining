from gurobipy import*

E={}
#兩項投入兩項產出  
I=2
O=2
#X、Y為各DMU的產出與投入
DMU,X,Y=multidict({('studentA'):[[13,144],[72,0.52]],('studentB'):[[23,115],[73,0.26]],('studentC'):[[33,117],[76,0.66]],('studentD'):[[43,124],[81,1]],('studentE'):[[53,134],[84,0.75]]})

for r in DMU:
    
    v,u={},{}

    m=Model("study_efficiency_model")
    
    for i in range(I):
        v[r,i]=m.addVar(vtype=GRB.CONTINUOUS,name="v_%s%d"%(r,i),lb=0.00000001)
    
    for j in range(O):
        u[r,j]=m.addVar(vtype=GRB.CONTINUOUS,name="u_%s%d"%(r,j),lb=0.00000001)
    
    m.update()
    
    m.setObjective(quicksum(u[r,j]*Y[r][j] for j in range(O)),GRB.MAXIMIZE)
        
    m.addConstr(quicksum(v[r,i]*X[r][i] for i in range(I))==1)
    for k in DMU:
        m.addConstr(quicksum(u[r,j]*Y[k][j] for j in range(O))-quicksum(v[r,i]*X[k][i] for i in range(I))<=0)
    
    m.optimize()
    E[r]="The efficiency of DMU %s:%0.3g"%(r,m.objVal)

    
for r in DMU:
    print (E[r])