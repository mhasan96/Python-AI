
from queue import Queue

sample_input = {
"0" :["1", "2", "3"],
"1" :["0", "3", "4"],
"2" :["0", "3"], 
"3" :["0", "1", "2", "5", "6"],
"4" :["1", "7", "8"],
"5" :["3", "6"],
"6" :["3", "5", "7"],
"7" :["4", "6", "8"],
"8" :["4", "8"]
}

node_parent = {}
node_visited = {}
travers = []
queue = Queue()
level = {}

for node in sample_input.keys():
    node_visited[node]=False
    node_parent[node]=None
    level[node]=-1

x="0"
node_visited[x]=True
level[x]=0
queue.put(x)

while not queue.empty():
    y = queue.get()
    travers.append(y)

    for z in sample_input[y]:
        if not node_visited[z]:
            node_visited[z]=True
            node_parent[z]=u
            level[z]=level[y]+1
            queue.put(z)

print ("BFS Traversal", level["6"])




from queue import Queue

sample_input = {
"0" :["1", "2", "3"],
"1" :["3", "4"],
"2" :[ "3"], 
"3" :[ "5",],
"4" :["7", "8"],
"5" :["6"],
"6" :[ "7", "9"],
"7" :["8"],
"8" :["9"],
"9" :[]
}

def dfs_util(x):
    
    global level
    color={}
    dfs_traversal_output=[]
    traversal_time={}
    node_parent={}

    for node in sample_input.keys():
        color[node]="M"
        node_parent [node]=None
        traversal_time[node]=[-1,-1]
        level=-1
        color[x]= "P"

    dfs_traversal_output.append(x)
    for y in sample_input[x]:
        if color[y]== "M":
            node_parent[y]=x
            dfs_util(y)
    color[x]="Q"
    level+=1

comp_tab=[None]*2
dfs_util("3")
comp_tab[0]=level
dfs_util("5")
comp_tab[1]=level

comp_tab.sort()
print(comp_tab[0])

from queue import Queue

sample_input = {
"0" :["1", "2", "3"],
"1" :["3", "4"],
"2" :[ "3"], 
"3" :[ "5",],
"4" :["7", "8"],
"5" :["6"],
"6" :[ "7", "9"],
"7" :["8"],
"8" :["9"],
"9" :[]
}


def dfs_util(x):

    
    global level
    color={}
    dfs_traversal_output=[]
    traversal_time={}
    node_parent={}

    for node in sample_input.keys():
        color[node]="M"
        node_parent [node]=None
        traversal_time[node]=[-1,-1]
        level=-1
    color[x]= "P"
    
    dfs_traversal_output.append(x)

    for y in sample_input[x]:
        if color[y]== "M":
            node_parent[y]=x
            dfs_util(y)
    color[x]="Q"
    
    level+=1

comp_tab=[None]*5
dfs_util("0")
comp_tab[0]=level
dfs_util("1")
comp_tab[1]=level
dfs_util("3")
comp_tab[2]=level
dfs_util("5")
comp_tab[3]=level
dfs_util("7")
comp_tab[4]=level

comp_tab.sort()

print (comp_tab[0])


