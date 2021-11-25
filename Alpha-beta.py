import numpy as np

maxi=100000
mini=-100000

def minimax(d, nodeInd, maxPlayer, note, alpha, beta):
    if (d== depth):
        return note[nodeInd]
    
    if maxPlayer:
        best=mini 
        for a in range (0,branch):
            value= minimax(d+1, nodeInd*branch+1, False, note, alpha, beta)
            best= max(best, value)
            alpha = max(alpha,best)
            
            if (beta<=alpha):
                break
        return best
    else:
        best=maxi
        for i in range(0,branch):
            value = minimax(d+1,nodeInd*branch+i,True,note,alpha,beta)
            best=min(best,value)
            beta=max(alpha,best)
            
            if beta<=alpha:
                break
        return best
    
    
file = open("input.txt", 'r')
data= file.read() 

sentences= data.splitlines()

no_turn=int(sentences[0])
depth= no_turn*2
branch= int(sentences[1])

print("Depth : ", depth)
print("Branch : ", branch)

no_notes= pow(branch,depth)
print("Terminal States (Leaf Nodes) : ", no_notes)

start_range= int(sentences[2].split()[0])
finish_range= int(sentences[2].split()[1])

notes=np.random.randint(start_range, finish_range, no_notes, int)

max_value= minimax(0,0,True,notes,mini,maxi)
print("maximum amount : ", max_value)

c=0
for b in range(0, len(notes)):
    c=c+1
    if (notes[b]== max_value):
        break

print("comparisons:",no_notes)
print("comparisons:",c)



Input file
1
3
1 20
