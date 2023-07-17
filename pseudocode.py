import numpy as np

# Heat kernel convolution 
def forward(input,kernel):
    a1 = np.dot(kernel,input[0]) 
    z1 = BMO(a1)
    a2 = np.dot(z1,input[1])
    z2 = BMO(a2)
    a3 = np.dot(z2,input[3])
    return a3

# BMO algorithm
def BMO(input):
    for i in input:
        if i > 0.5:
            i = 1
        else:
            i = 0

def main():
    # for a given heat kernel 
    k=np.array[1,0.5] 
    input = image
    y = forward(input,k)
    print(y)

main() 
