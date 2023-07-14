import struct
import numpy as np
import math
from numpy.random import *

def Wino_inputTran(x_4d,padding):
    batch    = x_4d.shape[0]
    height   = x_4d.shape[2]
    width    = x_4d.shape[3] 
    channel  = x_4d.shape[1]

    inside = height
    assert (height == width)
#     B = np.array([[ 1.0,   0,   0,     0,    0,  0],
#                   [   0, 2/3,-2/3,-1/12, 1/12,  4],
#                   [-5/4, 2/3, 2/3,-1/24,-1/24,  0],
#                   [   0, 1/6, 1/6, 1/12,-1/12, -5],
#                   [ 1/4,-1/6,-1/6, 1/24, 1/24,  0],
#                   [   0,   0,   0,     0,    0,  1],])

    
    B = np.array([[ 4,  0,  0,  0,  0,  0],
                  [ 0, -4,  4, -2,  2,  4],
                  [-5, -4, -4, -1, -1,  0],
                  [ 0,  1, -1,  2, -2, -5],
                  [ 1,  1,  1,  1,  1,  0],
                  [ 0,  0,  0,  0,  0,  1],])
    
    
    inside_beta = math.ceil((inside +2*padding-2)/4)*4+2 
#     print(inside,inside_beta)
    blockn = (int)((inside_beta+2*padding-2)/4 )
#     print(blockn)
    
    M = (int)(blockn * blockn * batch);
    MSize = M if (M%128 == 0) else math.ceil(M/128)*128
#     print(MSize)
    K = channel
    KSize = (int((K-1)/8)+1)*8
    
    # WTF of the CHWN code?????
    x_4d = np.pad(x_4d, [(0,0), (0,KSize-K),(padding, inside_beta - inside-padding), (padding, inside_beta - inside- padding)], mode='constant')
    inputTran = np.zeros((36,KSize,MSize)).astype(np.float32)
    
    for i in range(channel):
        for j in range(blockn):
            for k in range(blockn):
                for n in range(batch):
                    iny = 4*j
                    inx = 4*k
                    cache = x_4d[n,i,iny:iny+6,inx:inx+6]
                    inputTran[:,i,j*blockn*batch+k*batch+n] = (B.T@cache@B).reshape(-1)
    return inputTran

# def MM(a,b):
#     assert(len(a.shape)==2)
#     assert(len(b.shape)==2)
#     assert(a.shape[0] == b.shape[1])
#     M = a.shape[1]
#     K = a.shape[0]
#     N = b.shape[0]
    
#     c = np.zeros((M,N)).astype(np.float32)
#     for m in range(M):
#         for n in range(N):
#             for k in range(K):
#                 c[m,n] += a[m,k] * b[k,n]
    
#     return c

def Wino_kernelTran(k_4d):
    chn         = k_4d.shape[1]
    height      = k_4d.shape[2]
    width       = k_4d.shape[3] 
    numOfFilter = k_4d.shape[0]
    
    assert(height == width)
    assert(height == 3)
#     G = np.array([[ 1, 0, 0],
#                   [ 1, 1, 1],
#                   [ 1,-1, 1],
#                   [ 1, 2, 4],
#                   [ 1,-2, 4],
#                   [ 0, 0, 1]])

    G = np.array([[  1/4,    0,    0],
                  [ -1/6, -1/6, -1/6],
                  [ -1/6,  1/6, -1/6],
                  [ 1/24, 1/12,  1/6],
                  [ 1/24,-1/12,  1/6],
                  [    0,    0,    1]])

    
    
    N = numOfFilter
    NSize = (int((N-1)/128)+1)*128
    K = chn
    KSize = (int((K-1)/8)+1)*8
    
    k_4d = np.pad(k_4d, [(0,KSize-K),(0, 0), (0, 0), (0,NSize-N)], mode='constant')
    kernelTran = np.zeros((36,KSize,NSize)).astype(np.float32)
#     print(kernelTran.shape)
    for m in range(chn):
        for n in range(numOfFilter):
            cache = k_4d[n,m,:,:]
            kernelTran[:,m,n] = (G@cache@G.T).reshape(-1)
    return kernelTran

#     print(MdMT.shape)
#     for i in range(channel):
#         for j in range(tY):
#             iny = 3*j
#             for k in range(tX):
#                 inx    = 3*k
#                 cache  = x_3d[iny:iny+3,inx:inx+3,i]
#                 MdMT[j*tX+k,:,:,i] = M2@cache@M2.T
#     return MdMT

# print(input.shape)
# test0 = Wino_inputTran(input)
# test1 = Wino_kernelTran(kernel)

def gemm(inputTran,kernelTran,M,N, DEBUG):
    assert(inputTran.shape[2]==M)
    assert(kernelTran.shape[2]==N)
    assert(inputTran.shape[1] == kernelTran.shape[1])
    gemm_result = np.zeros((36,M,N)).astype(np.float32)
    K = inputTran.shape[1]
    
#     if(DEBUG): print("gemm process begin:")
    
#     for a in range(36):
#         for m in range(M):
#             for n in range(N):
#                 for k in range(K):
#                     gemm_result[a,m,n] += inputTran[a,k,m] * kernelTran[a,k,n]
#         progress = 100*(a+1)/(36)
#         if(DEBUG): print(round(progress,1),"% finished")
    for a in range(36):
        A = inputTran[a,:,:]
        B = kernelTran[a,:,:]
        gemm_result[a,:,:] = A.T@B 
        
#     if(DEBUG): print("gemm process complete.")
    return gemm_result

def Wino_OutputTran(matrix,outside,batch, channel):
    blockn = math.ceil((outside/4))
    M = blockn*blockn*batch
    MSize = M if (M%128 == 0) else math.ceil(M/128)*128
    assert(matrix.shape[0]==36)
    assert(MSize == matrix.shape[1])
    N = channel
    NSize = N if (N%128 == 0) else math.ceil(N/128)*128
    assert(NSize == matrix.shape[2])
    
    outputTran = np.zeros((batch,NSize,blockn*6,blockn*6)).astype(np.float32)
    
    for m in range(M):
        xblock = (int)(m/batch%blockn)
        yblock  = (int)(m/(batch*blockn))
        batch2 = (m%batch)
        for n in range(NSize):
            for pos in range (36):
                tx = pos%6
                ty = (int)(pos/6)
                outputTran[batch2,n,yblock*6+ty,xblock*6+tx] = matrix[pos,m,n]
    return outputTran

def Wino_inverseTran(outputTran,chn, batch, blockn,oside):
    numOfFilter = chn if (chn%128 == 0) else math.ceil(chn/128)*128
    assert(numOfFilter == outputTran.shape[1])
    assert(batch == outputTran.shape[0])
    assert(blockn == ( outputTran.shape[2]/6) )
    
    oside_beta = blockn *4
    
    output = np.zeros((batch , chn,oside_beta,oside_beta)).astype(np.float32)
    finalOutput = np.zeros((batch,chn,oside,oside)).astype(np.float32)
    
    A = np.array([[ 1, 0, 0, 0],
                  [ 1, 1, 1, 1],
                  [ 1,-1, 1,-1],
                  [ 1, 2, 4, 8],
                  [ 1,-2, 4,-8],
                  [ 0, 0, 0, 1]])

    
    for n in range( batch):
        for c in range (chn):
            for x in range(blockn):
                for y in range (blockn):
                    cache = outputTran[n,c,6*y:6*y+6,6*x:6*x+6]
                    # take care of the A and A.T, which one is in the lead and which one is following
                    output[n,c,4*y:4*y+4,4*x:4*x+4] = A.T@cache@A    
                    # take care of the leading dimension of the cache.
    finalOutput = output[:,:,0:oside,0:oside]
    return finalOutput

# Ground-Truth Naive Convolution of NCHW data layout
def Conv_NCHW(sample_input, sample_kernel,padding):
    assert(len(sample_input.shape)==4)
    assert(len(sample_kernel.shape)==4)
    assert(sample_input.shape[1]== sample_kernel.shape[1])
    
    sample_input = np.pad(sample_input, [(0,0),(0,0),(padding, padding), (padding, padding)], mode='constant')
    
    chn = sample_input.shape[1]
    inside = sample_input.shape[2]
    numOfConv = sample_input.shape[0]
    numOfFilter = sample_kernel.shape[0]
    oside = inside -2

#     print(chn, inside, numOfConv, numOfFilter)
    c = np.zeros((numOfConv,numOfFilter,oside,oside)).astype(np.float32)

    for chn_out in range(numOfFilter):
        for bat in range(numOfConv):
            for x in range(oside):
                for y in range(oside):
                    a = sample_input[bat,:,x:x+3,y:y+3].astype(np.float32)
                    b = sample_kernel[chn_out,:,:,:].astype(np.float32)
#                     assert(a.shape == b.shape)
                    c[bat,chn_out,x,y] = np.sum(np.multiply(a,b)).astype(np.float32)
#         progress = 100*(chn_out+1)/(numOfFilter)
#         print(round(progress,1),"% finished")          
        
    return c