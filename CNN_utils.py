




class MaxPoolOut(object):

    def __init__(self,stride):
        self.stride = self.int2list(stride)
    def __call__(self, N,C,H,W):
        Hout= (H-1)*self.stride[0]-2
        Wout = (W-1)*self.stride[1]-2
        #o = ((h+2*pad-di*(k-1)-1)
        return N,C,Hout,Wout

    def int2list(self, p):
        if type(p) == int:
            return [p, p]
        else:
            return p

class Conv2dOut(object):
    def __init__(self,k_size,stride = 1,pad = 0,dilation = 1):
        self.stride = self.int2list(stride)
        self.pad = self.int2list(pad)
        self.dilation = self.int2list(dilation)
        self.k_size = self.int2list(k_size)

    def __call__(self,N,C, H_in,W_in):
        return N,C,self.H_out(H_in),self.W_out(W_in)

    def H_out(self,H_in,):

        return ((H_in + 2*self.pad[0] - self.dilation[0] *(self.k_size[0]-1)-1)/self.stride[0])+1

    def W_out(self,W_in):

        return ((W_in + 2*self.pad[0] - self.dilation[0] *(self.k_size[0]-1)-1)/self.stride[0])+1

    def int2list(self,p):
        if type(p) == int:
            return [p, p]
        else:
            return p

