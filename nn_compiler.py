
from Tree import gTree, gNode
import numpy as np
import caffe
import types
    
#Virtual class of neuronal function
class nfunct(object):
    #def init_net(self): raise Exception("Virtual method init_net() is not implemented, please, check your code")
    #def update_grad(self): raise Exception("Virtual method update_grad() is not implemented, please, check your code")
    #Virtual class
    def __new__(cls, *args, **kwargs): 
        assert (isinstance(cls, type(nfunct)) and cls.__name__ != "nfunct" ),  "Caught constructor in non-instantiable class"
        #Check for virtual methods
        virtual_methods = ['init_net', 'calc_v', 'calc_dv', 'init_grad'] ; real_methods = [ real_method for real_method in dir(cls) if (isinstance(getattr(cls,real_method,None), types.FunctionType)) ]
        assert set(virtual_methods).issubset(set(real_methods)), "Inheritance error: not all virtual methods are implemented in the derived class" 
        return object.__new__(cls) #Make it's derived instance 
    def __init__(self, name = "perceptron", size_in = 1024, size_mid = 3072, size_out = 1024, trainable = True, dtype = np.float32):
        self.name = name ; self.size_in  = size_in ; self.size_out = size_out ; self.size_mid = size_mid ; self.trainable = trainable ; self.dtype = dtype
        assert  self.size_in, "A layer without input"
        assert (self.size_mid and self.size_out), "Inconsistent neural net is requested"
        self.init_net()  #Clarify the neural function
        assert hasattr(self, 'size_grad') and isinstance(self.size_grad, int), "error in constructor: init_net() should set integer size_grad variable"
        if self.trainable : self.grad = np.empty(shape=(self.size_grad,), dtype=self.dtype)
              
        
# https://gist.github.com/rafaspadilha/a67008cc3bd93bc2c1fc368c363ee363

#Instantiable perceptron.
#Note. (self.size_mid-self.size_out)//2 is actual amount of hidden neurons due separation of fully connected and sigmoid layers in Caffe
class nf_perceptron(nfunct): 
    def init_net(self): #kind of __init__ 
        assert not (self.size_mid-self.size_out)%2 , "Due to Caffe architecture perceptron layers has separate sigmoid layer and thus amount of hidden neurons should be divisible by two"
        net = caffe.NetSpec()  #Create void net
        #force_backward: true
        #net.data = caffe.layers.Input(num = 1, channels = 1, height = 1, width = self.size_in) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        net.data = caffe.layers.Input(shape=[dict(dim=[1, 1, 1, self.size_in])]) ; net.data.name = 'data'  
        net.fc1 = caffe.layers.InnerProduct(net.data, num_output=(self.size_mid-self.size_out)//2, weight_filler={'type' : 'constant', 'value' : 0}, bias_filler = {'type' : 'constant', 'value' : 0}) 
        net.sigmoid1 = caffe.layers.Sigmoid(net.fc1)
        net.fc2 = caffe.layers.InnerProduct(net.sigmoid1,  num_output=self.size_out, weight_filler=dict(type='xavier')) 
        net.sigmoid2 = caffe.layers.Sigmoid(net.fc2) ; net.sigmoid2.name = 'last_layer'
        with open("/tmp/_tmp_net.pb2", 'w') as f: f.write('force_backward: true\n') ; f.write(str(net.to_proto()))
        self.net = caffe.Net("/tmp/_tmp_net.pb2", caffe.TRAIN if self.trainable else caffe.TEST)
        #self.net = caffe.Net(net.to_proto(), caffe.TEST) 
        #self.net.ParseFromString(str(net.to_proto()))
        #self.net = caffe_pb2.NetParameter()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.size_grad =  (self.size_mid-self.size_out)//2 + self.size_out + self.size_in * (self.size_mid-self.size_out)//2 + self.size_out * (self.size_mid-self.size_out)//2 
        
    def set_net(self, p):
        assert ( isinstance(p, np.ndarray) or len(p.shape) != 1 or p.shape[0] != self.size_grad ), "Incorrect parameters assignment"
        shift = 0
        self.net.params['fc1'][0].data[...] = p[ shift : shift + self.size_in  * (self.size_mid-self.size_out)//2].reshape((self.size_mid-self.size_out)//2, self.size_in ) ; shift += self.size_in  * (self.size_mid-self.size_out)//2
        self.net.params['fc1'][1].data[...] = p[ shift : shift + (self.size_mid-self.size_out)//2] ; shift += (self.size_mid-self.size_out)//2
        self.net.params['fc2'][0].data[...] = p[ shift : shift + (self.size_mid-self.size_out)//2 * self.size_out].reshape(self.size_out, (self.size_mid-self.size_out)//2) ; shift += (self.size_mid-self.size_out)//2 * self.size_out
        self.net.params['fc2'][1].data[...] = p[ shift : shift + self.size_out] ; shift += self.size_out
    def get_net(self, p): 
        assert ( isinstance(p, np.ndarray) or len(p.shape) != 1 or p.shape[0] != self.size_grad ), "Incorrect parameters assignment"
        shift = 0
        p[ shift : shift + self.size_in  * (self.size_mid-self.size_out)//2] = self.net.params['fc1'][0].data[...].reshape(self.size_in  * (self.size_mid-self.size_out)//2) ; shift += self.size_in  * (self.size_mid-self.size_out)//2
        p[ shift : shift + (self.size_mid-self.size_out)//2] = self.net.params['fc1'][1].data[...] ; shift += (self.size_mid-self.size_out)//2
        p[ shift : shift + (self.size_mid-self.size_out)//2 * self.size_out] = self.net.params['fc2'][0].data[...].reshape((self.size_mid-self.size_out)/2* self.size_out) ; shift += (self.size_mid-self.size_out)//2 * self.size_out
        p[ shift : shift + self.size_out] = self.net.params['fc2'][1].data[...] ; shift += self.size_out

    def calc_v(self, v): pass
    def init_grad(self):
        if self.trainable : 
            if self.grad is not None and self.grad.size == self.size_grad : self.grad[ : ] = 0.
            else : self.grad = np.zeros(shape=(self.size_grad,), dtype=self.dtype)
    def calc_dv(self, v, dv): 
        assert (isinstance(v, np.ndarray) and v.size == self.size_in + self.size_mid + self.size_out and v.dtype == self.dtype), "Incompatible activity array provided to calc_dv()"
        assert (isinstance(dv, np.ndarray) and dv.size == self.size_out and dv.dtype == self.dtype), "Incompatible gradient array provided to calc_dv()"
        if self.trainable :
            diff = self.net.backward(['fc2', 'fc1', 'data'], sigmoid2 = dv.reshape((1,self.size_out)))
            shift = 0 
            self.grad[shift : shift + self.size_in  * (self.size_mid-self.size_out)//2] += np.outer(diff['fc1'].flatten(), v[ : self.size_in]).flatten() ; shift += self.size_in  * (self.size_mid-self.size_out)//2
            self.grad[shift : shift +                 (self.size_mid-self.size_out)//2] += diff['fc1'].flatten() ; shift += (self.size_mid-self.size_out)//2    
            self.grad[shift : shift + self.size_out * (self.size_mid-self.size_out)//2] += np.outer(diff['fc2'].flatten(), v[self.size_in + (self.size_mid-self.size_out)//2 : self.size_in + (self.size_mid-self.size_out)]).flatten() ; shift += self.size_out * (self.size_mid-self.size_out)//2
            self.grad[shift : shift + self.size_out]                                    += diff['fc2'].flatten() ; shift += self.size_out    
        else              :
            diff = self.root.nfunct.net.backward(['data'], sigmoid2 = dv.reshape((1,self.size_out)))
        return diff['data'].flatten()        
            
#The neural node    
class NeuralNode(gNode):

    # Fields:
    # nfunct - neural function (Caffe Net)
    # data - data
    # nodes - list of branches
    def __init__(self, *args, **kwargs):
        assert len(args) > 0 or not isinstance(args[0], int), "memory stack size is not provided for neural node constructor"
        self.size_mstack = args[0]
        assert len(args) > 1 or not isinstance(args[1], nfunct), "neural function is not provided to neural node constructor"
        assert self.size_mstack <=  args[1].size_in,  "Neural function is connected to only a fragment of tree memories stack"
        try    :
            if kwargs["isRoot"] is not True : raise
        except : assert self.size_mstack == args[1].size_out, "Neural function outputs only a fragment of tree memories stack, which is normally allowed only for root. In the latter case please assign optional keyword isRoot=True"
        self.nfunct = args[1] ; self.nodes = []
        try    : self.data = kwargs["data"] 
        except : self.data = None
        try    : self.nfn_transform_data = kwargs['nfn_transform_data'] 
        except : self.nfn_transform_data = None     
        try    : self.trainable = kwargs['trainable']
        except : self.trainable = False
            
    #Root stuff
    @property #getter
    def nfunct(self): return self._nfunct 
    @nfunct.setter #setter
    def nfunct(self, value):
        if not isinstance(value, nfunct) : raise TypeError("Invalid type of neural function.")  
        else                             : self._nfunct = value
        
    def node_add(self, node):
        assert isinstance(node, NeuralNode), "Trying insert unknown object into a neural node"
        assert self.nfunct.dtype == node.nfunct.dtype , "Limitation violated: only the same datatype is allowed in the nodes of process tree" 
        assert self.nfunct.size_in >= self.size_mstack and self.nfunct.size_in >= node.nfunct.size_out + sum([nnode.nfunct.size_out for nnode in self.nodes]), "Adding of a new function exceeds the input capacity"
        if node not in self.nodes : self.nodes.append(node)
    def node_del(self, node): 
        assert isinstance(node, NeuralNode), "Trying insert unknown object into a neural node"
        while node in self.nodes : self.nodes.remove(node)              
        
    #Memory management    
    def allocate_vmem(self,  v = None):
        if isinstance(v,  np.ndarray) : 
            assert v.shape[0]  != self.nf.size_in + self.nf.size_med + self.nf.size_out, "Inappropriate size of preallocated array is passed"
            self.v  = v
        self.v  = np.empty(shape=(self.nf.size_in + self.nf.size_med + self.nf.size_out,), dtype=self.fn.dtype) 
        return self.v
    def free_vmem(self):
        if self.v  is not None  : del(self.v)  ; self.v  = None
    def allocate_dvmem(self, dv = None):        
        if isinstance(dv, np.ndarray) : 
            assert dv.shape[0] != self.nf.size_in + self.nf.size_med + self.nf.size_out, "Inappropriate size of preallocated array is passed"
            self.dv = dv
        self.dv = np.empty(shape=(self.nf.size_in + self.nf.size_med + self.nf.size_out,), dtype=self.fn.dtype) 
        return self.dv
    def free_dvmem(self):                     
        if self.dv is not None : del(self.dv) ; self.dv = None
        
                    
#Process tree structure                    
class pTree(gTree):
    size_mstack = None    
    def __init__(self, *argc, **argv):
        super(gTree, self).__init__()
        try    : self.size_mstack = argv['size_mstack']
        except : raise "memory stack size is not provided!"
        #setup CPU computations
        caffe.set_mode_cpu() 
        
    # A method to clear temporary used memory of a process tree                    
    def clear(self):
        for nnode in self.traverse_df_po_generator() :
            nnode.v  = None ; nnode.dv = None      

    # A method to compute neurons activity  (forward)
    def  calc_v(self): 
        for nnode in self.traverse_df_po_generator():
            #Smart stack copy  
            nnode.v = np.empty(shape=(nnode.nfunct.size_in + nnode.nfunct.size_mid + nnode.nfunct.size_out,), dtype=nnode.nfunct.dtype) ; shift = 0
            if not len(nnode.nodes) : nnode.v[ : self.size_mstack] = 0 ; shift = self.size_mstack 
            else :
                for nnnode in nnode.nodes : nnode.v[shift : shift + nnnode.nfunct.size_out] = nnnode.v[-nnnode.nfunct.size_out : ] ; shift += nnnode.nfunct.size_out 
            if shift != nnode.nfunct.size_in : 
                if nnode.nfn_transform_data is not None : nnode.nfn_transform_data(shift)
                else                                    : nnode.v[shift : nnode.nfunct.size_in] = nnode.data[ : nnode.nfunct.size_in - shift]
            #Load data into the nfunct, forward it and save activities
            nnode.nfunct.net.blobs['data'].data[0][0][0][...] = nnode.v[ : nnode.nfunct.size_in]    
            nnode.nfunct.net.forward() #forward_prefilled
            blobs_iterator = iter(nnode.nfunct.net.blobs) ; next(blobs_iterator) 
            shift = nnode.nfunct.size_in 
            for blob in blobs_iterator :  
                nnode.v[shift : shift + nnode.nfunct.net.blobs[blob].data.size] = nnode.nfunct.net.blobs[blob].data.flat[...] ; shift += nnode.nfunct.net.blobs[blob].data.size
            
    # A method to compute gradient of neurons activity (backward)
    def calc_dv(self, loss_grad):
        assert (isinstance(loss_grad, np.ndarray) and loss_grad.shape == (self.root.nfunct.size_out,) and loss_grad.dtype == self.root.nfunct.dtype), "given loss function gradient is not a valid np.array"
        self.root.nfunct.init_grad() ; nfuncts = set([self.root.nfunct])
        self.root.dv = self.root.nfunct.calc_dv(self.root.v, loss_grad) 
        for nnode in self.traverse_df_pr_generator():
            shift = 0 
            for nnnode in nnode.nodes :
                if nnnode.nfunct not in nfuncts : nnnode.nfunct.init_grad() ; nfuncts.add(nnnode.nfunct)
                nnnode.dv = nnnode.nfunct.calc_dv(nnnode.v, nnode.dv[shift : shift + nnnode.nfunct.size_out]) ; shift += nnnode.nfunct.size_out
            nnode.v = None ; nnode.dv = None  #free some memory  

#self.root.nfunct.net.backward(['sigmoid2', 'fc2', 'sigmoid1', 'fc1', 'data'], sigmoid2 = lg)
