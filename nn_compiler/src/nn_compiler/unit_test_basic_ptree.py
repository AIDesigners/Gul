'''
Created on Jan 8, 2018

@author: alex
'''
import unittest

import numpy as np
import nn_compiler.nn_compiler as nncp

#The unit test
class test_nn_compiler(unittest.TestCase):
    
    def test_add(self):
                
        SIZE_MSTACK = 3 #Global (relatively to current process tree) memory stack size

        '''
        Construct a simple process tree: size_mstack = 3, size_in = 6, size_out = 1
        label (const 1)
        ^- nfunc2() #classifier perceptron (3, 3, ->1)
            ^- nfunct1() # memories stack aggregator(.->3 + .->3, 3, ->3)
                ^ ^- nfunct0(.->3+(3+3), 9, ->3) # branch{1,0}
                |
                nfunct0(.->3+(3+3), 9, ->3) # branch{0,1}
                ^-nfunct0(null->3+(3+3), 9, ->3) # branch{0,0} 
        nfunct0: { 
                 [{0.3, 0.6, 0.9},
                  { {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 }
                    {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 }, 
                    {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3 } }, 
                 { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 },
                  { {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3}, {0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3}, {0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5},          
                    {0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9}, {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2}, {0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7},
                    {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}, {0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8}, {0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6} } ]
                 } 
        nfunct1: { 
                 [{ 0.9, 0.6, 0.3},
                  { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} } 
                  { 0.3, 0.6, 0.9},
                  { {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, {0.7, 0.8, 0.9, 0.1, 0.2, 0.3}, {0.4, 0.5, 0.6, 0.7, 0.8, 0.9} } ]
                 }
        nfunct2 {
                [{ 0.6},
                 { {0.3, 0.6, 0.9} }
                 { 0.1, 0.3, 0.6},
                 { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} } ]
                }
        data branch(0,0) = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, branch(0,1) = {0.7, 0.8, 0.9, 0.1, 0.2, 0.3}, branch(1,0) = {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}       
        '''
        
        nfunct2 = nncp.nf_perceptron(name = "nfunct2", size_in = SIZE_MSTACK,               size_mid = 3*2 + 1,           size_out = 1,           trainable = True, dtype = np.float32)
        nfunct1 = nncp.nf_perceptron(name = "nfunct1", size_in = SIZE_MSTACK + SIZE_MSTACK, size_mid = 3*2 + SIZE_MSTACK, size_out = SIZE_MSTACK, trainable = True, dtype = np.float32)
        nfunct0 = nncp.nf_perceptron(name = "nfunct0", size_in = SIZE_MSTACK + 6,           size_mid = 9*2 + SIZE_MSTACK, size_out = SIZE_MSTACK, trainable = True, dtype = np.float32)
        nfunct2.set_net(np.asarray([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     0.1, 0.3, 0.6,
                                     0.3, 0.6, 0.9,
                                     0.6 ],
                                     dtype=np.float32))
        nfunct1.set_net(np.asarray([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     0.3, 0.6, 0.9,
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                     0.9, 0.6, 0.3 ],
                                     dtype=np.float32))
        nfunct0.set_net(np.asarray([ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5,          
                                     0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7,
                                     0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  
                                     0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,  
                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                     0.3, 0.6, 0.9 ],
                                     dtype=np.float32))
        
        process_tree = nncp.pTree(nncp.NeuralNode, size_mstack = SIZE_MSTACK); 
        process_tree.root = nncp.NeuralNode(SIZE_MSTACK, nfunct2, data = None, nfn_transform_data = None, trainable = True, isRoot = True)
        process_tree.root.node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct1, data = None, nfn_transform_data = None, trainable = True) )
        process_tree.root.nodes[0].node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct0, data = np.asarray([0.7, 0.8, 0.9, 0.1, 0.2, 0.3], dtype=nfunct0.dtype), nfn_transform_data = None, trainable = True) )
        process_tree.root.nodes[0].nodes[0].node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct0, data = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=nfunct0.dtype), nfn_transform_data = None, trainable = True) )
        process_tree.root.nodes[0].node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct0, data = np.asarray([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=nfunct0.dtype), nfn_transform_data = None, trainable = True) )
        
        process_tree.calc_v() #Forward

        self.assertTrue(np.allclose(process_tree.root.v[-1], np.asarray(0.894925848853193), atol=1e-04, equal_nan=False), msg = "Deviation in process tree classification is detected!")
        
        process_tree.calc_dv(np.array([1./process_tree.root.v[-1]],dtype=process_tree.root.nfunct.dtype)) #Backward
 
        #df/dx = [ 0.00801906, 0.010146435, 0.012273812 ]
        nfunct2_grad = np.asarray([ 0.005811560, 0.006314815, 0.006642064, 0.00704986, 0.007660355, 0.008057334, 0.004479056, 0.004866926, 0.005119143, 
                                    0.007129757, 0.008648947, 0.005495019, 
                                    0.068753291, 0.087828995, 0.098565573,
                                    0.105074342 ],
                                    dtype=np.float32); 
        nfunct1_grad = np.asarray([ 0.0000827575, 0.0000829521, 0.0000831124, 0.0000824158, 0.0000827509, 0.0000829678,
                                    0.0000361613, 0.0000362463, 0.0000363163, 0.0000360119, 0.0000361582, 0.0000362530,
                                    0.0000139447, 0.0000139775, 0.0000140045, 0.0000138871, 0.0000139435, 0.0000139801,
                                    0.000083533, 0.0000365001, 0.0000140753,
                                    0.001106325, 0.00117567, 0.00119831, 0.000940345, 0.000999288, 0.00101853, 0.000716006, 0.000760886, 0.000775538,
                                    0.00120850, 0.001027195, 0.000782135 ],
                                   dtype=np.float32); 
        nfunct0_grad = np.asarray([ 0.0000000121446, 0.0000000122349, 0.0000000122942, 0.0000000485976, 0.0000000598433, 0.0000000710890, 0.0000000711596, 0.0000000824053, 0.0000000936510,
                                    0.0000000187179, 0.0000000188571, 0.0000000189485, 0.0000000455943, 0.0000000555823, 0.0000000655703, 0.0000000583344, 0.0000000683224, 0.0000000783105,
                                    0.0000000158011, 0.0000000158777, 0.0000000556353, 0.0000000683677, 0.0000000156845, 0.0000000811001, 0.0000000794000, 0.0000000921324, 0.0000000104865,
                                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0., 0., 
                                    0.000000984253, 0.000000926432, 0.000000828694 ],
                                   dtype=np.float32); 
                                             
        self.assertTrue(np.allclose(nfunct2.grad, nfunct2_grad, atol=1e-06, equal_nan=False), msg = "Deviation in nfunct2 (memories stack classifier) gradient is detected!")
        self.assertTrue(np.allclose(nfunct1.grad, nfunct1_grad, atol=1e-08, equal_nan=False), msg = "Deviation in nfunct1 (aggregator of memory stacks) gradient is detected!")
#        self.assertTrue(np.allclose(nfunct0.grad, nfunct0_grad, atol=1e-10, equal_nan=False), msg = "Deviation in nfunct0 (input data memorization) gradient is detected!")
        
        