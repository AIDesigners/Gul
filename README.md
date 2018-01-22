# Gul
The Python Neural Compiler is about

To compute (or train) the net above you need to execute the code:
<pre><code><i>
#Define neural functions
nfunct2 = nncp.nf_perceptron(name = "nfunct2", size_in = SIZE_MSTACK,               size_mid = SIZE_N2*2 + 1,           size_out = 1,           trainable = True, dtype = np.float32)
nfunct1 = nncp.nf_perceptron(name = "nfunct1", size_in = SIZE_MSTACK + SIZE_MSTACK, size_mid = SIZE_N1*2 + SIZE_MSTACK, size_out = SIZE_MSTACK, trainable = True, dtype = np.float32)
nfunct0 = nncp.nf_perceptron(name = "nfunct0", size_in = SIZE_MSTACK + SIZE_IN,     size_mid = SIZE_N0*2 + SIZE_MSTACK, size_out = SIZE_MSTACK, trainable = True, dtype = np.float32)

#Define process tree
process_tree = nncp.pTree(nncp.NeuralNode, size_mstack = SIZE_MSTACK); 
process_tree.root = nncp.NeuralNode(SIZE_MSTACK, nfunct2, data = None, nfn_transform_data = None, trainable = True, isRoot = True)
process_tree.root.node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct1, data = None, nfn_transform_data = None, trainable = True) )
process_tree.root.nodes[0].node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct0, data = np.asarray(INPUT_B, dtype=nfunct0.dtype), nfn_transform_data = None, trainable = True) )
process_tree.root.nodes[0].nodes[0].node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct0, data = np.asarray(INPUT_C, dtype=nfunct0.dtype), nfn_transform_data = None, trainable = True) )
process_tree.root.nodes[0].node_add( process_tree.factoryNode(SIZE_MSTACK, nfunct0, data = np.asarray(INPUT_A, dtype=nfunct0.dtype), nfn_transform_data = None, trainable = True) )

#Compute net and derivatives (for training)
process_tree.calc_v() 
process_tree.calc_dv(np.array([1./process_tree.root.v[-1]], dtype=process_tree.root.nfunct.dtype))

#Call your solver
(PARAMS_NN0, PARAMS_NN1, PARAMS_NN2) = your_NN_solver(PARAMS_NN0, PARAMS_NN1, PARAMS_NN2)

#Update the net with new parameters
nfunct2.set_net(np.asarray(PARAMS_NN2, dtype=np.float32))
nfunct1.set_net(np.asarray(PARAMS_NN1, dtype=np.float32))
nfunct0.set_net(np.asarray(PARAMS_NN0, dtype=np.float32))

</i></code></pre>


