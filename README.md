# Gul
<p align="justify">The Python Neural Compiler 'Gul' (flower in Urdu) is a generalization of Recurrent Neural Nets, RNN, architecture via creating of (polymorphic) neural nets with external memories stack. The structure of NN in Gul is defined as assemble of rather simple, several layers only, neural nets. The assembly, in turn, can be of arbitrary size, but as a test example I’ll show how it works with simple assembly of only five nets.</p>
<p align="justify">Lets assume that our decision/classification is dependent on unstructured data. Let assume the given object is described in 3 sentences - Fact_A, Fact_B, Fact_C. Let’s also assume that Fact_B and Fact_C uses close language/words (Fact_B is mention later so we assume it’s to be more specific while Fact_C is, probably, more generic), but Fact_A is some distinct characteristic as if follows from semantic distance analysis. The scheme of classification process is then described on the left in fig below. To train machine on this irregular/unstructured data we dynamically compile a classification process tree (on the right of the fig).</p> 

<img src=https://github.com/AIDesigners/Gul/blob/master/doc/fig1.png /img>

<p align="justify">The nets assembly consists of 3 different neural nets (called neural functions in Gul). Initially the fact analyzing routine is called three times to form a memories stacks. The routine is a simple three-layered perceptron which is ‘written’ via training. Similarly to function it however has a strict signature - it inputs an array of memories, a array of fact representation and output an updated array of memories. In other words the fact analyzing perceptron takes existing memories stack (zeroes at the leaves of process tree) and a new input fact represented as a numpy array. The new knowledge is then integrated into current memories stack via passing of the fact data through neural net so it’s output is the updated memories stack. This way Fact_C updates empty memories stack, then Fact_B updates memories stack, which was created by analyzing of Fact_C, and Fact_A creates independent branch of memories.</p>
<p align="justify">At the second stage the memories about Fact_A and (Fact_B+Fact_C) are merged with second perceptron network, called memories aggregator. It takes two (independent) memories stacks and synthetases a joined one. Although not in this simple example, but memory aggregators can also be also nested, similarly to the case of nested fact analyzing calls.</p>
<p align="justify">Eventually a memories updates ends up with one memories stack which holds all data extracted from the bag of provided facts. The third neural function, classifier, is applied to resulting memories stack to make a conclusion about the described object.</p>

<p align="left">To compute (or train) the net assembly which is described above, with Gul, one need to execute the simple Pythonic code bellow:</p>
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


