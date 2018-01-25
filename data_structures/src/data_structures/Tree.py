
from data_structures import Stack

'''
gTree - generic tree class: unlimited and variable amount of branches/nodes

'''

#Tree almost virtual class - just Node interface and factory method in it 
class Node(object):
    def __init__(self, *args, **kwargs): pass
class Tree(object):
    @staticmethod        
    def _func_pass(item, **kwargs): pass
            
    #Factory initializer  
    def __new__(cls, *args, **kwargs):
        if ( (len(args) != 1) or (not isinstance(args[0], type(Node))) ) : raise("Tree instantiation problem with (not) given Node constructor")
        self = object.__new__(cls)
        if self is None : raise("Tree instantiation problem in __new__") 
        self._Node = args[0] ; return self
            
    def __init__(self, root=None): self.root = root
  
    #Root stuff
    @property #getter
    def root(self): return self._root 
    @root.setter #setter
    def root(self, value):
        if value is not None and not isinstance(value, Node) : raise TypeError("Invalid type of root.")  
        else                                                 : self._root = value
    @root.deleter   
    def root(self): self._root = None
  
    #Abstract factory
    def factoryNode(self, *args, **kwargs):
        node = self._Node(*args, **kwargs)
        if self.root is None : self.root = node
        return node
                              
#Generic node class
class gNode(Node):
    def __init__(self, **kwargs):
        try : self.value = kwargs['value']  #value in the node (object)
        except : self.value = None
        try : self.nodes = kwargs['nodes']  #array of children
        except : self.nodes = []


#Generic tree class
class gTree(Tree):
    #Template design pattern.  Deepth-first pre order traversing generator
    def traverse_df_pr_generator(self, root=None): #deep-first tree traversing
        if root is None : root = self.root 
        yield root 
        stack = Stack.Stack([root]) ; register = Stack.Stack([len(root.nodes)])
        while register.top() :
            register.decr() ; node = root.nodes[register.top()] ; register.push(len(node.nodes)) 
            yield node
            while node != root :      
                if register.top() :
                    register.decr() ; stack.push(node) 
                    node = node.nodes[register.top()] ; register.push(len(node.nodes))
                    yield node
                else              :
                    node = stack.pop() ; register.dlt()
        
    #Template design pattern.  Deepth-first post order traversing generator
    def traverse_df_po_generator(self, root=None): #deep-first tree traversing
        if root is None : root = self.root 
        stack = Stack.Stack([root]) ; register = Stack.Stack([len(root.nodes)])
        while register.top() :
            register.decr() ; node = root.nodes[register.top()] ; register.push(len(node.nodes))
            while node != root :      
                if register.top() :
                    register.decr() ; stack.push(node) 
                    node = node.nodes[register.top()] ; register.push(len(node.nodes))
                else              :
                    yield node ; node = stack.pop() ; register.dlt()
        yield root 
        
    #Template design pattern. Deepth-first traversing executor
    def execute_df(self, root=None, exec_pr=Tree._func_pass, exec_in=Tree._func_pass, exec_po=Tree._func_pass, **kwargs): #deep-first tree traversing
        if root is None : root = self.root
        exec_pr(root, **kwargs) #pre-order call
        stack = Stack.Stack([root]) ; register = Stack.Stack([len(root.nodes)])
        if not len(root.nodes) : exec_in(root,  **kwargs) #in-order call 
        while register.top() :
            if register.top() == len(root.nodes) - 1 : exec_in(root, **kwargs) #in-order call
            register.decr() ; node = root.nodes[register.top()] ; register.push(len(node.nodes))
            exec_pr(node, **kwargs) #pre-order call 
            while node != root :
                if not len(node.nodes) or register.top() == len(node.nodes) - 1 : exec_in(node, **kwargs) #in-order call 
                if register.top() :
                    register.decr() ; stack.push(node) ; node = node.nodes[register.top()] ; register.push(len(node.nodes))
                else              :
                    exec_po(node, **kwargs) #post-order call
                    node = stack.pop() ; register.dlt()
        exec_po(root, **kwargs) #post-order call 
        
        