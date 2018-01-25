
"""
 A stack implementation: LIFO principle
 It is derived from list and have self.__len and self.__capacity fields
"""
class Stack(list):
    def __init__(self, *args):
        if args : list.__init__(self, *args)
        else    : list.__init__(self, [])
        self.__len = list.__len__(self) ; self.__capacity = self.__len #Actual length of the stack
        
    def __len__(self): return self.__len
    def __repr__(self): return list.__repr__(self[:self.__len])
    def __iter__(self): 
        count = self.__len
        while count : count -= 1 ; yield self[count]
        raise StopIteration
    
    def __getitem__(self, key):
        if   isinstance(key,   int) :
            if key < 0 or key >= self.__len : raise RuntimeError("IndexError: stack index is out of range")
            else : return list.__getitem__(self, key)   
        elif isinstance(key, slice) :    
            (start, stop, step) = key.indices(self.__len) ; del step
            if   start < 0 or start > self.__len : raise RuntimeError("IndexError: stack slicing start index is out of range")
            elif stop  < 0 or stop  > self.__len : raise RuntimeError("IndexError: stack slicing stop  index is out of range")
            else : return list.__getitem__(self, key)   
        else : raise RuntimeError("IndexError: stack index is neither integer nor slice")   
           
    def push(self, item):
        if self.__capacity != self.__len : self[self.__len] = item
        else : self.append(item) ; self.__capacity += 1 
        self.__len += 1
            
    def pop(self): #Return the last element and delete it
        if not self.__len : raise RuntimeError("Exhausted stack is popped")
        value = self[self.__len - 1] ; self[self.__len - 1] = None ; self.__len -= 1 ;  
        return value    
        
    def top(self): #Return the last element but not delete it
        try    : return self[self.__len - 1]
        except : raise RuntimeError("Exhausted stack is topped")
        
    def dlt(self): #Delete item but not return it        
        try    : self[self.__len - 1] = None ; self.__len -= 1
        except : self.__len  = 0 ; raise RuntimeError("Exhausted stack is deleted")
    
    #Operations on the top element
    def incr(self, value=1): self[self.__len-1] += value
    def decr(self, value=1): self[self.__len-1] -= value              

