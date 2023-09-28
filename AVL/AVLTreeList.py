# username - pelegchen
# id1      - 314953159
# name1    - Peleg Chen



from math import ceil, floor
from random import choice


"""
A class representing a node in an AVL tree
"""
class AVLNode(object):
	"""Constructor. 

	@type value: str
	@param value: data of your node
	"""
	def __init__(self, value=None):
		self.value = value
		self.left = None
		self.right = None
		self.parent = None
		self.height = -1
		self.isReal = False # Node is Virtual by default, must use convertToRegularNode to make it active.
		self.size = 0

	"""Check if node is considered Virtual.
 
	@rtype: boolean
	@returns: True if node is considered Virtual, False otherwise
	"""
	def isVirtualNode(self):
		return not self.isReal

	"""sets Node to Virtual, resetting every attr, except parent.

	"""
	def convertToVirtualNode(self):
		self.value = None
		self.left = None
		self.right = None
		self.height = -1
		self.size = 0
		self.isReal = False

	"""sets Node to Regular (not Virtual), with custom value.
 
	@type value: str
	"""
	def convertToRegularNode(self, value):
		self.value = value
		temp_left = AVLNode()
		temp_left.setParent(self)
		temp_right = AVLNode()
		temp_right.setParent(self)
		self.left = temp_left
		self.right = temp_right
		self.updateHeightBySons()	
		self.updateSizeBySons()
		self.isReal = True

	"""returns the left child

	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""
	def getLeft(self):
		return self.left

	"""returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""
	def getRight(self):
		return self.right

	"""returns the parent 

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""
	def getParent(self):
		return self.parent

	"""returns the value

	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""
	def getValue(self):
		return self.value

	"""returns the height

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""
	def getHeight(self):
		return self.height

	"""sets left child

	@type node: AVLNode
	@param node: a node
	"""
	def setLeft(self, node):
		self.left=node

	"""sets right child

	@type node: AVLNode
	@param node: a node
	"""
	def setRight(self, node):
		self.right=node

	"""sets parent

	@type node: AVLNode
	@param node: a node
	"""
	def setParent(self, node):
		self.parent=node

	"""sets value

	@type value: str
	@param value: data
	"""
	def setValue(self, value):
		self.value=value

	"""sets the height of the node

	@type h: int
	@param h: the height
	"""
	def setHeight(self, h):
		self.height=h

	"""returns whether self is not a virtual node 

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""
	def isRealNode(self):
		return self.isReal

	"""sets the size of the node

	@type node: int
	@param node: the size
	"""
	def setSize(self,s):
		self.size=s

	"""returns the size of the node 

	@rtype: int
	@returns: the size of the current node
	"""
	def getSize(self):
		return self.size

	"""Checks if the node is the right son of the node's parent
 
	@rtype:boolean
	@return: if the node is the right son we will get true if not we will get false
	"""
	def isRightChild(self):
		if self.getParent().getRight() is self:
			return True
		else:
			return False

	"""Checks if the node is the left son of the node's parent
 
	@rtype:boolean
	@return: if the node is the left son we will get true if not we will get false
	"""
	def isLeftChild(self):
		if self.getParent().getLeft() is self:
			return True
		else:
			return False

	"""Get the predecessor of the current node
	
	@rtype:AVLNode
	@return: pointer to the predecessor if exists, otherwise returns None.
	@Complexity: O(logn)
	"""
	def getPredecessor(self):
		if (self.getLeft().isRealNode()==True):
			nodePredecessor= self.getLeft()
			if nodePredecessor.getRight().isRealNode()==False:
				return nodePredecessor
			else:
				while nodePredecessor.getRight().isRealNode()==True:
					nodePredecessor=nodePredecessor.getRight()
				return nodePredecessor
		return None

	"""Get node current balance factor, depends on children height.

    @rtype: int
    @return: node current balance factor
    """
	def getBalanceFactor(self):
		if (self.isVirtualNode()==True):
			return 0
		else:
			return self.getLeft().getHeight()-self.getRight().getHeight()
	
	"""Check if the node is the left child of a given node. 
 
 	@type node_parent: AVLNode
	@rtype: boolean
	@return: True if the node is the left child of node_parent, False otherwise
	"""
	def isLeftChildOf(self, node_parent):
		if node_parent.getLeft() is self:
			return True

	"""Check if the node is the right child of a given node. 
 
 	@type node_parent: AVLNode
	@rtype: boolean
	@return: True if the node is the right child of node_parent, False otherwise
	"""
	def isRightChildOf(self, node_parent):
		if node_parent.getRight() is self:
			return True

	"""Check if the node is a child of a given node. 
 
 	@type node_parent: AVLNode
	@rtype: boolean
	@return: True if the node is a child of node_parent, False otherwise
	"""
	def isChildOf(self, node_parent):
		return self.isLeftChildOf(node_parent) or self.isRightChildOf(node_parent)

	"""Check if the node is a parent of a given node. 
 
 	@type node_child: AVLNode
	@rtype: boolean
	@return: True if the node is a parent of node_child, False otherwise
	"""
	def isParentOf(self, node_child):
		return node_child.isChild(self)

	"""Update the node's height based on children height. 
 
	@rtype: int
	@return: Updated height of node.
	"""
	def updateHeightBySons(self):
		temp = max(self.getLeft().getHeight(), self.getRight().getHeight()) + 1
		self.setHeight(temp)
		return temp 

	"""Update the node's size based on children size. 
 
	@rtype: int
	@return: Updated size of node.
	"""
	def updateSizeBySons(self):
		temp = self.getLeft().getSize() + self.getRight().getSize() + 1
		self.setSize(temp)
		return temp 

	"""Rebalancing the tree by rotations if necessary, finally returns the number of rebalancing operations if occurred
    
    @type AVLtree: AVLtree
    @type beforeInsertHeightParent: int
    @param AVLtree: the AVL Tree we inserted node.
    @param beforeInsertHeightParent: this is the height of the parent of node before the we inserted node.
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    @help functions: getBalanceFactor(AVLNode), _rotationRight(node_parent, node_child), _rotationRight(node_parent, node_child), updateRoot(AVLTree), updateRoot(AVLTree), updateHeightBySons(AVLNode), updateSizeBySons(AVLNode), updateAttr(AVLTree)
    @Complexity: O(log n)
	"""
	def insertRebalance(self,AVLtree,beforeInsertHeightParent):
		#count number of changes
		countChanges = 0

		if  self.getParent()!=None:

			# let y be the parent of the inserted node
			y = self.getParent()

			while y != None:  # Traversal from node to root

				#compute BF(y):
				BalanceFactor_y = y.getBalanceFactor()


				if abs(BalanceFactor_y)<=1 and y.getParent()==None:
					#Dont need any changes!
					"root --R->NODE or root --L->NODE SUCH THAT y is the root!"
					break


				"pay attention y.getHeight() is the height of y after insert"
				if (y.getHeight() == beforeInsertHeightParent and abs(BalanceFactor_y) < 2):
					break

				"else if |BF(y)|<2 and y's height changed go back to the top of the while loop with y's parent"
				if (abs(BalanceFactor_y) < 2) and (beforeInsertHeightParent!=y.getHeight()):
					y = y.getParent()
					continue

				elif abs(BalanceFactor_y)==2:

					if BalanceFactor_y == 2:
						"look at the BF of the left son"
						if y.getLeft().getBalanceFactor() == 1:
							y._rotationRight(y.getLeft())
							AVLtree.updateRoot()
							countChanges += 1
							break

						if y.getLeft().getBalanceFactor() == -1:
							y.getLeft()._rotationLeft(y.getLeft().getRight())
							y._rotationRight(y.getLeft())
							AVLtree.updateRoot()
							countChanges += 2
							break

					if BalanceFactor_y == -2:
						"look at the BF of the right son"
						if y.getRight().getBalanceFactor() == -1:
							y._rotationLeft(y.getRight())
							AVLtree.updateRoot()
							countChanges += 1
							break
						if y.getRight().getBalanceFactor() == 1:
							y.getRight()._rotationRight(y.getRight().getLeft()) #left right is no good! and the opposie
							y._rotationLeft(y.getRight())
							AVLtree.updateRoot()
							countChanges += 2
							break
			"update size and height"
			while y!=None:
				y.updateHeightBySons()
				y.updateSizeBySons()
				y=y.getParent()

		AVLtree.updateAttr()
		return countChanges

	"""Static function - use left rotation swap on two nodes.
	
 	@pre: node_child, node_parent are Regular (not Virtual) nodes, node_child is right son of node_parent
  	@type node_parent: AVLNode
   	@type node_child: AVLNode
	@help functions: updateHeightBySons, updateSizeBySons
	"""
	def _rotationLeft(node_parent, node_child):
		temp_parent = node_parent.getParent()
		temp_b = node_child.getLeft()

		node_parent.setParent(node_child)
		node_child.setLeft(node_parent)
  
		node_parent.setRight(temp_b)
		temp_b.setParent(node_parent) if temp_b is not None else None

		node_child.setParent(temp_parent)
		if temp_parent is not None:
			if node_parent.isLeftChildOf(temp_parent):
				temp_parent.setLeft(node_child)
			else:
				temp_parent.setRight(node_child)
    
		for item in (node_parent, node_child):
			item.updateHeightBySons()
			item.updateSizeBySons()
		
	"""Static function - use right rotation swap on two nodes.
	
 	@pre: node_child, node_parent are Regular (not Virtual) nodes, node_child is left son of node_parent
  	@type node_parent: AVLNode
   	@type node_child: AVLNode
	@help functions: updateHeightBySons, updateSizeBySons
	"""
	def _rotationRight(node_parent, node_child):
		temp_parent = node_parent.getParent()
		temp_b = node_child.getRight()

		node_parent.setParent(node_child)
		node_child.setRight(node_parent)
    
		node_parent.setLeft(temp_b)
		temp_b.setParent(node_parent) if temp_b is not None else None

		node_child.setParent(temp_parent)
		if temp_parent is not None:
			if node_parent.isLeftChildOf(temp_parent):
				temp_parent.setLeft(node_child)
			else:
				temp_parent.setRight(node_child)
	
		for item in (node_parent, node_child):
			item.updateHeightBySons()
			item.updateSizeBySons()
	
	"""Static function - rotate two nodes if possible (if connected).
	
  	@type node1: AVLNode
   	@type node2: AVLNode
	@help functions: _rotationRight, _rotationLeft
	"""
	def rotate(node1, node2):
		if node1.isLeftChildOf(node2):
			AVLNode._rotationRight(node2, node1)
		elif node1.isRightChildOf(node2):
			AVLNode._rotationLeft(node2, node1)
		elif node2.isLeftChildOf(node1):
			AVLNode._rotationRight(node1, node2)
		elif node2.isRightChildOf(node1):
			AVLNode._rotationLeft(node1, node2)
	
	"""Retrieves the k'th rank node in the subtree.

	@type i: int
	@pre: 0 <= k < self.getSize()
	@param k: rank of the Node
	@rtype: AVLNode
	@returns: The Node with rank k
	@Complexity: O(log n)
	"""
	def select(self, k):
		if self.isVirtualNode():
			return None
		r = self.getLeft().getSize() + 1
		if k == r:
			return self
		elif k < r:
			return self.getLeft().select(k)
		else:
			return self.getRight().select(k-r)

	"""Retrieves the "left-most" node (minimal node in in-order walk) in the subtree.

	@rtype: AVLNode
	@returns: minimal node in in-order walk
 	@Help Functions: _min_rec
	@Complexity: O(log n)
	"""
	def min(self):	
		return self._min_rec()

	"""Recursive function of function min.

	@rtype: AVLNode
	@returns: left most node without a left child
	@Complexity: O(log n)
	"""     
	def _min_rec(self):
		current_node_left = self.getLeft()
		if current_node_left.isVirtualNode():
			return self
		return current_node_left._min_rec()


"""
A class implementing the ADT list, using an AVL tree.
"""

class AVLTreeList(object):

	"""Constructor.  

	"""
	def __init__(self):
		self.size = 0
		self.root = AVLNode()
		self.firstNode=None
		self.lastNode=None

	"""returns the height of tree.

	@rtype: int
	@returns: height of tree
	"""
	def getHeight(self):
		return self._getRoot().getHeight()

	"""Update the relevant tree attributes of size.

 	@Complexity: O(logn)
	"""
	def updateAttr(self):
		self.updateLength()
		self.firstNode = self.retrieveNode(0)
		self.lastNode = self.retrieveNode(self.length() - 1)
 
	"""Static function - Merge 2 lists.

	@type lst1: list
	@type lst2: list
	@rtype: list
	@returns: Merged list
	"""
	def merge_lists(lst1, lst2):
		lst1_len = len(lst1)
		lst2_len = len(lst2)
		merged_lst = [None for i in range(lst1_len+lst2_len)]
		lst1_curr_index=0; lst2_curr_index=0; merged_lst_curr_index=0
		while  lst1_curr_index<lst1_len  and  lst2_curr_index<lst2_len: 
			if lst1[lst1_curr_index] < lst2[lst2_curr_index]:
				merged_lst[merged_lst_curr_index] = lst1[lst1_curr_index]
				lst1_curr_index+=1
			else:
				merged_lst[merged_lst_curr_index] = lst2[lst2_curr_index]
				lst2_curr_index+=1
			merged_lst_curr_index+=1

		merged_lst[merged_lst_curr_index:] = lst1[lst1_curr_index:] + lst2[lst2_curr_index:]
		return merged_lst

	"""Static function - Sort list.

	@type lst: list
	@rtype: list
	@returns: Merged list
	@Complexity: O(nlogn)
	"""
	def sort_list(lst):
		list_len = len(lst)
		if list_len <= 1: 
			return lst
		else:            
			return AVLTreeList.merge_lists(AVLTreeList.sort_list(lst[0:list_len//2]), AVLTreeList.sort_list(lst[list_len//2:list_len]))
 
	"""Go over each node in the tree in-order and executes a function with current node as an argument.

	@type function: function
	@help functions: _inOrderFuncWalkRec
	@Complexity: O(n) * O(k) (k - function complexity)
	"""
	def inOrderFuncWalk(self, func):
		self._inOrderFuncWalkRec(self.root, func)

	"""Recursive function of inOrderFuncWalk.

	@type function: function
	@type current_node: AVLNode
	@Complexity: O(n) * O(k) (k - function complexity)
	"""   
	def _inOrderFuncWalkRec(self, current_node, func):
		if not current_node.isVirtualNode():
			self._inOrderFuncWalkRec(current_node.left, func)
			func(current_node)
			self._inOrderFuncWalkRec(current_node.right, func)
   
	"""Go over each node in the tree post-order and executes a function with current node as an argument.

	@type function: function
	@help functions: _postOrderFuncWalkRec
	@Complexity: O(n) * O(k) (k - function complexity)
	"""
	def postOrderFuncWalk(self, func):
		self._postOrderFuncWalkRec(self.root, func)
        
	"""Recursive function of postOrderFuncWalk.

	@type function: function
	@type current_node: AVLNode
	@Complexity: O(n) * O(k) (k - function complexity)
	""" 
	def _postOrderFuncWalkRec(self, current_node, func):
		if not current_node.isVirtualNode():
			self._postOrderFuncWalkRec(current_node.left, func)
			self._postOrderFuncWalkRec(current_node.right, func)
			func(current_node)
   
	"""Go over each node in the tree post-order and update node attr by size and height.

	@help functions: postOrderFuncWalk
	@Complexity: O(n)
	"""
	def postOrderUpdateHeightAndSizeWalk(self):
		func = lambda node: node.updateSizeBySons() and node.updateHeightBySons()
		self.postOrderFuncWalk(func)

	"""Go over each node in the tree in-order, append it to a list and return the list.
	
	@rtype: list
	@returns: Merged list
	@help functions: inOrderFuncWalk
	@Complexity: O(n)
	"""
	def inOrderAppendToListWalk(self):
		temp_list = []
		func = lambda node: temp_list.append(node)
		self.inOrderFuncWalk(func)
		return temp_list

	"""Static Function - Create a tree list with given size, filled with None values. 
	
	@pre: size >= 0
	@type size: int
	@rtype: AVLTreeList
	@returns: AVL Tree list filled with None values.
	@help functions: createEmptyAvlTreeRec, postOrderUpdateHeightAndSizeWalk
	@Complexity: O(n)
	"""
	def createEmptyAvlTree(size):
		temp_tree = AVLTreeList()
		temp_tree.root = AVLNode()
		if size == 1:
			temp_tree.root.convertToRegularNode(None)
		elif size > 0:
			AVLTreeList._createEmptyAvlTreeRec(temp_tree.root, size, None)
		temp_tree.postOrderUpdateHeightAndSizeWalk() # After creating the tree update all relevant tree and node attr 
		temp_tree.updateAttr()
		return temp_tree
  
	"""Static Function - Recursive Function for createEmptyAvlTree
	
	@pre: size >= 0
	@type current_virtual_node: AVLNode
	@type size: int
	@type parent_node: AVLNode
	@Complexity: O(n)
	"""
	def _createEmptyAvlTreeRec(current_virtual_node, size, parent_node):
		if size >= 1: # Stopping condition
			current_virtual_node.convertToRegularNode(None)
			current_virtual_node.setParent(parent_node)
			if size == 2: 
				current_virtual_node.getLeft().convertToRegularNode(None)
				current_virtual_node.getLeft().setParent(current_virtual_node)
			if size == 3:
				current_virtual_node.getLeft().convertToRegularNode(None)
				current_virtual_node.getLeft().setParent(current_virtual_node)
				current_virtual_node.getRight().convertToRegularNode(None)
				current_virtual_node.getRight().setParent(current_virtual_node)
			else:
				AVLTreeList._createEmptyAvlTreeRec(current_virtual_node.getLeft(), ceil((size-1)/2), current_virtual_node)
				AVLTreeList._createEmptyAvlTreeRec(current_virtual_node.getRight(), floor((size-1)/2), current_virtual_node)
    
	"""Generator function - Yields values from list in random order.
		
	@type lst: list
	@rtype: str
	@returns: value
	@help functions: choice
	"""
	def _permutationGenerator(lst):
		while (len(lst) > 0):
			index = choice(range(len(lst)))
			temp = lst[index]
			lst.remove(temp)
			yield temp

	"""Generator function - Yields values from list in sorted order.
		
	@type lst: list
	@rtype: str
	@returns: value
	@help functions: sort_list
	"""
	def _sortGenerator(lst):
		none_list = [value for value in lst if value is None] # Used for edge case if None is present as a non-Virtual Node value, we put all None values first is sorted tree-list
		value_list = [value for value in lst if value is not None]
		value_list = AVLTreeList.sort_list(value_list)
		sorted_list = none_list + value_list
		while (len(sorted_list) > 0):
			temp = sorted_list.pop(0)
			yield temp

	"""Fill tree using values from a generator.
	
	@type gen: generator function
	@help functions: inOrderFuncWalk
	@Complexity: O(n)
	"""
	def _inOrderFillByGenWalk(self, gen):
		func = lambda node: node.setValue(next(gen))
		self.inOrderFuncWalk(func)

	"""Fill Tree-list with given list values.
	
	@type lst: list
	@help functions: _inOrderFillByGenWalk
	@Complexity: O(n)
	"""
	def inOrderFillByArrayWalk(self, lst):
		gen =  (value for value in lst)
		self._inOrderFillByGenWalk(gen)

	"""Fill Tree-list with given list values, in a random way.
	
	@type lst: list
	@help functions: _inOrderFillByGenWalk, _permutationGenerator
	@Complexity: O(n)
	"""
	def inOrderFillByArrayPermutationWalk(self, lst):
		gen =  AVLTreeList._permutationGenerator(lst)
		self._inOrderFillByGenWalk(gen)

	"""Fill Tree-list with given list values, in a sorted way.
	
	@type lst: list
	@help functions: _inOrderFillByGenWalk, _sortGenerator
	@Complexity: O(n)
	"""
	def inOrderFillByArraySortWalk(self, lst):
		gen =  AVLTreeList._sortGenerator(lst)
		self._inOrderFillByGenWalk(gen)

	"""returns whether the list is empty

	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""
	def empty(self):
		return self.size==0
 
	"""retrieves the value of the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the value of the i'th item in the list
	@Help Functions: retrieveNode()
	@Complexity: O(log n)
	"""
	def retrieve(self, i):
		temp = self.retrieveNode(i)
		if temp is None:
			return None	
		return temp.getValue()

	"""retrieve the associated node of the i'th item in the list.

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: AVLNode
	@returns: The Node of the i'th item in the list
	@Help Functions: treeSelect
	@Complexity: O(log n)
	"""
	def retrieveNode(self, i):
		return self.treeSelect(i+1)

	"""
 	Retrieves the associated node with k'th rank in the tree.

	@type i: int
	@pre: 1 <= k <= self.length()
	@param k: rank of a Node
	@rtype: AVLNode
	@returns: The Node of the k rank item in the list
	@Help Functions: AVLNode.select
	@Complexity: O(log n)
	"""
	def treeSelect(self,k):
		return self.root.select(k)

	"""
 	Rotate two Regular nodes if possible (if connected). Change tree root to relevant Node if necessary. 
	
  	@type node1: AVLNode
   	@type node2: AVLNode
	@Help Functions: AVLNode.rotate
	"""
	def rotate(self, node1, node2): # rotate as an AVLTreeList function, which accounts for root changing during rotation
		is_root_node1 = True if node1 is self.root else False
		is_root_node2 = True if node2 is self.root else False
		AVLNode.rotate(node1, node2)
		if is_root_node1 or is_root_node2:
			if is_root_node1:
				self.root = node2
			else:
				self.root = node1

	"""
 	Retrieves the current_node's successor.

 	@type current_node: AVLNode
	@rtype: AVLNode
	@returns: The successor of current_node
	@Help Functions: _successor_rec()
	@Complexity: O(log n)
	"""
	def successor(self, current_node):
		return  self._successor_rec(current_node)     
    
	"""
 	Recursive function of successor.

 	@type current_node: AVLNode
	@rtype: AVLNode
	@returns: The successor of current_node
	@Complexity: O(log n)
	"""
	def _successor_rec(self, current_node):
		if current_node.getRight().isVirtualNode():
			temp = current_node
			while temp.getParent() is not None:
				if temp.getParent().getLeft() is temp:
					return temp.getParent()
				temp = temp.getParent()
			return None
		else:
			return current_node.getRight().min()

	"""remove given node per Deletion BST Algorithm.

 	@type current_node: AVLNode
	@rtype: AVLNode
	@returns: node, from which (parent and onward) we must rebalance the tree.
	@Help Functions: successor
	@Complexity: O(log n)
	"""
	def _deletion(self, current_node):
		if current_node.getLeft().isVirtualNode() and current_node.getRight().isVirtualNode(): # If target node has 2 Real children (Real = Not Virtual nodes)
			current_node.convertToVirtualNode() # Because of our implementation to delete a node we only need to reset it to a Virtual Node
			return current_node
		elif not (current_node.getLeft().isVirtualNode()) and not (current_node.getRight().isVirtualNode()): # If target node has no Real children
			successor_node = self.successor(current_node)
			current_node.setValue(successor_node.getValue())
			return self._deletion(successor_node) # While this is a recursive call we know it will only work once because the successor should not have 2 children by the algorithm
		else: # If target node has one Real child
			only_child_node = current_node.getRight() if current_node.getLeft().isVirtualNode() else current_node.getLeft()
			current_parent = current_node.getParent()
			if current_parent is None: # Determine if the deleted node is root (= has no parent)
				self.root = only_child_node
				only_child_node.setParent(None)
			else:
				only_child_node.setParent(current_parent)
				if current_node.isLeftChild(): # Determine if the deleted was a right or a left child
					current_parent.setLeft(only_child_node)
				else:
					current_parent.setRight(only_child_node)
			return only_child_node
	
	"""Commence rebalancing from current node towards the root, fixing the balance factor with rotations.

 	@type current_node: AVLNode
	@type sum_of_fixes: int
	@rtype: int
	@returns: the number of rebalancing operations due to AVL rebalancing
	@Help Functions: updateSizeBySons, updateHeightBySons, getBalanceFactor, rotate
	@Complexity: O(log n). 
	"""
	def _rebalance(self, current_node, sum_of_fixes=0):
		if current_node is None: # Stopping condition: If we reached None = The last iteration was on the root Node 
			return sum_of_fixes
		current_node.updateSizeBySons()
		current_node.updateHeightBySons()
		balance_factor = current_node.getBalanceFactor()
		if abs(balance_factor) < 2: # Analyze BF cases and execute rotations according to the algorithm, when completed move onto parent Node
			return self._rebalance(current_node.getParent(), sum_of_fixes)
		elif abs(balance_factor) == 2:
			if balance_factor == 2:
				left_child = current_node.getLeft()
				left_child_balance_factor = left_child.getBalanceFactor()
				if left_child_balance_factor in (0,1):
					self.rotate(current_node, left_child)
					sum_of_fixes += 1
				elif left_child_balance_factor == -1:
					self.rotate(left_child, left_child.getRight())
					self.rotate(current_node, current_node.getLeft())
					sum_of_fixes += 2
			else:
				right_child = current_node.getRight()
				right_child_balance_factor = right_child.getBalanceFactor()
				if right_child_balance_factor in (0,-1):
					self.rotate(current_node, right_child)
					sum_of_fixes += 1
				elif right_child_balance_factor == 1:
					self.rotate(right_child, right_child.getLeft())
					self.rotate(current_node, current_node.getRight())
					sum_of_fixes += 2
			return self._rebalance(current_node.getParent(), sum_of_fixes)

	"""inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    @Help functions: convertToRegularNode, updateAttr, updateRoot, insertRebalance, getPredecessor
    @Complexity: O(log n)
    """
	def insert(self, i, val):
		newNode=AVLNode()
		newNode.convertToRegularNode(val)

		if (self.length()==i): #we will insert it to the end of the list


			if(self.empty()==True):
				self.root=newNode
				self.size += 1
				self.updateAttr()
				return 0


			else:

				AVLTreeList.updateRoot(self)  # update the root
				currentNode = self.root  # we will start from the root

				while (currentNode.getRight().isRealNode()==True):  #We will go right until it Virtual Node
					currentNode = currentNode.getRight()

				# make newNode as a right child
				currentNode.setRight(newNode)

				newNode.setParent(currentNode)#update the parent of the new node

				#update the firstNode and lastNode

				# update the self size of the tree
				self.size += 1



				#The height of the node whice newNode its right child
				beforeInsertHeightParent = currentNode.getHeight()


				# update the size and the height after the insert before rotations

				#THIS IS THE ROOT
				if currentNode.getParent()==None:
					currentNode.setSize(currentNode.getSize() + 1)

					currentNode.setHeight(max(currentNode.getRight().getHeight(), currentNode.getLeft().getHeight()) + 1)

				else:
					while currentNode.getParent() != None:
						currentNode.setSize(currentNode.getSize() + 1)

						currentNode.setHeight(
							max(currentNode.getRight().getHeight(), currentNode.getLeft().getHeight()) + 1)
						currentNode = currentNode.getParent()

				return AVLNode.insertRebalance(newNode,self,beforeInsertHeightParent)


		else: #i<n

			# print("hello im here")
			#update the root
			AVLTreeList.updateRoot(self)

			#find the current node of rank i+1 (indices begin at 0)
			currentNode_rank_i_plus_1=AVLNode.select(self.root,i+1)

			#if it has no left child


			if currentNode_rank_i_plus_1.getLeft().isRealNode()==False:
				beforeInsertHeightParent = currentNode_rank_i_plus_1.getHeight()
				#make newNode its left child
				currentNode_rank_i_plus_1.setLeft(newNode)
				newNode.setParent(currentNode_rank_i_plus_1)

				#update the firstNode and lastNode
				self.size += 1



				# update the size and the height after the insert before rotations

				# THIS IS THE ROOT
				if currentNode_rank_i_plus_1.getParent() == None:
					currentNode_rank_i_plus_1.setSize(currentNode_rank_i_plus_1.getSize() + 1)

					currentNode_rank_i_plus_1.setHeight(max(currentNode_rank_i_plus_1.getRight().getHeight(), currentNode_rank_i_plus_1.getLeft().getHeight()) + 1)
				else:

					while currentNode_rank_i_plus_1.getParent() != None:
						currentNode_rank_i_plus_1.setSize(currentNode_rank_i_plus_1.getSize() + 1)

						currentNode_rank_i_plus_1.setHeight(
							max(currentNode_rank_i_plus_1.getRight().getHeight(), currentNode_rank_i_plus_1.getLeft().getHeight()) + 1)
						currentNode_rank_i_plus_1 = currentNode_rank_i_plus_1.getParent()

			
				return AVLNode.insertRebalance(newNode, self, beforeInsertHeightParent)


			else:
				#find Predecessor of currentNode_rank_i_plus_1

				Predecessor=currentNode_rank_i_plus_1.getPredecessor()

				beforeInsertHeightParent=Predecessor.getHeight()

				#make newNode its right child

				Predecessor.setRight(newNode)
				newNode.setParent(Predecessor)

				self.size += 1

				# update the size and the height after the insert before rotations

				# THIS IS THE ROOT
				if Predecessor.getParent() == None:
					Predecessor.setSize(Predecessor.getSize() + 1)

					Predecessor.setHeight(max(Predecessor.getRight().getHeight(),
															Predecessor.getLeft().getHeight()) + 1)
				else:


					while Predecessor.getParent() != None:
						Predecessor.setSize(Predecessor.getSize() + 1)
						Predecessor.setHeight(max(Predecessor.getRight().getHeight(),Predecessor.getLeft().getHeight()) + 1)

						Predecessor = Predecessor.getParent()

				return AVLNode.insertRebalance(newNode, self, beforeInsertHeightParent)

	"""update list length based on associated tree

	"""
	def updateLength(self):
		self.size = self.root.getSize()
 
	"""deletes the i'th item in the list

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	@Help Functions: _rebalance, _deletion, retrieveNode, updateAttr
 	@Complexity: O(log n).
	"""
	def delete(self, i):
		targetNode = self.retrieveNode(i)
		if targetNode is None:
			return -1
		deletedNode = self._deletion(targetNode) # Returns the deleted node location (Might be a Virtual Node)
		temp = self._rebalance(deletedNode.getParent()) # Start rebalancing from the first node that might be a BF criminal, the deleted node parent 
		self.updateAttr()
		return temp


	"""returns the value of the first item in the list

	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""
	def first(self):
		if self.firstNode is None:
			return None
		return self.firstNode.getValue()

	"""returns the value of the last item in the list

	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""
	def last(self):
		if self.lastNode is None:
			return None
		return self.lastNode.getValue()

	"""returns an array representing list 

	@rtype: list
	@returns: a list of strings representing the data structure
	@Help function: inOrderAppendToListWalk
	@Complexity: O(n)
	"""
	def listToArray(self):
		return [node.getValue() for node in self.inOrderAppendToListWalk()]


	"""returns the size of the list 

	@rtype: int
	@returns: the size of the list
	"""
	def length(self):
		return  self.size

	"""sort the info values of the list

	@rtype: list
	@returns: an AVLTreeList where the values are sorted by the info of the original list.
	@Help function: inOrderFillByArraySortWalk, createEmptyAvlTree
	@Complexity: O(nlogn)
	"""
	def sort(self):
		temp_tree = AVLTreeList.createEmptyAvlTree(self.length())
		value_list = self.listToArray()
		temp_tree.inOrderFillByArraySortWalk(value_list)
		return temp_tree


	"""permute the info values of the list 

	@rtype: list
	@returns: an AVLTreeList where the values are permuted randomly by the info of the original list.
	@Help function: inOrderFillByArrayPermutationWalk, createEmptyAvlTree
	@Complexity: O(n)
	"""
	def permutation(self):
		temp_tree = AVLTreeList.createEmptyAvlTree(self.length())
		value_list = self.listToArray()
		temp_tree.inOrderFillByArrayPermutationWalk(value_list)
		return temp_tree

	"""concatenates lst to self

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	@Help function: _concatHelper0, _concatHelper1, _concatHelper2, updateAttr
	@Complexity: O(logn)
	"""
	def concat(self, lst):
		temp = self.getHeight() - lst.getHeight()
		if lst.getHeight() == -1: # Edge case: the appended tree-list is empty - we don't change anything.
			return abs(temp)
		elif self.getHeight() == -1: # Edge case: the original tree-list is empty (and the appended is not) - we change pointers of original tree list attr to appended tree-list attr.
			self.root = lst._getRoot()
			self.updateLength()
			self.firstNode=lst.firstNode
			self.lastNode=lst.lastNode
			return abs(temp)
		else: # Check between which subtrees (root nodes) we will perform join with _concatHelper0.
			if abs(temp) <= 1: # in this case between both whole trees (main node roots)
				self._concatHelper0(self._getRoot(), lst._getRoot())
			else:
				if temp < 0: # in this case between original tree-list main root and a sub root of the appended tree-list (root of subtree), such that their height distance is 1 or less.
					root_node2 = AVLTreeList._concatHelper1(lst._getRoot(), self.getHeight())
					self._concatHelper0(self._getRoot(), root_node2, root_node2.getParent(), root_node2, lst._getRoot())
				else: # in this case between appended tree-list main root and a sub root of the original tree-list (root of subtree), such that their height distance is 1 or less.
					root_node1 = AVLTreeList._concatHelper2(self._getRoot(), lst.getHeight())
					self._concatHelper0(root_node1, lst._getRoot(), root_node1.getParent(), root_node1, self._getRoot())
			self.updateAttr()
			return abs(temp)

	"""create Real Node with default value (None) and connect given nodes.

	@type root_node1: AVLNode
	@type root_node2: AVLNode
	@type parent: AVLNode
	@type child: AVLNode
	@type actual_root: AVLNode
	@Help function: updateSizeBySons, _deletion, _rebalance
	"""
	def _concatHelper0(self, root_node1, root_node2, parent=None, child=None, actualRoot=None):
		x = AVLNode() # The temporary node which we will use to connect both root as it's children, it will be deleted at the end of the function.
		x.convertToRegularNode(None) # Must be real for deletion to work
		x.setParent(parent)
		if parent is not None: # Determine which child (left or right) x should be to the given 'parent' AVLNode (If we pass parent we also must pass it's child that x will "replace", can be Virtual)
			if child.isLeftChildOf(parent):
				parent.setLeft(x)
			else:
				parent.setRight(x)
		x.setLeft(root_node1)
		root_node1.setParent(x)
		x.setRight(root_node2)
		root_node2.setParent(x)
		if actualRoot is None: # In case one of the tree roots is sub root, we also must pass a root pointer to update self instance.
			actualRoot = x
		self.root = actualRoot
		x.updateSizeBySons()
		x.updateHeightBySons()
		deletedNode = self._deletion(x) # After x was successfully joined with the trees and self was updated accordingly to include both trees' nodes, remove x
		self._rebalance(deletedNode.getParent()) # rebalance after x removal
  
	"""Static function - get "left-most" subtree in tree, in which it's height is in distance 1 or less from given int.

	@type lstRootNode: AVLNode
	@type selfHeight: int
	@rtype: AVLNode
	@returns: root of target subtree
	@Complexity: O(logn)
	"""
	def _concatHelper1(lstRootNode, selfHeight):
		temp = lstRootNode
		while not temp.isVirtualNode():
			if abs(selfHeight - temp.getHeight()) <= 1:
				return temp
			temp = temp.getLeft()

	"""Static function - get "right-most" subtree in tree, in which it's height is in distance 1 or less from given int.

	@type lstRootNode: AVLNode
	@type selfHeight: int
	@rtype: AVLNode
	@returns: root of target subtree
	@Complexity: O(logn)
	"""
	def _concatHelper2(selfRootNode, lstHeight):
		temp = selfRootNode
		while not temp.isVirtualNode():
			if abs(temp.getHeight() - lstHeight) <= 1:
				return temp
			temp = temp.getRight()
  
	"""searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    @Help functions: listToArray 
    @Complexity: O(n)
	"""
	def search(self, val):
		if self.size==0:
			return -1
		else:
			orderedList = self.listToArray()  # O(n)
			i=0
			while i < len(orderedList): # O(n)
				if orderedList[i]==val:
					return i
				i+=1
			return -1
	
	"""returns the root of the tree representing the list

	@rtype: AVLNode
	@returns: the root, None if the list is empty
	"""
	def getRoot(self):
		temp = self.root
		if temp.isVirtualNode():
			return None
		return temp
	
	"""returns the root of the tree representing the list

	@rtype: AVLNode
	@returns: the root (Including "Virtual Root")
	"""
	def _getRoot(self):
		return self.root

	"""Used in "insert" function and in "insertRebalance" function to update the root after rotations
	@rtype: AVLNode
	@returns: the root, None if the list is empty
	@Complexity : O(logn)
	"""
	def updateRoot(self):
		if self.root.getParent() != None:#this is not a root so we have to update it
			while self.root.getParent() != None:
				self.root = self.root.getParent()
			return self.root
		else:
		    return self.root
