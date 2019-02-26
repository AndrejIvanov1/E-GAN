import tensorflow as tf
from tensorflow.python.training import slot_creator
#from trainer.optimizer import Optimizer

def _var_key(var):
	# TODO(ashankar): Consolidate handling for eager and graph
	if hasattr(var, "op"):
		return (var.op.graph, var.op.name)

	#print("Returning _unique id: ", var._unique_id)
	return var._unique_id  # pylint: disable=protected-access

class CustomAdamOptimizer(tf.train.AdamOptimizer):

	"""
	def __init__(self):
		super()
		self._saved = False
	"""

	def _create_slots(self, var_list):
		# Create the beta1 and beta2 accumulators on the same device as the first
		# variable. Sort the var_list to make sure this device is consistent across
		# workers (these need to go on the same PS, otherwise some updates are
		# silently ignored).

		first_var = min(var_list, key=lambda x: x.name)
		self._create_non_slot_variable(
		    initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
		self._create_non_slot_variable(
		    initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

	    # Create slots for the first and second moments.
		for v in var_list:
			self._zeros_slot(v, "m", self._name)
			self._zeros_slot(v, "v", self._name)

		#self._saved = True

	def _zeros_slot(self, var, slot_name, op_name):
	    """Find or create a slot initialized with 0.0.
	    Args:
	      var: A `Variable` object.
	      slot_name: Name for the slot.
	      op_name: Name to use when scoping the Variable that
	        needs to be created for the slot.
	    Returns:
	      A `Variable` object.
	    """
	    #print(var)
	    #input()
	    #var._unique_id = var.name
	    named_slots = self._slot_dict(slot_name)
	    #print("Name: ", var.name)
	    #print("Unique id: ", var._unique_id)
	    if _var_key(var) not in named_slots:
	      new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
	      self._restore_slot_variable(
	          slot_name=slot_name, variable=var,
	          slot_variable=new_slot_variable)
	      named_slots[_var_key(var)] = new_slot_variable
	    #return named_slots[_var_key(var)]
