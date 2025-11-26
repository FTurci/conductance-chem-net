# Base class for constructing a module
#  It need to have a bare-bone init that allows for easy combinations such as
#  A = B + C
#  (this should calculate the conductance of A from the conductance of B and C)
#  by using a dedicated __add_ method that implements the  operations
# that JW is using in his code

# You can have separate classes if that helps, you can try inherritance (but I do not see a clear case for it for now)

