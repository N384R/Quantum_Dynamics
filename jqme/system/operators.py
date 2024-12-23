from qutip import sigmaz, identity, destroy

SZ = sigmaz()     # Pauli Z matrix
SM = destroy(2)    # Annihilation operator
SP = destroy(2).dag()  # Creation operator
I = identity(2)    # Identity matrix
