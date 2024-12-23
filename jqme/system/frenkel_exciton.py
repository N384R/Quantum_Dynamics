from typing import Literal
from itertools import combinations
import numpy as np
from qutip import Qobj, basis, tensor, qzero, zero_ket
from .system import System
from .operators import SZ, SM, SP, I


class FrenkelExciton(System):
    '''Class representing a Frenkel-exciton system.

    Parameters
    ----------
    energies : float | list[float] | np.ndarray
        Energies of the system.
    couplings : float | list[float] | np.ndarray, optional
        Couplings between system elements. Default is None.
    dipole_moments : float | list[float] | np.ndarray, optional
        Dipole moments of the system. Default is None.
    '''

    def __init__(self,
                 energies: float | list[float] | np.ndarray,
                 couplings: float | list[float] | np.ndarray = None,
                 dipole_moments: float | list[float] | np.ndarray = None,
                 ):
        super().__init__(energies, couplings)
        self.dipole_moments = dipole_moments
        self.is_valid = self._validate_system()

    @property
    def dipole_moments(self):
        '''Dipole moments of the system.'''
        return self._quantities['dipoles']

    @dipole_moments.setter
    def dipole_moments(self, dipole_moments):
        if dipole_moments is None:
            self._quantities['dipoles'] = np.ones(self.system_size)
        else:
            self._quantities['dipoles'] = np.atleast_1d(dipole_moments)

    def _validate_system(self):
        '''Check if the system parameters are valid.'''
        return super()._validate_system() and self.dipole_moments.size == self.system_size

    @property
    def state_type(self):
        '''Type of the state.'''
        return self._quantities.get('state_type', None)

    @state_type.setter
    def state_type(self, state_type):
        self._quantities['state_type'] = state_type

    def hamiltonian(self):
        '''Compute the Frenkel-exciton Hamiltonian.'''
        if not self.is_valid:
            raise ValueError('System parameters are not valid.')

        H = qzero([2] * self.system_size)  # Initialize Hamiltonian as zero

        # Energy terms
        for i in range(self.system_size):
            H += -0.5 * self.energies[i] * self._apply_tensor(SZ, i)

        # Coupling terms
        for i, j in combinations(range(self.system_size), 2):
            H += (self.couplings[i, j] * (self._apply_tensor(SM, j) @ self._apply_tensor(SP, i)
                                          + self._apply_tensor(SP, j) @ self._apply_tensor(SM, i)))
        return H

    def _apply_tensor(self, op, pos):
        '''Helper function to apply an operator at a specific position in the tensor product.'''
        return tensor(*[I] * (self.system_size - pos - 1), op, *[I] * pos)

    def set_state(self,
                  state_type: Literal['state',
                                      'delocalized excitation',
                                      'localized excitation',
                                      'ground',
                                      ] = 'ground',
                  state: list | np.ndarray | int = 0,
                  ):
        '''Sets the state of the system based on the given type.'''
        if not self.is_valid:
            raise ValueError('System parameters are not valid.')

        if state_type == 'ground':
            state = 0
        elif state_type == 'localized excitation':
            if not isinstance(state, int) or not 0 <= state < self.system_size:
                raise ValueError(
                    'State must be an integer within the system size.')
        elif state_type in ('state', 'delocalized excitation'):
            state = np.array(state, dtype=complex)
            expected_size = 2**self.system_size if state_type == 'state' else self.system_size
            if state.size != expected_size:
                raise ValueError('State must match the system size.')
            norm = np.sum(np.abs(state)**2)
            if not np.isclose(norm, 1):
                state /= np.sqrt(norm)
            state = state.tolist()
        else:
            raise ValueError('Invalid state type.')

        self.state_type = state_type
        self.state = state

    def get_e_state(self):
        '''Returns the ket state of the system.
        '''
        if not self.is_valid:
            raise ValueError('System is not valid.')
        system_dims = [2] * self.system_size
        if self.state_type == 'ground':
            return basis(system_dims, [0] * self.system_size)
        if self.state_type == 'state':
            return Qobj(np.array(self.state),
                        dims=[system_dims, [1] * self.system_size],
                        type='ket',
                        )
        if self.state_type == 'delocalized excitation':
            state = zero_ket(dimensions=system_dims)
            for nc, c in enumerate(self.state):
                binary = bin(1 << nc)[2:].zfill(self.system_size)
                position = list(map(int, binary))
                state += c * basis(system_dims, position)
            return state
        if self.state_type == 'localized excitation':
            binary = bin(1 << self.state)[2:].zfill(self.system_size)
            position = list(map(int, binary))
            return basis(system_dims, position)
        raise ValueError('Invalid state type.')
