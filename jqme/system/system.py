'''Module to define a quantum system.
'''

from typing import Literal
from itertools import combinations
import numpy as np
from scipy.linalg import ishermitian
from qutip import (Qobj, sigmaz, identity, destroy,
                   basis, zero_ket, qzero, tensor)


class System:
    '''Class to define a quantum system.

    Parameters
    ----------
    energies : float | list[float] | np.ndarray
        Energies of the system.
    couplings : float | list[float] | np.ndarray, optional
        Couplings of the system. Default is None.
    dipole_moments : float | list[float] | np.ndarray, optional
        Dipole moments of the system. Default is None.
    '''

    def __init__(self,
                 energies: float | list[float] | np.ndarray,
                 couplings: float | list[float] | np.ndarray = None,
                 dipole_moments: float | list[float] | np.ndarray = None,
                 ):
        self._validity = False
        self._quantities = {}
        self.energies = energies
        self.couplings = couplings
        self.dipole_moments = dipole_moments

    @property
    def energies(self):
        '''Returns the energies of the system.
        '''
        return self._quantities['energies']

    @property
    def system_size(self):
        '''Returns the size of the system.
        '''
        return self._quantities['system_size']

    @property
    def couplings(self):
        '''Returns the couplings of the system.
        '''
        return self._quantities['couplings']

    @property
    def dipole_moments(self):
        '''Returns the dipole moments of the system.
        '''
        return self._quantities['dipoles']

    @staticmethod
    def check_validity(func):
        '''Decorator to check the validity of the system.
        '''

        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            try:
                if (self.energies.size == self.dipole_moments.size
                        == self.couplings.shape[0] == self.system_size):
                    self._validity = True
                else:
                    self._validity = False
            except KeyError:
                self._validity = False
            return result
        return wrapper

    @energies.setter
    def energies(self, energies):
        self._quantities['energies'] = np.atleast_1d(energies)
        self._quantities['system_size'] = self.energies.size

    @couplings.setter
    @check_validity
    def couplings(self, couplings):
        if couplings is None:
            self._quantities['couplings'] = np.zeros((self.system_size,
                                                      self.system_size))
        else:
            if np.isscalar(couplings) and self.system_size == 2:
                couplings = np.array([[0, couplings], [couplings, 0]])
            elif isinstance(couplings, list):
                couplings = np.atleast_2d(couplings)
            if not isinstance(couplings, np.ndarray):
                raise ValueError('Couplings in wrong format.')
            couplings -= np.diag(np.diag(couplings))
            if not ishermitian(couplings):
                raise ValueError('Coupling matrix is not Hermitian.')
            self._quantities['couplings'] = couplings

    @dipole_moments.setter
    @check_validity
    def dipole_moments(self, dipole_moments):
        if dipole_moments is None:
            self._quantities['dipoles'] = np.ones(self.system_size)
        else:
            self._quantities['dipoles'] = np.atleast_1d(dipole_moments)

    @property
    def state(self):
        '''Returns the state of the system.
        '''
        return self._quantities['state']

    @state.setter
    def state(self, state):
        self._quantities['state'] = state

    @property
    def state_type(self):
        '''Returns the type of the state.
        '''
        return self._quantities['state_type']

    @state_type.setter
    def state_type(self, state_type):
        self._quantities['state_type'] = state_type

    def set_state(self,
                  state_type: Literal['state', 'delocalized excitation',
                                      'localized excitation', 'ground'] = 'ground',
                  state: list | np.ndarray | int = 0,
                  ):
        '''Sets the state of the system.
        '''
        if not self._validity:
            raise ValueError('System is not valid.')
        if state_type == 'ground':
            state = 0
        elif state_type == 'localized excitation':
            if (not isinstance(state, int) or
                    not 0 <= state < self.system_size):
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

    def to_dict(self) -> dict:
        init_dict = self.__dict__
        new_dict = {'class': self.__class__.__name__}
        for key, value in init_dict.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            new_key = key[1:] if key.startswith('_') else key
            new_dict[new_key] = value
        return new_dict

    def get_e_Hamiltonian(self):
        '''Returns the Frenkel-exciton Hamiltonian.
        '''
        def apply_tensor(op, pos):
            return tensor(*[I] * (self.system_size - pos - 1), op, *[I] * pos)
        if not self._validity:
            raise ValueError('System is not valid.')
        sz = sigmaz()
        sm = destroy(2)
        sp = destroy(2).dag()
        I = identity(2)
        H = qzero(dimensions=[2]*self.system_size)

        for i in range(self.system_size):
            H += -0.5 * self.energies[i] * apply_tensor(sz, i)

        for i, j in combinations(range(self.system_size), 2):
            H += self.couplings[i, j] * (apply_tensor(sm, j) @ apply_tensor(sp, i)
                                         + apply_tensor(sp, j) @ apply_tensor(sm, i))
        return H

    def get_e_state(self):
        '''Returns the ket state of the system.
        '''
        if not self._validity:
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
