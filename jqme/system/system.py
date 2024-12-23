from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import ishermitian


class System(ABC):
    '''Abstract class representing a general quantum system.

    Parameters
    ----------
    energies : float | list[float] | np.ndarray
        Energies of the system.
    couplings : float | list[float] | np.ndarray, optional
        Couplings between system elements. Default is None.
    '''

    def __init__(self, energies, couplings=None):
        self._quantities = {}
        self.energies = energies
        self.couplings = couplings

    @property
    def energies(self):
        '''Energies of the system.'''
        return self._quantities['energies']

    @energies.setter
    def energies(self, energies):
        self._quantities['energies'] = np.atleast_1d(energies)
        self._quantities['system_size'] = self._quantities['energies'].size

    @property
    def system_size(self):
        '''Size of the system (number of elements).'''
        return self._quantities.get('system_size', 0)

    @property
    def couplings(self):
        '''Coupling matrix of the system.'''
        return self._quantities['couplings']

    @couplings.setter
    def couplings(self, couplings):
        if couplings is None:
            self._quantities['couplings'] = np.zeros((self.system_size,
                                                      self.system_size))
        else:
            self._quantities['couplings'] = self._get_couplings(couplings)

    def _get_couplings(self, couplings):
        '''Prepare and validate the coupling matrix.'''
        if np.isscalar(couplings) and self.system_size == 2:
            couplings = np.array([[0, couplings], [couplings, 0]])
        elif isinstance(couplings, list):
            couplings = np.atleast_2d(couplings)
        if not isinstance(couplings, np.ndarray):
            raise ValueError('Couplings in wrong format.')
        couplings -= np.diag(np.diag(couplings))  # Remove self-coupling
        if not ishermitian(couplings):
            raise ValueError('Coupling matrix is not Hermitian.')
        return couplings

    @property
    def state(self):
        '''State of the system.'''
        return self._quantities.get('state', None)

    @state.setter
    def state(self, state):
        self._quantities['state'] = state

    def _validate_system(self):
        '''Validate system parameters.'''
        try:
            valid = self.energies.size == self.couplings.shape[0] == self.system_size
        except KeyError:
            valid = False
        return valid

    @abstractmethod
    def hamiltonian(self):
        '''Method to compute the Hamiltonian of the system.'''

    @abstractmethod
    def set_state(self, state_type, state):
        '''Abstract method to set the system's state.'''

    def to_dict(self):
        '''Convert system properties to a dictionary format.'''
        init_dict = self.__dict__
        new_dict = {'class': self.__class__.__name__}
        for key, value in init_dict.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            new_key = key[1:] if key.startswith('_') else key
            new_dict[new_key] = value
        return new_dict
