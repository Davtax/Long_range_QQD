import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy.sparse as sp
import sympy
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
from scipy.integrate import romb
from tqdm.auto import tqdm

MAPPING = {'↑': 0, '↓': 1, 'up': 0, 'down': 1}
LINE_STYLES = ['-', '--', ':', '-.']
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def a(N: int, j: int, sigma: Union[str, int]):
    """
    Compute the spinful fermionic annihilation operator for a chain of N sites

    Parameters
    ----------
    N: int
        Number of sites
    j: int
        Index of site
    sigma: str or int
        Spin of the particle

    Returns
    -------
    qt.operator
    """

    if type(sigma) is str:
        sigma = MAPPING[sigma]

    if sigma not in (0, 1):
        raise ValueError(f'The spin index must be 0=↑, or 1=↓. The provided value is {sigma}.')

    j = N - j - 1  # The first site is the leftmost one in the basis

    return qt.operators.fdestroy(2 * N, 2 * j + sigma)


def a_dag(N: int, j: int, sigma: Union[str, int]):
    return a(N, j, sigma).dag()


def num(N: int, j: int, sigma: Optional[Union[str, int]] = None):
    n_s = 0

    if sigma is None:
        sigmas = [0, 1]
    else:
        sigmas = [sigma]

    for sigma in sigmas:
        a_sigma = a(N, j, sigma)
        n_s += a_sigma.dag() * a_sigma

    return n_s


class Basis:
    def __init__(self, basis_labels_binary: List[str]):
        self.basis_labels_binary = basis_labels_binary
        self.L = len(self.basis_labels_binary[0].split('|')[0])
        self.arrows = self.to_arrows(latex=False)

    def __repr__(self) -> str:
        temp = ''
        for i, label in enumerate(self.basis_labels_binary):
            temp += f'{i}: {label}\n'
        return temp

    def __len__(self) -> int:
        return len(self.basis_labels_binary)

    def __getitem__(self, item: Union[int, List[int]]) -> Union[str, list[str], None]:
        if type(item) is int:
            return self.arrows[item]
        elif type(item) is list:
            return [self.arrows[i] for i in item]
        return None

    def index(self, up: str, down: Optional[str] = None) -> int:
        """
        If no down state is provided, it is assumed that the convention used is up|down, otherwise, the up and down
        states are separated.
        """
        if down is None:
            if '|' in up:
                up, down = up.split('|')
            else:
                raise ValueError('No down state provided.')

        target_state = up + '|' + down

        try:
            index = self.basis_labels_binary.index(target_state)
            return index
        except ValueError:
            raise ValueError('The desired state {} not found in the basis.'.format(target_state))

    def to_arrows(self, latex: Optional[bool] = True, show_indices: Optional[bool] = False) -> Union[str, List[str]]:
        arrows = self._quspin2arrows(self.basis_labels_binary, latex)

        if show_indices:
            arrows = [(i, arrow) for (i, arrow) in enumerate(arrows)]

        return arrows

    def _quspin2arrows(self, state: Union[str, List[str]], latex: Optional[bool] = True) -> Union[str, List[str]]:
        """
        Change from the convention of quspin for the state labels to a ket with arrows written in LaTeX for matplotlib
        The convention for the state is: |spin up population_ss>|spin down population_ss>. The population_ss for each spin, and
        each state can only take the values {0, 1}.

        For example:
        |1 0 1>|0 1 1>    ------>    |↑, ↓, ↑↓>

        Parameters
        ----------
        state: str
            State label in the quspin convention
        latex: bool, optional, default=True
            If false, write the states with ascii, instead of latex

        Returns
        -------
        state_str: str
            State label in a LaTeX format
        """

        if type(state) is list:
            labels = []
            for label in state:
                labels.append(self._quspin2arrows(label, latex=latex))
            return labels

        up, down = state.split('|')

        state_str = '|'
        if latex:
            state_str = r'$' + state_str

        for i in range(self.L):
            particle = False
            if up[i] == '1':
                if latex:
                    state_str += r'\uparrow'
                else:
                    state_str += '↑'
                particle = True
            if down[i] == '1':
                if latex:
                    state_str += r'\downarrow'
                else:
                    state_str += '↓'
                particle = True

            if not particle:
                state_str += '0'

            if i != self.L - 1:
                state_str += ', '

        if latex:
            state_str += r'\rangle$'
        else:
            state_str += '>'

        return state_str


class Hamiltonian:
    def __init__(self, name: str, weights_change_basis: List[dict] = None, convert_to_qutip: bool = True,
                 hbar: float = 1, indices_keep: Optional[List[int]] = None, basis_labels: Optional[List[str]] = None,
                 generator: Optional[callable] = None, **parameters):
        self.L = None  # Liuovillian
        self.name = name

        self._convert_to_qutip = convert_to_qutip
        self._hbar = hbar

        self._parameters = parameters

        self._indices_keep = indices_keep

        if generator is not None:
            self._generator = generator

        self._hamiltonians_computed = 0

        if basis_labels is None:
            self.basis = None
        else:
            self.basis = Basis(basis_labels)

        self.D = len(self.basis)

        if weights_change_basis is not None:
            self._change_basis = self._create_change_basis_matrix(weights_change_basis)
            self._change_basis_inv = np.linalg.inv(self._change_basis)
            self._change_basis_bool = True
        else:
            self._change_basis_bool = False

        self._H0 = None
        self._create_hamiltonian(**parameters)

    def __call__(self, **parameters):
        if parameters is not None:
            self._create_hamiltonian(**parameters)

        self._hamiltonians_computed += 1

        return self._H0

    def __repr__(self):
        return self._H0.__repr__()

    def _create_hamiltonian(self, **parameters) -> np.ndarray:
        if self._H0 is not None:
            prev_diag = np.diag(self._H0[:])
        else:
            prev_diag = None

        self._parameters = {**self._parameters, **parameters}
        self._H0 = self._generator(**self._parameters) / self._hbar

        if self._indices_keep is not None:
            self._H0 = self._H0[self._indices_keep, :]
            self._H0 = self._H0[:, self._indices_keep]

        if self._change_basis_bool:
            self._H0 = self._change_basis @ self._H0 @ self._change_basis_inv

        self._energies = np.diag(self._H0[:]).real

        if self._convert_to_qutip:
            # self._H0 = qt.Qobj(self._H0)
            self._H0 = qt.Qobj(self._H0).to('CSR')

        # The energies have changed, so the Liouvillian must be updated
        if prev_diag is not None and self.L is not None:
            incoherent_parameters = ['Gammas', 'T1', 'T2', 'mus', 'beta', 'Deltas']
            condition_diagonal = not np.allclose(prev_diag, np.diag(self._H0[:]))
            condition_incoherent = any(x in incoherent_parameters for x in self._parameters)
            if condition_diagonal or condition_incoherent:
                self._create_Liouvillian()
        return self._H0

    def get_operator(self, op) -> np.ndarray:
        if self._indices_keep is not None:
            op = op[self._indices_keep, :]
            op = op[:, self._indices_keep]

        if self._change_basis_bool:
            op = self._change_basis @ op @ self._change_basis_inv
        return op

    def _generator(self, **parameters) -> np.ndarray:
        pass

    def compute_intensity(self, rho: Union[np.ndarray, qt.Qobj]) -> float:
        pass

    def _create_Liouvillian(self):  # To be implemented in the child classes
        pass

    def _create_change_basis_matrix(self, weights: List[dict]) -> np.ndarray:
        """
        Create the matrix that changes the basis of the Hamiltonian to the one defined by the weights. The weights are
        dictionaries with the keys being the states in the new basis and the values the weights.

        Parameters
        ----------
        weights: List[dict]
            List of dictionaries with the new basis states and the weights. For example: [{'uudd': 1}, {'ud0u': 1}]

        Returns
        -------
        U: np.ndarray
            Matrix that changes the basis of the Hamiltonian as U @ H @ U^(-1)

        """
        U = np.zeros((self.D, self.D), dtype=complex)
        for i, weight in enumerate(weights):
            for state in weight.keys():
                state_bin = letters_2_binary(state)

                index = self.basis.index(state_bin)
                U[i, index] = weight[state]

        return U


class Hamiltonian1D(Hamiltonian):
    """
        Hamiltonian of a spinless 1D chain of N sites with nearest-neighbor spin-conserving tunneling.

        Parameters
        ----------
        N: int
            Number of sites
        eps: np.ndarray or float
            On-site energies. If eps is a float, all sites have the same on-site energy.
        tau: np.ndarray or float
            Nearest-neighbor tunneling. If taus is a float, all neighbors have the same tunneling.
        U: np.ndarray or float
            On-site interaction. If U is a float, all sites have the same on-site interaction.

        Returns
        -------
        H: np.ndarray
            Hamiltonian of the system.
    """

    def __init__(self, convert_to_qutip: bool = True, hbar: float = 1, indices_keep: Optional[List[int]] = None,
                 states_keep: Optional[List[str]] = None, states_remove: Optional[List[str]] = None,
                 indices_remove: Optional[List[int]] = None, states_add: Optional[List[str]] = None,
                 indices_add: Optional[List[int]] = None, n_particles: Optional[Union[int, List[int]]] = None,
                 intensity_contribution=None, weights_change_basis: List[dict] = None, **parameters):

        self.N = parameters['N']  # Number of sites
        self.basis_labels = self._create_basis_labels_binary()
        self.D = len(self.basis_labels)  # Dimension of the Hilbert space

        if n_particles is None:
            n_particles = list(range(2 * self.N + 1))

        self.n_particles_all = [base.count('1') for base in self.basis_labels]

        if type(n_particles) is int:
            n_particles = [n_particles]

        if max(self.n_particles_all) < max(n_particles):
            raise ValueError(
                f'The maximum number of particles in the basis ({max(self.n_particles_all)}) is smaller than the '
                f'maximum number of particles requested ({max(n_particles)}).')

        if states_keep is not None:
            indices_keep = self._check_indices(indices_keep, states_keep)
            indices_add = []
            indices_remove = []
        else:
            indices_add = self._check_indices(indices_add, states_add)
            indices_remove = self._check_indices(indices_remove, states_remove)

        indices_keep = self.generate_indices_keep(indices_keep, indices_add, indices_remove, n_particles)

        self.n_particles_states = []
        for basis_label in self.basis_labels:
            ups, downs = basis_label.split('|')
            self.n_particles_states.append(np.array(list(ups), dtype=int) + np.array(list(downs), dtype=int))

        super().__init__('Spinful 1D chain', convert_to_qutip=convert_to_qutip, hbar=hbar, indices_keep=indices_keep,
                         basis_labels=self.basis_labels, weights_change_basis=weights_change_basis, **parameters)

        incoherent_terms = ['Gammas', 'T1', 'T2']

        self._spin_flip_matrix = []
        for m in range(self.D):
            self._spin_flip_matrix.append([])
            for n in range(self.D):
                result = self._check_spin_flip(m, n)
                self._spin_flip_matrix[m].append(result)

        self._contact_coupling_plus = np.zeros((self.N, self.D, self.D), dtype=bool)
        self._contact_coupling_minus = np.zeros_like(self._contact_coupling_plus)
        for l in range(self.N):
            for m in range(self.D):
                for n in range(self.D):
                    self._contact_coupling_plus[l, m, n] = self._check_contact_coupling(l, m, n, -1)
                    self._contact_coupling_minus[l, m, n] = self._check_contact_coupling(l, m, n, 1)

        if any(x in incoherent_terms for x in parameters):
            self._create_Liouvillian()
        else:
            self.Gammas_plus = None
            self.Gammas_minus = None
            self.Gammas = None
            self.Lambdas = None
            self.L = None

        self.intensity_contribution = intensity_contribution

    def _check_indices(self, indices: Optional[List[int]] = None, states: Optional[List[str]] = None) -> List[int]:
        if indices is None:
            indices = []

        if states is not None:
            for state in states:
                index = self.basis_labels.index(state)
                if index not in indices:
                    indices.append(index)

        return indices

    def _create_basis_labels_binary(self, indices_keep: Optional[List[int]] = None) -> List[str]:
        basis_labels_binary = []
        for i in range(2 ** (2 * self.N)):
            temp = bin(i)[2:].zfill(2 * self.N)
            basis_labels_binary.append(temp[::2][::-1] + '|' + temp[1::2][::-1])

        if indices_keep is not None:
            basis_labels_binary = [basis_labels_binary[i] for i in indices_keep]

        return basis_labels_binary

    def generate_indices_keep(self, indices_keep: List[int], indices_add: List[int], indices_remove: List[int],
                              n_particles: Optional[Union[int, List[int]]] = None) -> List[int]:
        """The priority is to remove states, then to add states, and finally to keep a given number of particles."""
        final_indices_keep = []

        # Check indices for number of particles
        if n_particles is not None:
            if indices_keep is None:
                indices_keep = list(range(self.D))

            if type(n_particles) is int:
                n_particles = [n_particles]

            for i, n in enumerate(self.n_particles_all):
                if n in n_particles and i in indices_keep:
                    final_indices_keep.append(i)

        if not final_indices_keep:  # If final_indices_keep is empty
            final_indices_keep = list(range(self.D))

        # Add new states if desired
        for index in indices_add:
            if index not in final_indices_keep:
                final_indices_keep.append(index)

        # Remove states if desired
        for index in indices_remove:
            if index in final_indices_keep:
                final_indices_keep.remove(index)

        self.basis_labels = self._create_basis_labels_binary(final_indices_keep)  # Reset basis labels
        self.n_particles_all = [base.count('1') for base in self.basis_labels]
        return final_indices_keep

    def _generator(self, N: int, eps: np.ndarray, tau: np.ndarray, U: np.ndarray, Vs: np.ndarray = None,
                   Deltas: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if isinstance(eps, (int, float)):
            eps = np.full(N, eps)

        if isinstance(tau, (int, float)):
            tau = np.full(N - 1, tau)

        if isinstance(U, (int, float)):
            U = np.full(N, U)

        if Deltas is None:
            Deltas = np.zeros(N)

        if Vs is None:
            Vs = np.zeros((N, N))

        if np.shape(Vs) != (N, N):
            Vtemp = np.zeros((N, N))
            for i in range(N):
                vals = [Vs[i]] * (N - i)
                Vtemp += np.diag(vals, i) + np.diag(vals, -i)

            Vs = Vtemp

        H = num(N, 0) * 0  # Initialize the Hamiltonian with zeros

        for i in range(N):
            H += eps[i] * num(N, i)
            H += U[i] * num(N, i, '↑') * num(N, i, '↓')
            H += Deltas[i] / 2 * (num(N, i, '↑') - num(N, i, '↓'))

            for j in range(i + 1, N):
                H += Vs[i, j] * num(N, i) * num(N, j)

            if i < N - 1:
                for sigma in ['↑', '↓']:
                    H += tau[i] * (a_dag(N, i, sigma) * a(N, i + 1, sigma) + a_dag(N, i + 1, sigma) * a(N, i, sigma))

        return H[:]

    def _generate_open_operators(self):
        self.Gammas_plus = np.zeros((self.D, self.D), dtype=float)
        self.Gammas_minus = np.zeros_like(self.Gammas_plus)
        self.Lambdas = np.zeros_like(self.Gammas_plus)
        spin_relaxation = np.zeros_like(self.Gammas_plus, dtype=float)

        if 'Gammas' in self._parameters:
            Gammas = self._parameters['Gammas']
        else:
            Gammas = np.zeros(self.N)
        if 'mus' in self._parameters:
            mus = self._parameters['mus']
        else:
            mus = np.zeros(self.N)
        if 'beta' in self._parameters:
            beta = self._parameters['beta']
        else:
            beta = np.inf

        if 'T1' in self._parameters:
            T1 = self._parameters['T1']
            if 'Deltas' in self._parameters:
                Deltas = self._parameters['Deltas']
                if isinstance(Deltas, (int, float)):
                    Deltas = np.full(self.N, Deltas)
                elif isinstance(Deltas, list):
                    Deltas = np.array(Deltas)
            else:
                Deltas = np.zeros(self.N)
            exponentials = np.exp(-Deltas * self._parameters['beta'])

        else:
            T1 = np.inf

        if 'T2' in self._parameters:
            T2 = self._parameters['T2']
        else:
            T2 = np.inf

        for m in range(len(self.basis_labels)):
            for n in range(len(self.basis_labels)):
                self.Gammas_plus[m, n], self.Gammas_minus[m, n] = _generate_Gammas(self.N, m, n, Gammas, mus, beta,
                                                                                   self._contact_coupling_plus,
                                                                                   self._contact_coupling_minus,
                                                                                   self._energies)

                if T1 != np.inf:
                    result = self._spin_flip_matrix[m][n]
                    if result is not False:
                        index, flip_type = result
                        if flip_type == 'up':
                            W = 1 / ((1 + exponentials[index]) * T1)
                        elif flip_type == 'down':
                            W = exponentials[index] / ((1 + exponentials[index]) * T1)

                        spin_relaxation[m, n] += W

        self.Gammas = self.Gammas_plus + self.Gammas_minus + spin_relaxation

        for m in range(len(self.basis_labels)):
            for n in range(len(self.basis_labels)):
                self.Lambdas[m, n] = _generate_Lambdas(self.Gammas, m, n, T2)

    def _create_Liouvillian(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self._generate_open_operators()

        L = sp.lil_matrix((self.D ** 2, self.D ** 2), dtype=complex)

        for m in range(self.D):
            for n in range(self.D):
                if m == n:
                    for k in range(self.D):
                        if k != n:
                            L[m + self.D * n, k + self.D * k] += self.Gammas[n, k]
                            L[m + self.D * n, n + self.D * n] -= self.Gammas[k, n]
                else:
                    L[m + self.D * n, m + self.D * n] -= self.Lambdas[m, n]

        L = qt.Qobj(L, dims=[[[self.D], [self.D]], [[self.D], [self.D]]])
        self.L = L / self._hbar

    def _check_contact_coupling(self, l: int, m: int, n: int, change: int) -> bool:
        """
        Check if the contact coupling between the states m and n is allowed by the change of particles in the site l.

        First, check if all the sites except l have the same state. Then, check if the number of particles in the site l
        is correct.

        Parameters
        ----------
        l: Dot where the coupling is happening
        m: State 1
        n: State 2
        change: Number of particles that change from m to n
        """

        state_1 = self.basis_labels[m]
        state_2 = self.basis_labels[n]

        # Remove the indices of the coupled site l (spin up and spin down)
        state_1 = state_1[:l] + state_1[l + 1:l + self.N + 1] + state_1[l + self.N + 2:]
        state_2 = state_2[:l] + state_2[l + 1:l + self.N + 1] + state_2[l + self.N + 2:]

        if state_1 != state_2:
            return False

        # Change the particle number in the site l
        diff_particles = np.zeros(self.N)
        diff_particles[l] = change

        return np.array_equal(self.n_particles_states[m], self.n_particles_states[n] + diff_particles)

    def compute_intensity(self, rho: Union[np.ndarray, qt.Qobj]) -> float:
        if self.intensity_contribution is None:
            raise ValueError('The intensity contribution has not been defined.')

        I = 0

        for m in range(self.D):
            for n in range(self.D):
                if self.Gammas_plus[m, n] != 0 or self.Gammas_minus[m, n] != 0:
                    index_change = np.nonzero(self.n_particles_states[m] != self.n_particles_states[n])[0][0]
                    I += (self.Gammas_plus[m, n] - self.Gammas_minus[m, n]) * rho[n, n] * self.intensity_contribution[
                        index_change]
        return I.real

    def _check_spin_flip(self, index_i: int, index_f: int) -> Union[bool, Tuple[int, str]]:
        state_i = self.basis_labels[index_i]
        state_f = self.basis_labels[index_f]

        up_i, down_i = np.array([list(temp) for temp in state_i.split('|')], dtype=int)
        up_f, down_f = np.array([list(temp) for temp in state_f.split('|')], dtype=int)

        # Check if the states have the same number of particles
        if np.sum(up_i + down_i) != np.sum(up_f + down_f):
            return False

        delta_S = (up_f - down_f) - (up_i - down_i)
        indices = np.nonzero(delta_S)[0]

        if len(indices) != 1:
            return False
        else:
            if delta_S[indices[0]] == 2:
                return indices[0], 'up'
            elif delta_S[indices[0]] == -2:
                return indices[0], 'down'
            else:
                return False

    def generate_sympy(self, short_diagonals: bool = False) -> sympy.Matrix:
        convert_to_qutip_temp = self._convert_to_qutip
        self._convert_to_qutip = False

        N = self.N

        epss = [sympy.Symbol(f'epsilon{i + 1}') for i in range(N)]
        taus = [sympy.Symbol(f'tau{i + 1}') for i in range(N - 1)]
        Us = [sympy.Symbol(f'U{i + 1}') for i in range(N)]
        Vss = [sympy.Symbol(f'V{i}') for i in range(N)]
        Vss[0] = 0

        parameters = {'N': N, 'eps': [0] * N, 'Vs': [0] * N, 'U': [0] * N, 'tau': [0] * (N - 1)}

        hamiltonian = sympy.Matrix(self._create_hamiltonian(**parameters))
        for site in range(N):
            Us_temp = [0] * N
            Us_temp[site] = 1
            new_parameters = {**parameters, 'U': Us_temp}

            hamiltonian = hamiltonian + Us[site] * sympy.Matrix(self._create_hamiltonian(**new_parameters))

            Vs_temp = [0] * N
            Vs_temp[site] = 1
            new_parameters = {**parameters, 'Vs': Vs_temp}

            hamiltonian = hamiltonian + Vss[site] * sympy.Matrix(self._create_hamiltonian(**new_parameters))

            epss_temp = [0] * N
            epss_temp[site] = 1
            new_parameters = {**parameters, 'eps': epss_temp}

            hamiltonian = hamiltonian + epss[site] * sympy.Matrix(self._create_hamiltonian(**new_parameters))

        for site in range(N - 1):
            taus_temp = [0] * N
            taus_temp[site] = 1
            new_parameters = {**parameters, 'tau': taus_temp}

            hamiltonian = hamiltonian + taus[site] * sympy.Matrix(self._create_hamiltonian(**new_parameters))

        self._convert_to_qutip = convert_to_qutip_temp

        if short_diagonals:
            hamiltonian = hamiltonian - sympy.diag(*hamiltonian.diagonal())

            energies = []
            for i in range(self.D):
                charge = ''.join(str(number) for number in self.n_particles_states[i])
                energies.append(sympy.Symbol(f'E{charge}'))

            energies = np.diag(energies)
            if self._change_basis_bool:
                hamiltonian = hamiltonian + sympy.Matrix(self._change_basis @ energies @ self._change_basis_inv)

        return hamiltonian


class Hamiltonian1DVirtual(Hamiltonian1D):
    def __init__(self, convert_to_qutip: bool = True, hbar: float = 1,
                 n_particles: Optional[Union[int, List[int]]] = None, **kwargs):
        super().__init__(convert_to_qutip=convert_to_qutip, hbar=hbar, n_particles=n_particles, **kwargs)
        self.name = 'Spinfull 1D chain with virtual states'

    def _generator(self, **parameters):
        Vs = parameters['V_i']
        alpha = parameters['alpha']
        eps_0 = parameters['eps_0']

        eps = alpha @ Vs + eps_0
        parameters = {**parameters, 'eps': eps}
        return super()._generator(**parameters)


@njit
def _generate_Gammas(N: int, m: int, n: int, Gammas: np.ndarray, mus: np.ndarray[float], beta: float,
                     contact_coupling_plus: np.ndarray, contact_coupling_minus: np.ndarray,
                     energies: np.ndarray[float]) -> Tuple[float, float]:
    Gamma_plus = 0.
    Gamma_minus = 0.

    for l in range(N):
        if contact_coupling_minus[l, m, n]:
            Gamma_minus += Gammas[l] * fermi_dirac_distribution((energies[m] - energies[n]).real, mus[l], beta)

        if contact_coupling_plus[l, m, n]:
            Gamma_plus += Gammas[l] * (1 - fermi_dirac_distribution((energies[n] - energies[m]).real, mus[l], beta))

    return Gamma_plus, Gamma_minus


def letters_2_binary(state: str) -> str:
    """
    Convert a state in letters to a binary representation. The letters are 'u', 'd', and 's' for up, down, and singlet.
    For example, 'uds' is converted to '101|011'.

    Parameters
    ----------
    state: str
        State in letters.

    Returns
    -------
    str
        State in binary representation.
    """
    n = len(state)
    ups = ['0'] * n
    downs = ['0'] * n

    for site, particle in enumerate(state):
        if particle == 'u':
            ups[site] = '1'
        elif particle == 'd':
            downs[site] = '1'
        elif particle == 's':
            ups[site] = '1'
            downs[site] = '1'
    return ''.join(ups) + '|' + ''.join(downs)


@njit
def _generate_Lambdas(Gammas: np.ndarray, m: int, n: int, T2: float) -> float:
    Lambda = (np.sum(Gammas[:, m] + Gammas[:, n])) / 2
    Lambda -= (Gammas[m, m] + Gammas[n, n]) / 2 - 1 / T2

    return Lambda


@njit
def fermi_dirac_distribution(energy: float, mu: float, beta: float) -> float:
    """
    Fermi-Dirac distribution.

    Parameters
    ----------
    energy: float
        Energies.
    mu: float
        Chemical potential.
    beta: float
        Inverse temperature.

    Returns
    -------
    f: float
        Fermi-Dirac distribution.
    """

    return 1 / (np.exp(beta * (energy - mu)) + 1)


class Simulation:
    def __init__(self, hamiltonian: Hamiltonian):
        self.hamiltonian = hamiltonian
        self.evolution = None

        self.population = None  # dims: (n_time, n_states)
        self.time = None

    def compute_dynamics(self, psi_0: qt.Qobj, time: np.ndarray, pbar: Optional[bool] = False,
                         qt_kw: Optional[dict] = None, new_diag: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        self.time = time

        if 'c_ops' in kwargs:
            c_ops = kwargs.pop('c_ops')
            if not isinstance(c_ops, list):
                c_ops = [c_ops]
        else:
            c_ops = []

        H0 = self.hamiltonian(**kwargs)
        L = self.hamiltonian.L

        if new_diag is not None:
            H0 = H0 + qt.qdiags(new_diag - H0.diag())

        # print(H0.diag())

        if L is not None:
            c_ops.append(L)

        # print('Computing dynamics...')
        if qt_kw is None:
            qt_kw = {}
        psi_t = qt.mesolve(H0, psi_0, time, c_ops=c_ops, options={**{'progress_bar': pbar}, **qt_kw}).states

        if len(c_ops) > 0:
            rho_t = np.array(psi_t)
            self.evolution = np.array([rho_i[:] for rho_i in rho_t])
            self.population = np.diagonal(self.evolution, axis1=1, axis2=2).real
        else:
            self.evolution = np.array([psi_i[:] for psi_i in psi_t])[:, :, 0]
            self.population = np.abs(self.evolution) ** 2

        return self.evolution

    def compute_steadystate(self, ss_kw: Optional[dict] = None, **kwargs) -> qt.Qobj:
        if 'c_ops' in kwargs:
            c_ops = kwargs.pop('c_ops')
            if not isinstance(c_ops, list):
                c_ops = [c_ops]
        else:
            c_ops = []

        H = qt.liouvillian(self.hamiltonian(**kwargs))

        if ss_kw is None:
            ss_kw = {}

        rho_ss = qt.steadystate(H + self.hamiltonian.L, c_ops=c_ops, **ss_kw)
        return rho_ss

    def compute_intensity_integrated(self, t1: float, t2: float, qt_kw: dict, **kwargs) -> Tuple[
        float, float, np.ndarray]:
        n_time = 100
        time = np.append(np.linspace(0, t1, n_time)[:-1], np.linspace(t1, t2, n_time))

        # psi_0 = qt.basis(self.hamiltonian.D, 0)
        H0 = self.hamiltonian(**kwargs)
        # psi_0 = H0.groundstate()[1]
        psi_0 = H0.eigenstates()[1][0]

        # qt_kw = {'method': 'diag', 'eigensolver_dtype': 'csr'}
        # qt_kw = {'method': 'diag'}

        self.compute_dynamics(psi_0, time, qt_kw=qt_kw, pbar=True, **kwargs)

        final_population = self.population[-1]

        current_t = [self.hamiltonian.compute_intensity(rho_t) for rho_t in self.evolution]

        integrated_current = np.trapz(current_t[n_time - 1:], time[n_time - 1:]) / (t2 - t1)
        average_population = np.trapz(self.population[n_time - 1:], time[n_time - 1:], axis=0) / (t2 - t1)
        average_population_dots = np.sum(average_population * self.hamiltonian.n_particles_all)

        return integrated_current, average_population_dots, final_population

    def compute_current_ss(self, ss_kw: Optional[dict] = None, **kwargs) -> Tuple[float, float, np.ndarray]:
        """Current through the system in steady state."""

        try:
            rho_ss = self.compute_steadystate(ss_kw=ss_kw, **kwargs)
        except ValueError:
            print('Steadystate not converged')
            # return np.nan, np.nan, np.zeros(self.hamiltonian.D) * np.nan
            return 0, 0, np.zeros(self.hamiltonian.D)

        average_population = np.sum(np.diagonal(rho_ss[:]).real * self.hamiltonian.n_particles_all)
        population_ss = np.diagonal(rho_ss[:]).real

        current = self.hamiltonian.compute_intensity(rho_ss)
        return current, average_population, population_ss

    def compute_sensor_response(self, tf: float, distances_sensor: np.ndarray, psi_0: qt.Qobj = None) -> float:
        """
        Compute the sensor current in a closed system in order to compute the stability diagram.

        Parameters
        ----------
        tf: float
            Total integration time.
        distances_sensor: np.ndarray
            Distances of the sensor to each site. Array of shape (N,).
        psi_0: qt.Qobj (optional, default None)
            Initial state of the system. If None, the ground state of the Hamiltonian in absence of coupling is used.

        Returns
        -------
        sensor: float
            Sensor current.
        """
        if psi_0 is None:
            psi_0 = qt.Qobj(np.diag(self.hamiltonian().diag())).groundstate()[1]

        time = np.linspace(0, tf, 2 ** 9 + 1)
        self.compute_dynamics(psi_0, time)

        integrand = np.sum(self.population / distances_sensor[None, :], axis=1)
        sensor = 1 / time[-1] * romb(integrand, dx=time[1] - time[0])

        return sensor

    def plot_dynamics(self, **kwargs):
        if self.evolution is None:
            raise ValueError('Evolution is not computed yet.')

        if self.hamiltonian.basis is not None and 'basis_labels' not in kwargs:
            kwargs['basis_labels'] = self.hamiltonian.basis

        return plot_dynamics(self.population, self.time, **kwargs)

    def reach_steadystate(self, tf: Optional[int] = 1000, threshold: Optional[float] = 1e-3, **kwargs) -> Tuple[
        np.ndarray, np.ndarray]:
        rho_ss = self.compute_steadystate()[:]

        nsteps = 100
        exponent = 1.3

        self.evolution = [qt.basis(self.hamiltonian.D, 0)]
        rho_t = np.empty((0, self.hamiltonian.D, self.hamiltonian.D))
        time_total = np.empty(0)

        # Check if the dynamics reaches closely the steady state
        counter = 0
        while True:
            try:
                t_init = tf * exponent ** counter * (counter > 0)
                t_final = tf * exponent ** (counter + 1)
                time = np.linspace(t_init, t_final, nsteps)
                self.compute_dynamics(qt.Qobj(self.evolution[-1]), time)
            except qt.IntegratorException:
                nsteps *= 2
                print('Integrator exception, increasing nsteps', end='\r')
                continue

            rho_t = np.append(rho_t, self.evolution, axis=0)
            time_total = np.append(time_total, time)

            error = np.linalg.norm(self.evolution[-1] - rho_ss)
            if error < threshold:
                break
            else:
                print(f'Step {counter} not converged, error {error}', end='\r')
                counter += 1

        return rho_t, time_total

    def check_unpopulated_states(self, tf: Optional[int] = 1000, threshold_ss: Optional[float] = 1e-3,
                                 threshold_population: Optional[float] = 1e-3) -> Tuple[np.ndarray, List[int]]:
        rho_t, _ = self.reach_steadystate(tf, threshold_ss)

        rho_t = np.array(rho_t).reshape(-1, self.hamiltonian.D, self.hamiltonian.D)
        population = np.diagonal(rho_t, axis1=1, axis2=2).real

        # Check if the population_ss of the states is below a threshold
        unpopulated_states = []
        for i in range(self.hamiltonian.D):
            if np.max(population[:, i]) < threshold_population:
                unpopulated_states.append(i)

        return rho_t, unpopulated_states

    def compute_time_to_ss(self, **kwargs) -> Tuple[float, float, float]:
        _, time = self.reach_steadystate(**kwargs)
        t_ss = float(time[-1])

        current, avg_pop, _ = self.compute_current_ss(**kwargs)

        return t_ss, current, avg_pop

    def population_dot(self, population_states: np.ndarray, n_dot: int) -> np.ndarray:
        """Compute the population_ss of a given dot, given the population_ss of the states and the number of particles in each
         for each state. The axis of the population_states is (... , n_states). The parameter n_particles is usually
         given by the parameter inside the Hamiltonian class, as hamiltonian.n_particles_all. Each element of the list
         has the form [n_0, n_1, ..., n_N-1], where n_i is the number of particles in the i-th dot. The parameter n_dot
         is the index of the dot of interest.
         """
        population_n_dot = [n_state[n_dot] for n_state in self.hamiltonian.n_particles_states]
        return np.sum(population_states * population_n_dot, axis=-1)

    def sensor_response(self, distances: np.ndarray, population_states: np.ndarray):
        response = 0
        for i, distance in enumerate(distances):
            response += self.population_dot(population_states, i) / distance
        return response


def plot_dynamics(population: np.ndarray, time: np.ndarray, x_label: str = 'time', y_label: str = 'population',
                  basis_labels: Union[bool, List[str], Basis] = None, ax: plt.axis = None, min_population: float = 0,
                  legend: bool = True, **kwargs) -> Union[None, Tuple[plt.figure, plt.axis]]:
    """
    Plot the dynamics of a system.
    Parameters
    ----------
    population: ndarray
        Population with shape (len(time), dim), where dim is the Hilbert space dimension
    time: ndarray
        Vector of times
    x_label: str, optional, default='time'
        Label for the x axes
    y_label: str, optional, default='population_ss'
        Label for the y axes
    basis_labels: list[str], optional
        Labels for the basis elements. If not provided, a default [0, 1, ...] legend is printed
    ax: matplotlib.axes, optional
        Axis where plot the dynamics. If not provided, a new figure with axes is created
    min_population: float, optional, default=0
        Only plot states with some given minimum population_ss
    legend: bool, optional, default=True
        If True, print the legend
    """

    n_basis = population.shape[1]

    if basis_labels is None or basis_labels is True:
        basis_labels = [str(i) for i in range(n_basis)]
    elif basis_labels is False:
        basis_labels = [None] * n_basis
    elif type(basis_labels) is Basis:
        basis_labels = basis_labels.to_arrows()

    if ax is None:
        fig, ax = plt.subplots()
        return_fig = True
    else:
        fig = None
        return_fig = False

    if 'c' in kwargs:
        colors = kwargs.pop('c')
        if isinstance(colors, str):
            colors = [colors]
    else:
        colors = COLORS

    counter = 0
    for i in range(n_basis):
        if np.max(population[:, i]) > min_population:

            if 'ls' in kwargs:
                ax.plot(time, population[:, i], label=basis_labels[i], color=colors[counter % len(colors)], **kwargs)
            else:
                ax.plot(time, population[:, i], label=basis_labels[i],
                        ls=LINE_STYLES[(counter // len(colors)) % len(LINE_STYLES)],
                        color=colors[counter % len(colors)], **kwargs)
            counter += 1

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xlim(min(time), max(time))
    ax.set_ylim(0, 1)

    if legend:
        ax.legend()

    if return_fig:
        return fig, ax


def sort_bands(energies: np.ndarray, modes: np.ndarray, tolerance: Optional[float] = 0.1,
               progress_bar: Optional[bool] = False, window: Optional[int] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort the bands of the system to avoid changes in the color of the bands when no coupling is detected. The sorting is
    done by comparing the energies of the bands in each step and swapping the bands if the difference is smaller than
    the tolerance. The modes are also swapped to keep the correspondence between the energies and the modes.

    Parameters
    ----------
    energies: ndarray (n_x, n)
        Energies of the bands to sort. The energies[:, i] corresponds to the i-th band.
    modes: ndarray (n_x, n, n)
        Modes corresponding to the energies. The modes[:, :, i] corresponds to the i-th band.
    tolerance: float, optional (default=0.1)
        Tolerance to consider that two bands are too close, so they will check if they should be swapped.
    progress_bar: bool, optional (default=False)
        Show a progress bar for the sorting.
    window: int, optional (default=1)
        Number of steps to look for the bands to swap. If the window is greater than 1, the bands are swapped with the
        band that has the smallest distance in the window. Using a window greater than 1 speed up the sorting.

    Returns
    -------
    energies: ndarray (n_x, n)
        Sorted energies of the bands.
    modes: ndarray (n_x, n, n)
        Sorted modes corresponding to the energies.
    """
    n_x = len(energies)  # Number of steps for the independent variable
    n = np.shape(energies)[-1]  # Number of bands

    pbar = tqdm(range(n_x - 1), desc='Sorting energies', disable=not progress_bar)

    distances_energies = np.ones((n_x, n, n)) * np.inf
    for i in range(n):
        for j in range(i + 1, n):
            distances_energies[:, i, j] = np.abs(energies[:, i] - energies[:, j])

    distances_energies = np.min(distances_energies, axis=(1, 2))

    for i in pbar:
        if np.any(distances_energies[i: np.min([i + window, n_x])] < tolerance):
            swaps = []
            indices_swaps = []
            for j in range(n):
                if j not in indices_swaps:
                    distance_modes = np.zeros(n)
                    for k in range(n):
                        if np.abs(energies[i, j] - energies[i, k]) < tolerance:
                            if k not in indices_swaps:
                                distance_modes[k] = _compute_distance_modes(modes[i, :, j], modes[i + 1, :, k])
                    temp_swap = np.argmax(distance_modes)
                    if temp_swap != j:
                        swaps.append((j, temp_swap))
                        indices_swaps.append(temp_swap)
                    indices_swaps.append(j)
            for swap in swaps:
                energies, modes = _swap_bands(energies, modes, swap, i + 1)

    return energies, modes


def _swap_bands(energies: np.ndarray, modes: np.ndarray, swap_indices: Tuple[int, int], index_0: int) -> Tuple[
    np.ndarray, np.ndarray]:
    indices = np.arange(np.shape(energies)[-1])
    indices[list(swap_indices)] = swap_indices[::-1]
    energies_temp = energies[:, indices]
    modes_temp = modes[:, :, indices]

    energies = np.append(energies[:index_0], energies_temp[index_0:], axis=0)
    modes = np.append(modes[:index_0], modes_temp[index_0:], axis=0)

    return energies, modes


def _compute_distance_modes(mode_1: np.ndarray, mode_2: np.ndarray) -> float:
    return np.abs(np.sum(mode_1.conj() * mode_2))


def _create_equation_eps(E: str, U: np.ndarray, Vs: np.ndarray) -> sympy.Add:
    populations = [int(i) for i in E]
    eps_s = [sympy.symbols(f'e{i}') for i in range(len(populations))]

    energy = 0
    for i in range(len(populations)):
        energy += populations[i] * eps_s[i]

        if populations[i] == 2:
            energy += U[i]

        for j in range(i + 1, len(populations)):
            energy += populations[i] * populations[j] * Vs[j - i]

    return energy


def create_equation_virtual(E: str, U: np.ndarray, Vs: np.ndarray, alpha: np.ndarray, eps_0: np.ndarray) -> sympy.Add:
    eps_s = [sympy.symbols(f'e{i}') for i in range(len(E))]
    V_i = [sympy.symbols(f'V{i}') for i in range(len(E))]

    eq = _create_equation_eps(E, U, Vs)
    for i in range(len(E)):
        eq = eq.subs(eps_s[i], alpha[i, 0] * V_i[0] + alpha[i, -1] * V_i[-1] + eps_0[i])

    return eq


def _solve_resonance_inner(hamiltonian: Hamiltonian, E1: str, E2: str, v1_values: Optional[List[float]] = None) -> \
        Union[sympy.Add, List[float]]:
    eq1 = create_equation_virtual(E1, hamiltonian._parameters['U'], hamiltonian._parameters['Vs'],
                                  hamiltonian._parameters['alpha'], hamiltonian._parameters['eps_0'])
    eq2 = create_equation_virtual(E2, hamiltonian._parameters['U'], hamiltonian._parameters['Vs'],
                                  hamiltonian._parameters['alpha'], hamiltonian._parameters['eps_0'])

    solution = sympy.solve(eq1 - eq2, sympy.symbols(f'V{len(E1) - 1}'))[0]

    if v1_values is not None:
        result = [float(solution.subs(sympy.symbols('V0'), v1)) for v1 in v1_values]
        return result
    else:
        return solution


def solve_resonance(hamiltonian: Hamiltonian, resonances: Union[List[List[str]], List[str]],
                    V1_extremes: Optional[List[float]] = None) -> Union[sympy.Add, List[float], List[List[float]]]:
    if type(resonances[0]) is str:
        resonances = [resonances]

    solutions = []
    for resonance in resonances:
        E1 = resonance[0]
        E2 = resonance[1]
        solution = _solve_resonance_inner(hamiltonian, E1, E2, V1_extremes)
        solutions.append(solution)

    if len(solutions) == 1:
        return solutions[0]
    else:
        return solutions


def compute_map(simulation, run_parameter: list, run_name: str, x: np.ndarray, y: np.ndarray,
                n_workers: Optional[int] = -1, ss_kw: Optional[dict] = None, **kwargs) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    from parallel_utils import parameter_run

    result = parameter_run(simulation.compute_current_ss, run_name, run_parameter, n_workers=n_workers, ss_kw=ss_kw,
                           **kwargs)

    intensity = np.array(result[0]).reshape((len(y), len(x)))
    average_population = np.array(result[1]).reshape((len(y), len(x)))
    population = np.array(result[2]).reshape((len(y), len(x), -1))

    return intensity, average_population, population


def register_colormap(name_cmap: str, colors: List[str]):
    if name_cmap in colormaps():
        colormaps.unregister(name_cmap)
        colormaps.unregister(name_cmap + '_r')

    my_cmap = LinearSegmentedColormap.from_list(name_cmap, colors)
    my_cmap_r = my_cmap.reversed()

    colormaps.register(cmap=my_cmap)
    colormaps.register(cmap=my_cmap_r)

    return my_cmap
