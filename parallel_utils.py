import contextlib
import sys
from datetime import datetime
from itertools import product
from time import time
from typing import List, Optional, Tuple, Union

import numpy as np
from joblib import delayed, Parallel, parallel
from tqdm.auto import tqdm


def normalized_time(time_seg: float) -> str:
    """
    Normalize the time to a human-readable format
    """
    if time_seg < 60:
        return f'{time_seg:.2f} s'
    elif time_seg < 3600:
        mins, sec = divmod(time_seg, 60)
        return f'{int(mins)} min {sec:.2f} s'
    else:
        hours, time_seg = divmod(time_seg, 3600)
        mins, sec = divmod(time_seg, 60)
        return f'{int(hours)} h {int(mins)} min {sec:.2f} s'


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _log(message: str, verbose: bool, hostname: Optional[str] = None) -> None:
    if verbose:
        # Compute timestamp
        time_stamp = datetime.now().strftime("%H:%M:%S, %d/%m/%Y")
        msg = f'{time_stamp}: {message}'
        if hostname is not None:
            msg = f'[{hostname}]: {msg}'

        print(msg, flush=True)


def _parameter_run_joblib(fun: callable, param_names: List[List[str]], param_values: List[List[List]],
                          indices_iterate: list, desc, n_workers: Optional[int] = -1, pbar_bool: bool = True,
                          leave: Optional[bool] = True, **fun_kwargs) -> Tuple[List[List], int]:
    n_total = len(indices_iterate)

    if n_workers == 1:  # If only one worker, do not use joblib
        result = []
        pbar = tqdm(indices_iterate, desc=desc, leave=leave, disable=not pbar_bool)
        for index in pbar:
            result.append(fun(**{**_get_iteration(index, param_names, param_values), **fun_kwargs}))
    else:
        with tqdm_joblib(tqdm(indices_iterate, desc=desc, leave=leave, disable=not pbar_bool)) as _:
            result = Parallel(n_jobs=n_workers)(
                delayed(fun)(**{**_get_iteration(index, param_names, param_values), **fun_kwargs}) for index in
                indices_iterate)

    n_output = len(result[0]) if type(result[0]) is tuple else 1
    if n_output == 1:
        result = [(result_i,) for result_i in result]

    result_temp = []
    for i in range(n_output):
        result_temp.append([result[j][i] for j in range(n_total)])

    return result_temp, n_output


def _parameter_run_mpi(fun: callable, param_names: List[List[str]], param_values: List[List[List]],
                       indices_iterate: list, desc: str, pbar_bool: bool = True, verbose: Optional[bool] = False,
                       leave: Optional[bool] = True, **fun_kwargs) -> Tuple[List[List], int]:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_total = len(indices_iterate)
    indices_compute = np.array_split(np.arange(n_total), size)[rank]
    indices_compute = np.array(indices_iterate)[indices_compute]

    pbar = tqdm(indices_compute, desc=desc, leave=leave, disable=not (pbar_bool and rank == 0), file=sys.stdout,
                dynamic_ncols=True)

    n_outputs = 0
    result = []
    if 'Open MPI v5.0' in MPI.Get_library_version():
        mpi_flush = True
    else:
        mpi_flush = False

    for index in pbar:
        if pbar_bool and rank == 0 and mpi_flush:
            print('\r')  # Needed to show the progress bar in some mpi versions

        result_temp = fun(**{**_get_iteration(index, param_names, param_values), **fun_kwargs})

        if type(result_temp) is not tuple:
            result_temp = (result_temp,)

        # Check how many parameters are returned
        if len(result) == 0:  # First iteration
            n_outputs = len(result_temp)
            for _ in range(n_outputs):
                result.append([])

        for i, result_temp_i in enumerate(result_temp):
            result[i].append(result_temp_i)

    _log(f'Rank {rank} finished', verbose, hostname=MPI.Get_processor_name())

    # Notify root when each worker is done, and show a progress bar
    if rank == 0:
        progress = 0
        pbar = tqdm(total=size - 1, desc='Waiting for workers', leave=leave, disable=not pbar_bool, file=sys.stdout)

        while progress < size - 1:
            if pbar_bool and mpi_flush:
                print(f'\r', flush=True)  # Needed to show the progress bar in some mpi versions

            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=0):
                comm.recv(source=MPI.ANY_SOURCE, tag=0)  # Receive a signal from each worker
                progress += 1
                pbar.update(1)

        pbar.close()
    else:
        comm.send(1, dest=0, tag=0)  # Notify root that the work is done

    n_outputs = comm.bcast(n_outputs, root=0)
    if not result:
        result = [[] for _ in range(n_outputs)]

    results_gathered = []
    for i in range(n_outputs):
        _log(f'Gathering results {i}', verbose and rank == 0, hostname=MPI.Get_processor_name())

        result_gathered = comm.gather(result[i], root=0)

        _log('Results gathered', verbose and rank == 0, hostname=MPI.Get_processor_name())

        # Flatten the list
        if rank == 0:
            result_gathered = [item for sublist in result_gathered for item in sublist]
            results_gathered.append(result_gathered)

        _log('Results reshaped', verbose and rank == 0, hostname=MPI.Get_processor_name())

    return results_gathered, n_outputs


def _get_iteration(index, param_names, param_values):
    n_groups = len(param_values)
    n_parameters = [len(group) for group in param_values]

    dic_temp = {}
    for group_index in range(n_groups):
        for parameter_index in range(n_parameters[group_index]):
            dic_temp = {**dic_temp,
                        param_names[group_index][parameter_index]: param_values[group_index][parameter_index][
                            index[group_index]]}

    return dic_temp


def _format_input(param_names: Union[str, list], param_values: list) -> Tuple[List[List[str]], List[List[List]]]:
    # Check the depth of the parameters

    if type(param_names) is str:
        depth = 1
    elif type(param_names[0]) is str:
        depth = 2
        if type(param_values) is not list:
            raise ValueError('The parameters values must be a list of values')
    elif type(param_names[0][0]) is str:
        depth = 3
        if type(param_values[0]) is not list:
            raise ValueError('The parameters values must be a list of values')
    else:
        raise ValueError('Unknown input format for the parameters names')

    # Correct the input if the depth is not equal to 3
    if depth == 1:
        # Only one parameter is provided
        param_names = [[param_names]]
        param_values = [[param_values]]
    elif depth == 2:
        # Groups with a single parameters are provided
        param_names = [[param_name_i] for param_name_i in param_names]
        param_values = [[param_values_i] for param_values_i in param_values]

    # Check that the number of groups of parameters is the same as the number of groups of values
    if len(param_names) != len(param_values):
        raise ValueError('The number of groups of parameters must be the same as the number of groups of values')

    # Check that all parameters in the same group have the same length
    for group in param_values:
        if len(set([len(param) for param in group])) != 1:
            raise ValueError('All parameters in the same group must have the same length')

    return param_names, param_values


def parameter_run(fun: callable, param_names: Union[str, List[str], List[List[str]]],
                  param_values: Union[List, List[List], List[List[List]]], n_workers: Optional[int] = -1,
                  pbar_bool: bool = True, verbose: Optional[bool] = False, pbar_kwargs: Optional[dict] = None,
                  reshape: Optional[bool] = True, backend: Optional[str] = None, desc: Optional[str] = None,
                  **kwargs) -> Union[list, np.ndarray, None]:
    """
    Run a function with multiple parameters in parallel. To indentify the parameters of interest, user must provide its
    name as is written in the function definition. If more parameters should be pass to the function, they can be
    provided as kwargs. The backend used for the parallelization is chosen automatically between joblib and mpi4py.
    If mpi4py is used, the function must be run with mpirun -n n_workers python script.py. Finally, the result can be
    reshaped as a (hyper)matrix if the reshape parameter is set to True, if the functions return multiple values,
    the result is a list of arrays.

    The parameters can be grouped in different ways, the possible options are:
    - A single parameter: The function is run with a single parameter. In this case, param_names='param_name' and
    param_values=[value1, value2, ...]
    - Multiple parameters: The function is run with multiple parameters. In this case, param_names=['param_name1',
    'param_name2', ...] and param_values=[[value1, value2, ...], [value1, value2, ...], ...]
    - Multiple groups of parameters: The function is run with multiple groups of parameters, running the parameters in
    the same group at the same time. In this case, param_names=[['param_name1', 'param_name2', ...], ['param_name1',
    'param_name2', ...], ...] and param_values=[[[value1, value2, ...], [value1, value2, ...], ...], [[value1, value2,
    ...], [value1, value2, ...], ...], ...]. Note that in this case, the number of values for each parameter in the same
    group must be the same.

    Parameters
    ----------
    fun : callable
        Function to run in parallel.
    param_names : str or list
        Name of the parameters to run in parallel.
    param_values : list
        Values of the parameters to run in parallel.
    n_workers : int, optional (default=-1)
        Number of workers to use in parallel. If -1, the number of workers is the maximum number of cores available.
    pbar_bool : bool, optional (default=True)
        If True, show a progress bar.
    backend : str, optional (default=None)
        Backend to use for the parallelization. If None, the backend is chosen automatically between joblib and mpi.
    verbose : bool, optional (default=False)
        If True, print information about the parallelization.
    pbar_kwargs : dict, optional (default=None)
        Dictionary with the parameters to pass to the tqdm such as desc and leave.
    reshape : bool, optional (default=True)
        If True, reshape the result as a (hyper)matrix. Sometimes, due to the nature of the output, the reshape is not
        possible, in this case, the result is not reshaped.
    desc: str, optional (default=None)
        Description of the progress bar. If None, the description denotes the function and parameters names.
    kwargs :
        Additional parameters to pass to the function.

    Returns
    -------
    result : list or np.ndarray
        Result of the function run in parallel. The shape of the (hyper)matrix is (n_values1, n_values2, ...,
         n_valuesN), where n_valuesI is the number of values in the I-th group.
    """

    if pbar_kwargs is None:
        pbar_kwargs = {}

    time_start = time()

    param_names, param_values = _format_input(param_names, param_values)

    # Create the list of dictionaries with the parameters in a nested loop
    n_values = [len(group[0]) for group in param_values]
    indices = [list(range(n_value)) for n_value in n_values]
    indices_iterate = product(*indices)

    # Choose the backed
    if backend is not None:
        available_backends = ['joblib', 'mpi']
        if backend not in available_backends:
            raise ValueError(f'Unknown backend {backend}. Available backends are {available_backends}')

    size = 1
    rank = 0

    if backend is None or backend == 'mpi':
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
        except ImportError:
            pass

        if size == 1 and backend is None:
            backend = 'joblib'
        else:
            backend = 'mpi'

    if desc is None:
        param_names_pbar = [parameters_names_i for parameters_names_i in param_names]
        desc = f'{fun.__name__}: {param_names_pbar}'

        if len(desc) > 40:
            desc = desc[:40] + '(...)]'

    # Execute the parallel run
    if backend == 'joblib':
        _log('Running under joblib', verbose)

        result, n_outputs = _parameter_run_joblib(fun, param_names, param_values, list(indices_iterate), desc,
                                                  n_workers=n_workers, pbar_bool=pbar_bool, **pbar_kwargs, **kwargs)

    elif backend == 'mpi':
        _log(f'Running under mpi with {size} workers', verbose and rank == 0, hostname=MPI.Get_processor_name())

        result, n_outputs = _parameter_run_mpi(fun, param_names, param_values, list(indices_iterate), desc,
                                               pbar_bool=pbar_bool, verbose=verbose, **pbar_kwargs, **kwargs)

    else:
        raise ValueError('Unknown backend')

    if rank == 0:
        # Reshape the result to get the (hyper)matrix
        if reshape:
            try:
                if n_outputs == 1:
                    result = np.array(result[0])
                else:
                    result = [np.array(result[i]) for i in range(n_outputs)]
            except ValueError:
                print('Could not reshape the result')

        total_time = time() - time_start
        _log(f'Total time: {normalized_time(total_time)}', verbose)

        return result
    else:
        return None
