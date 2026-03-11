"""
credits Nikolaos Patsalidis
"""

import numpy as np
from matplotlib import pyplot as plt

from time import perf_counter
from scipy.optimize import minimize #,dual_annealing
from numba import jit,prange
from numba.experimental import jitclass
import os
import sys
import inspect
import logging
import coloredlogs

from scipy.integrate import simpson
from pytrr import GroTrrReader
from scipy.optimize import dual_annealing,differential_evolution
import pandas as pd

import matplotlib

import lammpsreader

import pickle


@jit(nopython=True, fastmath=True)
def compute_residual(pars, A, y):
    """Compute RMS residual between linear model prediction and observed data.
    Parameters
    ----------
    pars : array-like, Model parameters (weights + bias)
    A : array-like, Input features (n_samples x n_features)
    y : array-like, Observed target values
    Returns
    -------
    float, RMS residual
    """
    w = pars[:-1]  # weights (all parameters except last entry)
    bias = pars[-1]  # scalar bias term
    r = np.dot(A, w) + bias - y  # residuals between model prediction and data
    return np.sqrt(np.sum(r*r) / y.size)  # root-mean-square residual


@jit(nopython=True, fastmath=True)
def dCRdw(pars, A, y):
    """Gradient of RMS residual w.r.t model parameters.
    Parameters
    ----------
    pars : array-like, Model parameters (weights + bias)
    A : array-like, Input features
    y : array-like, Observed target values
    Returns
    -------
    dw : array-like, Gradient of RMS residual
    """
    dw = np.zeros(pars.shape)  # gradient vector (same shape as parameters)
    n = y.size  # number of samples
    w = pars[:-1]  # weights
    bias = pars[-1]  # bias
    r = np.dot(A, w) + bias - y  # residual vector
    c = np.sqrt(np.sum(r*r) / n)  # current RMS residual (used for normalization)
    # compute gradient for weights
    for j in range(w.size):  # loop over weights
        for i in range(n):  # accumulate derivative over samples
            dw[j] += r[i] * A[i,j]
    # gradient for bias
    dw[-1] = np.sum(r)  # derivative w.r.t. bias is just sum of residuals
    dw *= 1.0 / n / c  # scale by number of samples and RMS residual
    return dw


@jit(nopython=True, fastmath=True)
def constraint(pars, A, y, residual):
    """Difference between target residual and current RMS residual.
    Parameters
    ----------
    pars : array-like, Model parameters (weights + bias)
    A : array-like, Input features
    y : array-like, Observed target values
    residual : float, Target RMS residual
    Returns
    -------
    float, Residual difference
    """
    return residual - compute_residual(pars, A, y)  # zero when constraint is satisfied


@jit(nopython=True, fastmath=True)
def dCdw(pars, A, y, residual):
    """Negative gradient of RMS residual w.r.t parameters.
    Parameters
    ----------
    pars : array-like, Model parameters (weights + bias)
    A : array-like, Input features
    y : array-like, Observed target values
    residual : float, Target RMS residual (not used)
    Returns
    -------
    dw : array-like, Negative gradient
    """
    return -dCRdw(pars, A, y)  # constraint gradient is negative residual gradient


@jit(nopython=True,fastmath=True)
def smoothness(pars,dlogtau):
    """Compute smoothness penalty (sum of squared second differences).
    Parameters
    ----------
    pars : array-like, Parameters including bias
    dlogtau : float, Scaling factor
    Returns
    -------
    float, Smoothness penalty
    """
    w = pars[:-1]  # exclude bias from smoothness penalty
    x = np.empty_like(w)  # second-difference vector
    n = w.size-1
    # interior points: discrete second derivative w[j+1] - 2*w[j] + w[j-1]
    for j in range(1,n):
        x[j] = w[j+1]-2*w[j]+w[j-1]
    # boundaries: one-sided approximations for the endpoints
    x[0] = w[2]-2*w[1]+w[0]
    x[n] = w[n]-2*w[n-1]+w[n-2]
    return np.sum(x*x)/dlogtau**3  # scale by dlogtau**3 for grid independence


@jit(nopython=True,fastmath=True)
def dSdw(pars,dlogtau):
    """Gradient of smoothness penalty.
    Parameters
    ----------
    pars : array-like, Parameters including bias
    dlogtau : float, Scaling factor
    Returns
    -------
    dw : array-like, Gradient array
    """
    w = pars[:-1]  # weights (excluding bias)
    dw = np.empty_like(pars)  # gradient including zero entry for bias
    # interior points: derivative of sum of squared second differences
    for j in range(2,w.size-2):
        dw[j] = 12*w[j]-8*(w[j+1]+w[j-1])+2*(w[j+2]+w[j-2])
    # boundary points: one-sided finite-difference derivatives
    dw[0] = 2*(w[2]-2*w[1]+w[0])
    dw[1] = 18*w[1]-12*w[2]-8*w[0]+2*w[3]
    n = w.size-1
    dw[n] = 2*(w[n-2]-2*w[n-1]+w[n])
    dw[n-1] = 18*w[n-1]-12*w[n-2]-8*w[n]+2*w[n-3]
    dw[:-1] /= dlogtau**3  # same scaling as in smoothness
    dw[-1] = 0  # bias gradient is zero by construction
    return dw


@jit(nopython=True,fastmath=True)
def Trelax(w,fi):
    """Sum of w*fi.
    Parameters
    ----------
    w : array-like, Weights
    fi : array-like, Relaxation factors
    Returns
    -------
    float
    """
    return np.sum(w*fi)


@jit(nopython=True,fastmath=True)
def Frelax(w,fi):
    """Sum of w/fi.
    Parameters
    ----------
    w : array-like, Weights
    fi : array-like, Relaxation factors
    Returns
    -------
    float
    """
    return np.sum(w/fi)


@jit(nopython=True,fastmath=True)
def FrelaxCost(w,fi):
    """Relaxation cost ignoring bias.
    Parameters
    ----------
    w : array-like, Weights including bias
    fi : array-like, Relaxation factors
    Returns
    -------
    float
    """
    return Frelax(w[:-1],fi)  # ignore last entry (bias) when evaluating cost


@jit(nopython=True,fastmath=True)
def dFdw(w,fi):
    """Gradient of FrelaxCost; zero for bias.
    Parameters
    ----------
    w : array-like, Weights including bias
    fi : array-like, Relaxation factors
    Returns
    -------
    dw : array-like
    """
    d = np.empty_like(w)  # gradient vector including bias slot
    d[:-1] = 1.0/fi  # derivative w.r.t. each weight (excluding bias)
    d[-1] = 0  # cost does not depend on bias explicitly
    return d


@jit(nopython=True,fastmath=True)
def FrelaxCon(w,fi,target):
    """Constraint: target minus sum(w/fi); bias ignored.
    Parameters
    ----------
    w : array-like, Weights including bias
    fi : array-like, Relaxation factors
    target : float
    Returns
    -------
    float
    """
    return target - np.sum(w[:-1]/fi)  # constraint is zero when sum(w/fi) == target


@jit(nopython=True,fastmath=True)
def dFCdw(w,fi,target):
    """Gradient of FrelaxCon; zero for bias.
    Parameters
    ----------
    w : array-like, Weights including bias
    fi : array-like, Relaxation factors
    target : float
    Returns
    -------
    dw : array-like
    """
    d = np.empty_like(w)  # gradient of constraint w.r.t. weights
    d[:-1] = -1.0/fi  # derivative of target - sum(w/fi)
    d[-1] = 0  # bias has no effect on the constraint
    return d


@jit(nopython=True,fastmath=True)
def L2(w):
    """Compute L2 norm squared: sum(w^2)/size.
    Parameters
    ----------
    w : array-like, Weights
    Returns
    -------
    float
    """
    return np.sum(w*w)/w.size  # mean-squared L2 norm


@jit(nopython=True,fastmath=True)
def dL2dw(w):
    """Gradient of L2 norm squared: 2*w/size.
    Parameters
    ----------
    w : array-like, Weights
    Returns
    -------
    dw : array-like
    """
    return 2*w/w.size  # derivative of mean-squared L2 norm


@jit(nopython=True,fastmath=True)
def FitCost(pars,dlogtau):
    """Compute smoothness-based fit cost.
    Parameters
    ----------
    pars : array-like, Parameters including bias
    dlogtau : float
    Returns
    -------
    float
    """
    return smoothness(pars,dlogtau)  # alias to keep optimizer API consistent


@jit(nopython=True,fastmath=True)
def dFitCostdw(pars,dlogtau):
    """Gradient of smoothness-based fit cost.
    Parameters
    ----------
    pars : array-like, Parameters including bias
    dlogtau : float
    Returns
    -------
    dw : array-like
    """
    return dSdw(pars,dlogtau)  # corresponding gradient wrapper




class logs():
    """Configure and expose a module-wide logger.

    This small helper class sets up a ``logging.Logger`` instance with
    both file and stream handlers and applies colored output via
    ``coloredlogs``. The resulting logger is stored on the instance
    (``self.logger``) and a single shared instance is created at the
    module level (``logger``).

    The intention is that all other classes and functions in this
    module use the same configured logger.
    """
    def __init__(self):
        self.logger = self.get_logger()

    def get_logger(self):

        LOGGING_LEVEL = logging.CRITICAL  # global verbosity threshold

        logger = logging.getLogger(__name__)  # module-level logger
        logger.setLevel(LOGGING_LEVEL)
        logger.propagate = False
        logFormat = '%(asctime)s\n[ %(levelname)s ]\n[%(filename)s -> %(funcName)s() -> line %(lineno)s]\n%(message)s\n --------'
        formatter = logging.Formatter(logFormat)  # shared formatter for file/stream


        logger.handlers.clear()

        stream = logging.StreamHandler(stream=sys.stdout)  # stdout stream
        stream.setLevel(LOGGING_LEVEL)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        fieldstyle = {'asctime': {'color': 'magenta'},
                      'levelname': {'bold': True, 'color': 'green'},
                      'filename':{'color':'green'},
                      'funcName':{'color':'green'},
                      'lineno':{'color':'green'}}

        levelstyles = {'critical': {'bold': True, 'color': 'red'},
                       'debug': {'color': 'blue'},
                       'error': {'color': 'red'},
                       'info': {'color':'cyan'},
                       'warning': {'color': 'yellow'}}

        coloredlogs.install(level=LOGGING_LEVEL,  # apply coloured formatting to logger
                            logger=logger,
                            fmt=logFormat,
                            datefmt='%H:%M:%S',
                            field_styles=fieldstyle,
                            level_styles=levelstyles)
        return logger

    def __del__(self):
        return


logobj = logs()
logger = logobj.logger



class supraClass():
    """High-level convenience wrapper around the analysis classes.

    This class hides the details of choosing between
    :class:`Analysis` and :class:`Analysis_Confined`, manages
    multiple-trajectory handling via :class:`multy_traj`, and exposes
    a small, user-friendly API for computing structural and dynamical
    properties.

    Parameters
    ----------
    topol_file : str
        Path to the topology / .gro or .dat file used to initialise the
        molecular system.
    connectivity_info : list or str
        Connectivity / (Sequence of .itp files, str of .itp file or .dat file force-field information passed on to the
        underlying :class:`Analysis` class.
    memory_demanding : bool, optional
        If ``True``, trajectories are streamed from disk instead of
        being kept entirely in memory.
    keep_frames : tuple of (int or None, int or None), optional
        Frame indices that delimit the time window to keep when
        reading trajectories.
    **kwargs
        Additional keyword arguments forwarded to :class:`Analysis` or
        :class:`Analysis_Confined`.
    """

    def __init__(self, topol_file, connectivity_info,
                 memory_demanding=False, keep_frames=(None,None), **kwargs):
        """Initialise the underlying analysis object and trajectory window.

        The appropriate analysis backend is selected based on the
        presence of the ``'conftype'`` keyword (confined vs unconfined
        systems). All other keyword arguments are forwarded untouched
        to the backend class.
        """
        # Choose the appropriate system class based on confinement type
        if 'conftype' not in kwargs:
            systemClass = Analysis
        else:
            systemClass = Analysis_Confined

        # Initialize MD analysis object
        self.mdobj = systemClass(topol_file, connectivity_info, memory_demanding, **kwargs)
        self.keep_frames = keep_frames  # store frames to keep

    def set_keep_frames(self, num_start=None, num_end=None):
        """Set the trajectory frame window to keep in memory."""
        self.keep_frames = (num_start, num_end)

    def get_property(self, trajf, funcname, *func_args, **func_kwargs):
        """Compute a property by applying a method over trajectories.

        Parameters
        ----------
        trajf : str or sequence of str
            Trajectory file(s) to process.
        funcname : str
            Name of the method of this class that is applied to each
            trajectory (e.g. ``"computeDynamics"``).
        *func_args, **func_kwargs
            Additional arguments forwarded to the selected method.

        Returns
        -------
        object
            Whatever object is returned by
            method: multy_traj.multiple_trajectory` for the chosen
            property ('func') function.
        """
        self.traj_files = trajf  # remember which files were used for potential deallocation
        func = getattr(self, funcname)  # method of supraClass used as callback for each trajectory
        # Dispatch over all trajectories with multy_traj, which handles looping and aggregation
        data = multy_traj.multiple_trajectory(trajf, func, *func_args, **func_kwargs)
        return data

    def dealloc_timeframes(self):
        """Delete cached trajectory frames to free memory if applicable."""
        try:
            # Only delete if multiple trajectory files were used; single-trajectory cases are cheap
            if type(self.traj_files) is not str:
                del self.mdobj.timeframes
                self.mdobj.timeframes = dict()  # reset to empty dictionary
        except:
            # Warning if unable to deallocate; memory may fill but not critical
            pass
            logger.warning('WARNING: timeframes not been able to deallocated. Unless you have multiple trajectories you should not have any problem other than filling the memory')

    def read_timeframes(self, trajf):
        """Read trajectory frames from file, respecting memory settings.

        If ``memory_demanding`` is ``False`` all frames up to
        ``keep_frames`` are read into :attr:`mdobj.timeframes`. For
        memory-demanding cases only streaming is prepared and the
        frames are not fully cached.
        """
        if not self.mdobj.memory_demanding:
            # In non-memory-demanding mode we cache all frames up to num_end
            num_end = 1e16 if self.keep_frames[1] is None else self.keep_frames[1]
            self.mdobj.read_file(trajf, num_end)  # read frames into memory
        else:
            self.mdobj.setup_reading(trajf)  # streaming mode: only prepare reader state
        # Restrict cached frames to desired window, if requested
        if not self.keep_frames == (None, None):
            self.mdobj.cut_timeframes(*self.keep_frames)

    def handleCharge(self, appendhydro=[]):
        """Add ghost hydrogens and append atoms if requested.

        This is a thin wrapper around :mod:`add_atoms` helpers.
        """
        if len(appendhydro) > 0:
            add_atoms.add_ghost_hydrogens(self.mdobj, appendhydro)  # add ghost H
            add_atoms.append_atoms(self.mdobj, 'ghost')  # append to system

    def handleWeights(self, ft, dynOptions):
        """Prepare weight arrays for dynamics / TACF, MSD, etc. calculations.

        This separates selection masks in ``ft`` from per-entity weights
        and moves the latter into ``dynOptions`` where the inner kernels
        expect them.
        """
        # Extract degree weights from filter if present
        if 'degree' in ft:
            weights = ft['degree']
            ft = {t: v for t, v in ft.items() if t != 'degree'}
        # Handle weights in dynamic options
        if 'w' in dynOptions:
            if 'degree' in ft:
                dynOptions['weights_t'] = weights
            del dynOptions['w']  # remove 'w' key
        return ft, dynOptions

    @staticmethod
    def check_direction(direction):
        """Validate and convert a 3D direction vector to a NumPy array."""
        if len(direction) != 3:
            raise Exception('direction must be a 3D vector (list) or np.array')
        td = type(direction)
        if not (td is list or td is type(np.ones(3)) or td is tuple):
            raise Exception('direction is not the proper type')
        return np.array(direction)

    def computeTACF(self, prop, xt, ft, dynOptions):
        """Compute time autocorrelation function (TACF) for a property.
        Parameters
        ----------
        prop : str, property name
        xt : trajectory coordinates
        ft : filter options
        dynOptions : additional options for TACF
        """
        ft, dynOptions = self.handleWeights(ft, dynOptions)  # process weights
        if len(ft) > 0:
            # Compute TACF for each subset
            dyn = {k: self.mdobj.TACF(prop, xt, fs, **dynOptions) for k, fs in ft.items()}
            dyn['system'] = self.mdobj.TACF(prop, xt)  # also compute for the whole system
        else:
            dyn = self.mdobj.TACF(prop, xt)  # compute for full system
        return dyn

    def computeDynamics(self, prop, xt, ft, dynOptions):
        """Compute dynamics property (e.g., MSD, Fs) for trajectories."""
        ft, dynOptions = self.handleWeights(ft, dynOptions)  # handle weights
        if len(ft) > 0:
            # Compute property for each filter subset
            dyn = {k: self.mdobj.Dynamics(prop, xt, fs, **dynOptions) for k, fs in ft.items()}
            dyn['system'] = self.mdobj.Dynamics(prop, xt, **dynOptions)  # full system
        else:
            dyn = self.mdobj.Dynamics(prop, xt, **dynOptions)  # full system
        return dyn

    def computeKinetics(self, filt_t):
        """Compute kinetics from filtered trajectory data.
        Parameters
        ----------
        filt_t : dict, filtered trajectory masks
        Returns
        -------
        kinetics : dict, kinetics results with 'res', 'nonres', and 'time'
        """
        kinetics = dict()
        for key, ft in filt_t.items():
            # Invert filter for non-residence calculation
            ftads = {t: np.logical_not(v) for t, v in ft.items()}
            kinet = self.mdobj.Kinetics(ftads)
            kinetics[f'{key}-nonres'] = kinet['K'].copy()
            kinet = self.mdobj.Kinetics(ft)  # non-resident
            kinetics[f'{key}-res'] = kinet['K'].copy()
        kinetics['time'] = kinet['time'].copy()  # common time array
        return kinetics


    def residence_dynamics(self, trajf, topol_vec, filters = None):
        """Compute residence dynamics for a segmental molecule.
        Parameters
        ----------
        trajf : str or list, trajectory file(s) to analyze
        topol_vec : defines the segement
        filters : dict of filters
        Returns
        -------
        kinetics : dict, residence/non-residence kinetics with fraction and time arrays
        """


        # Load frames
        self.read_timeframes(trajf)

        # Segment CM and filters
        cm_t, filt_t = self.mdobj.calc_segCM_t(topol_vec, filters=filters)

        # Release memory
        self.dealloc_timeframes()

        # Compute kinetics
        kinetics = self.computeKinetics(filt_t)

        # Residence fractions
        for key, ft in filt_t.items():
            fraction = np.array([np.count_nonzero(f)/f.shape[0] for f in ft.values()])
            kinetics[f'{key}-fraction'] = fraction

        return kinetics

    def cluster_dynamics(self, trajf, mol, rcut, method='min'):
        """Compute cluster-size time evolution and isolated/clustered kinetics.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        mol : str, molecule name
        rcut : float, cutoff distance for clustering
        method : str, clustering method ('min' default)
        Returns
        -------
        kinet : dict, includes cluster sizes, isolated/in-cluster kinetics and time
        """
        # Load trajectory
        self.read_timeframes(trajf)
        data = self.mdobj.calc_cluster_size_t(mol, rcut, method)
        self.dealloc_timeframes()

        # Number of molecules of this type
        n = np.unique(self.mdobj.mol_ids[self.mdobj.mol_names == mol]).shape[0]

        # Cluster membership per time
        clusters_time = data['clusters']

        isol = dict()
        # Identify isolated molecules (clusters of size 1)
        for k, clusters in clusters_time.items():
            isisolated = np.zeros(n, dtype=bool)
            for c in clusters:
                if len(c) == 1:
                    mid = list(c)[0]
                    isisolated[mid] = True
            isol[k] = isisolated

        # Complement: molecules in clusters
        notisol = {t: np.logical_not(v) for t, v in isol.items()}

        kinet = dict()

        # Kinetics of isolated molecules
        det = self.mdobj.Kinetics(isol)
        kinet['isolated'] = det['K']

        # Kinetics of clustered molecules
        att = self.mdobj.Kinetics(notisol)
        kinet['in cluster'] = att['K']

        # Add cluster-size time data
        kinet.update(data)

        return kinet

    def cluster_size(self, trajf, mol, rcut, method='min'):
        """Compute cluster-size evolution for a given molecule type.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        mol : str, molecule name
        rcut : float, cutoff distance for clustering
        method : str, clustering method ('min' default)
        Returns
        -------
        dyn : dict, cluster-size data over time
        """
        # Load frames
        self.read_timeframes(trajf)

        # Compute cluster size per frame
        dyn = self.mdobj.calc_cluster_size_t(mol, rcut, method)

        # Free memory
        self.dealloc_timeframes()

        return dyn


    def dynamic_structure_factor(self, trajf, q, filters=dict(), dynOptions=dict()):
        """Compute the dynamic structure factor Fs(q,t).
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        q : array-like, wavevector(s)
        filters : dict, selection filters for atoms/segments
        dynOptions : dict, extra options for dynamics calculation
        Returns
        -------
        dyn : dict, dynamic structure factor data
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Insert q into options
        dynOptions["q"] = q

        # Determine which IDs to use (polymer or atoms)
        try:
            ids = self.mdobj.polymer_ids
        except:
            ids = self.mdobj.at_ids

        # Extract coordinates through time
        coords_t, ft = self.mdobj.calc_coords_t(ids, filters=filters)

        # Compute Fs(q,t)
        dyn = self.computeDynamics('Fs', coords_t, ft, dynOptions)

        # Free cached frames
        self.dealloc_timeframes()

        return dyn


    def segmental_dynamics(self, trajf, topol_vec=4, filters=dict(),
                           prop='P1', dynOptions=dict()):
        """Compute segmental reorientation dynamics.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        topol_vec : int, index defining segment vector
        filters : dict, selection filters
        prop : str, correlation type (e.g. 'P1')
        dynOptions : dict, options for dynamics computation
        Returns
        -------
        dyn : dict, segmental dynamics results
        """
        self.read_timeframes(trajf)

        # Segment vectors over time
        seg_t, fseg = self.mdobj.calc_vectors_t(topol_vec, filters=filters)

        # Compute correlation dynamics
        dyn = self.computeDynamics(prop, seg_t, fseg, dynOptions)

        self.dealloc_timeframes()
        return dyn


    def segmental_dipole_dynamics(self, trajf, topol_vec,
                                  appendhydro=[], filters=dict(),
                                  prop='P1', dynOptions=dict()):
        """Compute segmental dipole-moment dynamics.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        topol_vec : int, segment vector index
        appendhydro : list, hydrogens to include for charge handling
        filters : dict, selection filters
        prop : str, correlation function type
        dynOptions : dict, dynamics options
        Returns
        -------
        dyn : dict, dipole dynamics results
        """
        self.read_timeframes(trajf)

        # Adjust partial charges if hydrogens added
        self.handleCharge(appendhydro)

        # Dipole moment time series
        dm_t, fdm = self.mdobj.calc_segmental_dipole_moment_t(topol_vec, filters=filters)

        dyn = self.computeDynamics(prop, dm_t, fdm, dynOptions)

        self.dealloc_timeframes()
        return dyn


    def segmental_desorption(self, trajf, topol_vec, kin=['des','ads'], method='space'):
        """Compute segmental desorption/adsorption kinetics.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        topol_vec : int, segment vector index
        kin : str or list, which kinetics to compute ('des','ads', or both)
        method : str, filtering method ('space' or 'conf')
        Returns
        -------
        kinetics : dict, desorption/adsorption kinetics and time data
        """
        # Define filtering scheme
        if method == 'space':
            filt = {'space': self.mdobj.adsorption_interval}
        elif method == 'conf':
            filt = {'conformations': ['train']}
        else:
            raise ValueError('method should be equal to "space" or "conf", you gave {}'.format(method))

        # Normalize/validate kin
        valuerror = 'acceptable values are "des","ads", ["des","ads"], you gave {}'.format(kin)
        if kin == 'des' or kin == 'ads':
            kin = [kin]
        else:
            try:
                for k in kin:
                    if k not in ['des', 'ads']:
                        raise ValueError(valuerror)
            except:
                raise ValueError(valuerror)

        self.read_timeframes(trajf)

        # Segment vectors with filter applied
        seg_t, fseg = self.mdobj.calc_vectors_t(topol_vec, filters=filt)

        # Merge multiple interval masks into a single one
        keys = list(fseg.keys())
        ft = {t: v for t, v in fseg[keys[0]].items()}
        for key in keys[1:]:
            for t in ft:
                ft[t] = np.logical_or(ft[t], fseg[key][t])

        kinetics = dict()
        # Compute adsorption/desorption kinetics
        for k in kin:
            if k == 'ads':
                ftads = {t: np.logical_not(v) for t, v in ft.items()}
                kinet = self.mdobj.Kinetics(ftads)
                kinetics['ads'] = kinet['K'].copy()

            if k == 'des':
                kinet = self.mdobj.Kinetics(ft)
                kinetics['des'] = kinet['K'].copy()

            # Add time array (same for both)
            kinetics['time'] = kinet['time'].copy()

        self.dealloc_timeframes()
        return kinetics


    def segmental_msd(self, trajf, topol_vec, direction=[1,1,1],
                      filters=dict(), dynOptions=dict()):
        """ Compute segmental mean-squared displacement (MSD).
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        topol_vec : int, index of segment vector
        direction : array-like, projection direction (e.g., [1,1,1])
        filters : dict, optional trajectory filters
        dynOptions : dict, MSD computation options
        Returns
        -------
        dyn : dict, MSD results
        """
        direction = self.check_direction(direction)

        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute 0) segment center-of-mass trajectory 1) filter(s) trajectory
        cmt, ft = self.mdobj.calc_segCM_t(topol_vec, filters=filters)

        # Apply directional projection if required
        if not (direction == np.ones(3)).all():
            cmt = {t: c * direction for t, c in cmt.items()}

        # Compute MSD from projected coordinates
        dyn = self.computeDynamics('MSD', cmt, ft, dynOptions)

        self.dealloc_timeframes()
        return dyn

    def dihedral_dynamics(self, trajf, phi, prop='sin', filters=dict(), dynOptions=dict()):
        """Compute dihedral angle time autocorrelation function (TACF).
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        phi : str, dihedral type to analyze
        prop : str, correlation property ('sin','cos', etc.)
        filters : dict, optional filters applied to dihedrals
        dynOptions : dict, TACF computation options
        Returns
        -------
        dyn : dict, TACF results
        """
        # Ensure dihedral type exists
        if phi not in self.mdobj.dihedral_types:
            raise ValueError('{} is not in dihedrals'.format(phi))

        # Load frames
        self.read_timeframes(trajf)

        # Compute dihedral angles over time
        phit, ft = self.mdobj.calc_dihedrals_t(phi, filters=filters)

        # Compute correlation function
        dyn = self.computeTACF(prop, phit, ft, dynOptions)

        self.dealloc_timeframes()
        return dyn


    def dihedral_distribution(self, trajf, phi, filters=dict(), degrees=True):
        """Compute distribution of a dihedral angle over simulation time.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        phi : str, dihedral type to analyze
        filters : dict, optional filters applied to dihedral values
        degrees : bool, convert values to degrees if True
        Returns
        -------
        distr : dict or array, dihedral distributions for each filter group
        """
        # Validate dihedral type
        if phi not in self.mdobj.dihedral_types:
            raise ValueError('{} is not in dihedrals'.format(phi))

        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute time-resolved dihedral values
        phit, ft = self.mdobj.calc_dihedrals_t(phi, filters=filters)

        # Initialize distribution groups
        distr = {'system': []}
        distr.update({k: [] for k in ft})

        # Accumulate dihedral values
        for t, dih in phit.items():
            distr['system'].extend(dih)
            for k in ft.keys():
                distr[k].extend(dih[ft[k][t]])

        # Convert to degrees if requested
        if degrees:
            scale = 180 / np.pi
            distr = {k: np.array(distr[k]) * scale for k in distr}
        else:
            distr = {k: np.array(distr[k]) * scale for k in distr}

        self.dealloc_timeframes()

        # If only one distribution, return it directly
        if len(distr) == 1:
            distr = distr[list(distr.keys())[0]]

        return distr


    def conformation_evolution(self, trajf, option=''):
        """Compute time evolution of chain conformations.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        option : str, conformation selection mode ('' for default)
        Returns
        -------
        confs : dict, conformations indexed by time
        """
        # Load trajectory
        self.read_timeframes(trajf)

        # Compute conformations across frames
        confs = self.mdobj.calc_conformations_t(option)

        self.dealloc_timeframes()
        return confs


    def density_profile(self, trajf, binl, dmax, offset=0, option='', mode='mass', flux=None, types=None):
        """Compute density profile along a spatial axis.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        binl : float, bin length
        dmax : float, maximum distance along axis
        offset : float, optional position offset
        option : str, selection mode for atoms/molecules
        mode : str, 'mass' or 'number' density
        flux : array-like, optional flux weighting
        types : list, optional atom/molecule types to include
        Returns
        -------
        densprof : array, density profile
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute density profile
        densprof = self.mdobj.calc_density_profile(binl, dmax, offset, option, mode, flux, types)

        self.dealloc_timeframes()
        return densprof


    def orientation(self, trajf, topol_vec, binl, dmax, offset=0, option=''):
        """Compute segmental orientation parameter (P2) along an axis.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        topol_vec : int or list, topology vector index(es)
        binl : float, bin length
        dmax : float, maximum distance
        offset : float, optional offset
        option : str, selection mode
        Returns
        -------
        p2 : array, orientation parameter P2 profile
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute P2 orientation parameter
        p2 = self.mdobj.calc_P2(topol_vec, binl, dmax, offset, option)

        # Clear memory of trajectory frames
        self.dealloc_timeframes()
        return p2


    def Rg(self, trajf, option=''):
        """Compute radius of gyration (Rg) of chains.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        option : str, optional selection mode
        Returns
        -------
        cha : array or dict, Rg values
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute Rg
        cha = self.mdobj.calc_Rg(option)

        # Clear memory
        self.dealloc_timeframes()
        return cha


    def chain_structure(self, trajf, binl, dmax, offset=0, option=''):
        """Compute chain structural characteristics (e.g., density, bond lengths).
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        binl : float, bin length
        dmax : float, maximum distance
        offset : float, optional offset
        option : str, optional selection mode
        Returns
        -------
        cha : dict or array, chain structure metrics
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute chain structural characteristics
        cha = self.mdobj.calc_chain_characteristics(binl, dmax, offset)

        # Clear trajectory frames from memory
        self.dealloc_timeframes()
        return cha

    def static_dipole_correlations(self, trajf, topol_vec, appendhydro=[], filters=dict()):
        """Compute static dipole moment correlations for segments.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        topol_vec : int or list, topology vector indices
        appendhydro : list, optional hydrogen atoms to append
        filters : dict, optional selection filters
        Returns
        -------
        corrs : array or dict, dipole moment correlation results
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Optionally append hydrogen atoms
        self.handleCharge(appendhydro)

        # Compute static dipole correlations
        corrs = self.mdobj.calc_segmental_dipole_moment_correlation(topol_vec, filters)

        # Clear memory
        self.dealloc_timeframes()
        return corrs


    def segmental_pair_distribution(self, trajf, binl, dmax, topol_vector, far_region=0.8):
        """Compute segmental pair distribution function.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        binl : float, bin length
        dmax : float, maximum distance
        topol_vector : int or list, topology vector indices
        far_region : float, cutoff for far regions, default 0.8
        Returns
        -------
        paird : array, segmental pair distribution function
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute segmental pair distribution
        paird = self.mdobj.calc_segmental_pair_distribution(binl, dmax, topol_vector, far_region)

        # Clear memory
        self.dealloc_timeframes()
        return paird


    def pair_distribution(self, trajf, binl, dmax, type1=None, type2=None,
                          intra=False, inter=False, far_region=0.8):
        """Compute pair distribution function (g(r)) between particle types.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        binl : float, bin length
        dmax : float, maximum distance
        type1 : int or str, optional first particle type
        type2 : int or str, optional second particle type
        intra : bool, include intra-molecular pairs
        inter : bool, include inter-molecular pairs
        far_region : float, cutoff for far regions
        Returns
        -------
        pd : array, pair distribution function
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute pair distribution
        pd = self.mdobj.calc_pair_distribution(binl, dmax, type1, type2, intra, inter, far_region)

        # Clear memory
        self.dealloc_timeframes()
        return pd


    def Sq(self, trajf, dq, qmax, method='inverse', qmin=2, dmin=0, dr=None, dmax=None, ids=None, direction=None):
        """Compute static structure factor S(q) from trajectory.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        dq : float, q increment
        qmax : float, maximum q
        method : str, 'inverse' for g(r) inversion or direct calculation
        qmin : float, minimum q
        dmin : float, minimum distance for inverse method
        dr : float, optional bin size for inverse method
        dmax : float, optional max distance for inverse method
        ids : list or array, optional particle IDs
        direction : array-like, optional direction vector
        Returns
        -------
        dat : dict, contains 'q', 'Sq', 'qmax', and 'Sqmax'
        """
        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute S(q) based on chosen method
        if method == 'inverse':
            if dr is None: dr = dq
            if dmax is None: dmax = qmax/10.0
            dat = self.mdobj.calc_Sq_byInverseGr(dr, dmax, dq, qmax, qmin, ids, direction)
        else:
            dat = self.mdobj.calc_Sq(qmin, dq, qmax, direction=None, ids=None)

        # Find qmax and Sqmax
        q = dat['q']; Sq = dat['Sq']
        dat.update({'qmax': q[q>1][Sq[q>1].argmax()], 'Sqmax': Sq.max()})

        # Clear memory
        self.dealloc_timeframes()
        return dat


    def get_dirs_for_stress_relaxation(self, u):
        """Get indices of stress components based on direction.
        Parameters
        ----------
        u : str or int, stress direction ('shear','normal','x','y','z','xy','xz','yz') or index
        Returns
        -------
        dirs : list of int, indices of stress components
        """
        valueErr = 'u (direction of stress) must be one of {shear, normal, x, y, z, xy, xz, yz, yx, zx, zy, 0-7}'
        if u == 'shear': dirs = [1,2,3,5,6,7]
        elif u == 'normal': dirs = [0,4,8]
        elif u == 'z': dirs = [8]
        elif u == 'y': dirs = [4]
        elif u == 'x': dirs = [0]
        elif u in ['xy','yx']: dirs = [1,3]
        elif u in ['xz','zx']: dirs = [2,6]
        elif u in ['yz','zy']: dirs = [5,8]
        else:
            try: dirs = [int(u)]
            except: raise ValueError(valueErr)
            else:
                if dirs[0]<0 or dirs[0]>8: raise ValueError(valueErr)
        return dirs


    def average_the_inner(self, srel, n, key):
        """Average inner dictionary values over multiple realizations.
        Parameters
        ----------
        srel : dict, nested structure {group:{i:{key:array}}}
        n : int, number of elements to average
        key : str, key to extract from inner dict
        Returns
        -------
        average_s : dict, averaged data with 'time' and 'g'
        """
        average_s = dict()
        for k in srel:
            average_s[k] = np.zeros(n, dtype=float)
            for i in srel[k]:
                average_s[k] += srel[k][i][key]
            average_s[k] /= len(srel[k])

        for k in srel:
            i0 = list(srel[k].keys())[0]
            average_s[k] = {'time': srel[k][i0]['time'], 'g': average_s[k]}
        return average_s


    def region_stress_relaxation(self, trajf, u='shear', filters=dict()):
        """Compute region-averaged stress relaxation correlations.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        u : str, stress direction
        filters : dict, optional masks for regions
        Returns
        -------
        average_s : dict, averaged stress correlation functions
        """
        dirs = self.get_dirs_for_stress_relaxation(u)

        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute stress tensor per timeframe
        stress_t, ft = self.mdobj.stress_per_t(filters=filters)

        # Prepare region-wise stress values
        vregion = {'system': {i: {t: np.mean(stress_t[t], axis=0)[i] for t in stress_t} for i in dirs}}
        if len(ft) != 0:
            vregion.update({k: {t: np.mean(stress_t[t][f[t]], axis=0) for t in stress_t} for k, f in ft.items()})

        # Compute relaxation using multy_tau_average
        srel = {k: {i: self.mdobj.multy_tau_average(v[i]) for i in dirs} for k, v in vregion.items()}

        # Average over inner indices
        average_s = self.average_the_inner(srel, len(stress_t), 'corr')

        # Clear memory
        self.dealloc_timeframes()
        return average_s


    def atom_stress_relaxation(self, trajf, u='shear', filters=dict(), dynOptions=dict()):
        """Compute atom-resolved stress relaxation correlations.
        Parameters
        ----------
        trajf : str or list, trajectory file(s)
        u : str, stress direction
        filters : dict, optional masks for atoms
        dynOptions : dict, optional dynamics computation options
        Returns
        -------
        average_s : dict or array, averaged stress correlation functions
        """
        dirs = self.get_dirs_for_stress_relaxation(u)

        # Load trajectory frames
        self.read_timeframes(trajf)

        # Compute stress tensor per timeframe
        stress_t, ft = self.mdobj.stress_per_t(filters=filters)

        # Extract stress components for desired directions
        sti = {i: {t: v[:, i] for t, v in stress_t.items()} for i in dirs}

        # Compute dynamics for each direction
        srel = {i: self.computeDynamics('scalar', st, ft, dynOptions) for i, st in sti.items()}

        # Handle region-based rearrangement if filters exist
        if len(ft) == 0:
            srel = {'system': srel}
        else:
            srel = ass.rearrange_dict_keys(srel)

        # Average over inner structure
        average_s = self.average_the_inner(srel, len(stress_t), 'scalar')

        # Return system if no regions
        if len(ft) == 0: return average_s['system']

        # Clear memory
        self.dealloc_timeframes()
        return average_s


class multy_traj():
    """Class for handling multiple trajectory files and averaging/wrapping their data."""

    def __init__(self):
        return

    @staticmethod
    def average_data(files,function,*fargs, **fkwargs):
        """Compute average and standard deviation over multiple files.
        Parameters
        ----------
        files : list, set, or tuple, trajectory file identifiers
        function : callable, function to read/process a single file
        fargs : additional positional arguments for function
        fkwargs : additional keyword arguments for function
        Returns
        -------
        averaged_data : dict, averaged data with added standard deviation fields
        """

        def ave1():
            """Average for dict-of-arrays data structure.

            Assumes that ``function`` returns a flat dict mapping keys
            to arrays of identical shape for each file.
            """
            ave_data = dict()
            # Initialize lists of values for each key from first file
            ldata = {k:[data[k]] for k in data}
            for i in range(1,len(files)):
                file = files[i]
                di = function(file,*fargs,**fkwargs)
                # Append each key's data from current file
                for k in ldata:
                    ldata[k].append( di[k])
            # Compute mean and std for each key across files
            for k in ldata:
                ave_data[k] = np.nanmean( ldata[k], axis=0 )
                ave_data[k+'_std'] =  np.nanstd( ldata[k], axis=0 )
                ave_data[k + '(rawlist)'] = ldata[k]
            return ave_data

        def ave2():
            """Average for dict-of-dict-of-arrays data structure.

            Used when ``function`` returns nested dictionaries (e.g.
            region -> property -> array) for each file.
            """
            ave_data = {k1:dict() for k1 in data}
            # Initialize lists of values for each nested key
            ldata = {k1:{k2: [data[k1][k2] ] for k2 in data[k1]} for k1 in data}
            for i in range(1,len(files)):
                file = files[i]
                di = function(file,*fargs,**fkwargs)
                # Append each nested key's data from current file
                for k1 in ldata:
                    for k2 in ldata[k1]:
                        ldata[k1][k2].append(di[k1][k2])
            # Compute mean and std for each nested key
            for k1 in ldata:
                for k2 in ldata[k1]:
                    ave_data[k1][k2] = np.nanmean( ldata[k1][k2], axis=0 )
                    ave_data[k1][k2+'_std'] =  np.nanstd( ldata[k1][k2], axis=0 )
                    ave_data[k1][k2 + '(rawlist)'] = ldata[k1][k2]
            return ave_data

        files = list(files)
        data = function(files[0],*fargs,**fkwargs)  # probe structure using first file

        # Choose appropriate averaging method based on the structure of the result
        if ass.is_dict_of_dicts(data):
            ave_method =  ave2
        elif type(data) is dict:
            ave_method = ave1
        else:
            raise Exception('Type of data and arrangement not recognized')

        averaged_data = ave_method()
        return averaged_data

    @staticmethod
    def wrap_the_data(data_to_wrap,wrapon):
        """Concatenate multiple datasets along a specified axis/key.
        Parameters
        ----------
        data_to_wrap : dict, data from multiple files
        wrapon : str, key to determine concatenation points
        Returns
        -------
        wraped_data : dict, concatenated data across files
        """
        wraped_data = dict()
        print(data_to_wrap.keys())  # debug aid: show contributing file keys
        for i,(file,data) in enumerate(data_to_wrap.items()):
            if i==0:
                wraped_data = data  # first file as base
                continue

            # Case for dict-of-dict data (e.g. region -> property -> array)
            if ass.is_dict_of_dicts(data):
                print(data.keys())
                # Save old wrap key values to determine overlapping time/coordinate overlap
                varw_old =  {k: wraped_data[k][wrapon] for k in wraped_data}
                for k,d  in data.items():
                    jp0 = d[wrapon][0]  # start of new segment
                    jp1 = d[wrapon][1]  # end of new segment
                    joint_point = jp0 if jp0 !=0 else jp1  # define joint point (skip leading zeros)
                    fk = varw_old[k]<joint_point  # old points before joint
                    # Cut overlapping points from old data
                    for ik in d:
                        try:
                            wraped_data[k][ik] = wraped_data[k][ik][fk]
                        except KeyError:
                            pass
                    fnk = d[wrapon]>=joint_point  # new points to append from current segment
                    # Append new points to wrapped data
                    for ik in d:
                        if  ik in wraped_data[k]:
                            ctupl = (wraped_data[k][ik], d[ik][fnk])
                        else:
                            ctupl =(d[ik][fnk],)
                        wraped_data[k][ik] = np.concatenate(ctupl)

            # Case for dict-of-array data (single-level dict)
            elif type(data) is dict:
                varw_old = wraped_data[wrapon]  # previous wrap key
                jp0 = data[wrapon][0]
                jp1 = data[wrapon][1]
                joint_point = jp0 if jp0 !=0 else jp1
                fk = varw_old<joint_point  # mask for old points to keep
                # cut overlapping old points
                for ik in data:
                    try:
                        wraped_data[ik] = wraped_data[ik][fk]
                    except KeyError:
                        pass
                fnk = data[wrapon]>=joint_point  # mask for new points to append
                for ik in data:
                    if ik in wraped_data:
                        ctupl = (wraped_data[ik], data[ik][fnk])
                    else:
                        ctupl = (data[ik][fnk],)
                    wraped_data[ik] = np.concatenate(ctupl)
            else:
                raise Exception('Unrecognized type of data for wrapping')
        return wraped_data

    @staticmethod
    def multiple_trajectory(files,function,*fargs,**fkwargs):
        """Process multiple trajectory files with averaging and optional wrapping.
        Parameters
        ----------
        files : str, list, tuple, set, or dict, input trajectory files
        function : callable, function to read/process a single file
        fargs : additional positional arguments for function
        fkwargs : additional keyword arguments for function (can include 'wrapon')
        Returns
        -------
        mult_data : dict or array, combined/averaged trajectory data
        """
        if 'wrapon' not in fkwargs:
            wrapon = 'time'  # default concatenation key
        else:
            wrapon = fkwargs['wrapon']

        type_files = type(files)

        # Case: dict of file sets or lists (e.g. grouping by state/ensemble)
        if type_files is dict:
            changing_args = ass.is_tuple_of_samesized_tuples(fargs)  # per-key argument tuples?
            if changing_args:
                # Each dict key gets its corresponding tuple of args
                if len(fargs) != len(files):
                    raise Exception('if you pass tuple of tuples then they should match dictionary size')
                return {k:
                        multy_traj.multiple_trajectory(f, function, *fargs[i], **fkwargs)
                        for i,(k,f) in enumerate(files.items())
                        }
            else:
                # Same args for all files in dict
                return {k:
                        multy_traj.multiple_trajectory(f, function, *fargs, **fkwargs)
                        for k,f in files.items()
                        }

        # Case: set of files -> only averaging, no wrapping
        if type_files is set:
            mult_data = multy_traj.average_data(files,function,*fargs,**fkwargs)

        # Case: list or tuple of files -> process each, then wrap in a single dataset
        elif type_files is list or type_files is tuple:
            data_to_wrap = dict()
            for file in files:
                type_file  = type(file)
                if type_file is set:
                    data = multy_traj.average_data(file,function,*fargs,**fkwargs)
                elif type_file is str:
                    data  =  function(file,*fargs,**fkwargs)
                # Use concatenated file names as key if set, else the filename itself
                if type_file is set:
                    key = '-'.join(file)
                else:
                    key = file
                data_to_wrap[key] = data
            # Wrap all processed files into single dataset
            mult_data = multy_traj.wrap_the_data(data_to_wrap,wrapon)

        # Case: single file
        elif type_files is str:
            mult_data = function(files,*fargs,**fkwargs)

        else:
            raise Exception('type {} is not recognized as file or files to read'.format(type_files))

        return mult_data

class ass():
    """
    The ASSISTANT class
    Provides utility functions to assist in data analysis including:
    1) Dictionary manipulation functions
    2) Data wrapping and trajectory handling helpers
    3) Printing, logging, and time formatting
    4) Numerical and averaging utilities
    """

    @staticmethod
    def update_dict_in_object(obj,name,di):
        """Update an object's attribute dictionary or create it if it doesn't exist.
        Parameters
        ----------
        obj : object
            Target object
        name : str
            Attribute name to update or create
        di : dict
            Dictionary to merge
        """
        if not hasattr(obj,name):
            setattr(obj,name,di)  # create new attribute
        else:
            getattr(obj,name).update(di)  # merge into existing dict
        return

    @staticmethod
    def list_ifint(i):
        """Return [i] if i is int, else return i as is."""
        if type(i) is int:
            return [i,]
        else:
            return i

    @staticmethod
    def list_ifstr(i):
        """Return [i] if i is str, else return i as is."""
        if type(i) is str:
            return [i,]
        else:
            return i

    @staticmethod
    def list_iffloat(i):
        """Return [i] if i is float, else return i as is."""
        if type(i) is float:
            return [i,]
        else:
            return i

    @staticmethod
    def make_dir(name):
        """Create nested directories if they do not exist.
        Parameters
        ----------
        name : str
            Directory path
        Returns
        -------
        a : int
            System call result (0 if successful)
        """
        name = name.replace('\\','/')  # normalise separators
        n = name.split('/')
        lists = ['/'.join(n[:i+1]) for i in range(len(n))]  # incremental paths
        a = 0
        for l in lists:
            if not os.path.exists(l):
                # Try creating directory
                a = os.system('mkdir {:s}'.format(l))
                if a!=0:
                    # Fallback for Windows-style path
                    s = l.replace('/','\\')
                    a = os.system('mkdir {:s}'.format(s))
        return a

    @staticmethod
    def numerical_derivative(x,y):
        """Compute numerical derivative of y with respect to x using central differences.
        Parameters
        ----------
        x : array_like
        y : array_like
        Returns
        -------
        d : np.ndarray, same shape as x
        """
        d = np.empty_like(x)  # one derivative value per x
        # Central difference for inner points
        d[1:-1] = (y[2:] -y[:-2])/(x[2:]-x[:-2])
        # Forward difference for first point
        d[0] = (y[1] -y[0])/(x[1]-x[0])
        # Backward difference for last point
        d[-1] = (y[-1]-y[-2])/(x[-2]-x[-1])
        return d

    beebbeeb = True

    @staticmethod
    def is_tuple_of_samesized_tuples(x):
        """Check if input is a tuple of tuples of equal lengths."""
        if type(x) is not tuple:
            return False
        for i in x:
            if type(i) is not tuple:
                return False
        try:
            x[0]
        except IndexError:
            return False  # empty tuple
        else:
            l = len(x[0])
            for i in x:
                if l != len(i):  # length mismatch
                    return False
        return True

    @staticmethod
    def rename_key(d,oldname,newname):
        """Rename a dictionary key."""
        d[newname] = d[oldname]
        del d[oldname]
        return

    @staticmethod
    def rename_keys_via_keyvalue(new_names,data_dict):
        """Rename dictionary keys using a mapping dictionary {old:new}."""
        new_dict = dict()
        for k,v in new_names.items():  # k: old key, v: new key
            new_dict[v] = data_dict[k]
        return new_dict

    @staticmethod
    def rename_keys_via_enumeration(new_names,data_dict):
        """Rename dictionary keys via enumeration over a list of new keys."""
        new_dict = dict()
        l = list(data_dict.keys())  # original key order
        for i,k in enumerate(new_names):
            new_dict[k] = data_dict[l[i]]
        return new_dict

    @staticmethod
    def rename_keys(new_names,data_dict):
        """General function to rename dictionary keys using a mapping or a list."""
        if type(new_names) is dict:
            data_dict = ass.rename_keys_via_keyvalue(new_names,data_dict)
        elif type(new_names) is list or type(new_names) is tuple:
            data_dict = ass.rename_keys_via_enumeration(new_names,data_dict)
        else:
            raise ValueError('new_names must be dict,list or tuple ')
        return data_dict

    @staticmethod
    def write_pickle(data,data_file):
        """Save data to a pickle file."""
        with open(data_file,'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def read_pickle(data_file):
        """Load data from a pickle file."""
        with open(data_file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    @staticmethod
    def save_data(data,fname,method='pickle'):
        """Save data using specified method."""
        if not ass.iterable(method):
            methods = [method,]
        for m in methods:
            if m =='pickle':
                ass.write_pickle(data,fname)
            else:
                raise NotImplementedError
        return

    @staticmethod
    def try_beebbeeb():
        """Optional beep sound to notify user."""
        if ass.beebbeeb:
            try:
                import winsound
                winsound.Beep(500, 1000)
                import time
                time.sleep(0.5)
                winsound.Beep(500, 1000)
            except:
                pass
        return

    @staticmethod
    def change_key(d,keyold,keynew):
        """Rename a dictionary key with error handling."""
        try:
            value = d[keyold]
        except KeyError as e:
            raise Exception('{} This key "{}" does not belong to the dictionary'.format(e,keyold))
        d[keynew] = value
        del d[keyold]
        return

    @staticmethod
    def dict_slice(d,i1,i2):
        """Slice a dictionary by index range."""
        return {k:v for i,(k,v) in enumerate(d.items()) if i1<=i<i2 }

    @staticmethod
    def numpy_keys(d):
        """Return dictionary keys as a numpy array."""
        return np.array(list(d.keys()))

    @staticmethod
    def numpy_values(d):
        """Return dictionary values as a numpy array."""
        return np.array(list(d.values()))

    @staticmethod
    def common_keys(d1,d2):
        """Return common keys between two dictionaries."""
        k1 = ass.numpy_keys(d1) ; k2 = ass.numpy_keys(d2)
        return np.intersect1d(k1,k2)

    @staticmethod
    def trunc_at(dold,dnew):
        """Return index in old dictionary before first key in new dictionary."""
        new_key_first = list(dnew.keys())[1]
        dko = ass.numpy_keys(dold)
        if len(dold) ==0:
            return 0
        try:
            fa = dko< new_key_first  # mask for keys strictly before new first key
        except:
            return 0
        else:
            dl = dko[fa]
            itrunc = np.where(dl[-1] == dko)[0][0]  # index of last old key to keep
            return itrunc

    @staticmethod
    def is_dict_of_dicts(data):
        """Check if a dictionary contains dictionaries as values."""
        ks = list(data.keys())
        try:
            v0 = data[ks[0]]
            return isinstance(v0, dict)
        except IndexError:
            return False

    @staticmethod
    def print_time(tf,name,nf=None):
        """Print formatted elapsed time for a function.
        Parameters
        ----------
        tf : float, total time in seconds
        name : str, function name
        nf : int, optional number of frames for average
        """
        s1 = ass.readable_time(tf)
        if nf is None:
            s2 =''
        else:
            s2 = ' Time/frame --> {:s}\n'.format( ass.readable_time(tf/nf))
        logger.info('Function "{:s}"\n{:s} Total time --> {:s}'.format(name,s2,s1))
        return

    @staticmethod
    def print_stats(stats):
        """Print percentages of chain types from stats dictionary."""
        print('ads chains  = {:4.4f} %'.format(stats['adschains_perc']*100))
        x = [stats[k] for k in stats if '_perc' in k and k.split('_')[0] in ['train','tail','loop','bridge'] ]
        tot = np.sum(x)
        for k in ['train','loop','tail','bridge']:
            print('{:s} = {:4.2f}%'.format(k,stats[k+'_perc']/tot*100))
        return

    @staticmethod
    def stay_True(dic):
        """Return cumulative logical AND across dictionary values (True accumulation)."""
        keys = list(dic.keys())
        stayTrue = {keys[0]:dic[keys[0]]}
        for i in range(1,len(dic)):
            stayTrue[keys[i]] = np.logical_and(stayTrue[keys[i-1]],dic[keys[i]])
        return stayTrue

    @staticmethod
    def become_False(dic):
        """Return cumulative logical AND of first value with NOT of subsequent values."""
        keys = list(dic.keys())
        bFalse = {keys[0]:dic[keys[0]]}
        for i in range(1,len(dic)):
            bFalse[keys[i]] = np.logical_and(bFalse[keys[0]],np.logical_not(dic[keys[i]]))
        return bFalse

    @staticmethod
    def iterable(arg):
        """Check if object is list, tuple, or dict."""
        return type(arg) is list or type(arg) is tuple or type(arg) is dict

    @staticmethod
    def readable_time(tf):
        """Convert seconds to human-readable h:m:s format."""
        hours = int(tf/3600)
        minutes = int((tf-3600*hours)/60)
        sec = tf-3600*hours - 60*minutes
        dec = sec - int(sec)
        sec = int(sec)
        return '{:d}h : {:d}\' : {:d}" : {:0.3f}"\''.format(hours,minutes,sec,dec)

    @staticmethod
    def rearrange_dict_keys(dictionary):
        """Swap levels of a dict-of-dict structure.
        Parameters
        ----------
        dictionary : dict of dicts
        Returns
        -------
        x : dict of dicts with swapped keys
        """
        x = {k2 : {k1:None for k1 in dictionary} for k2 in dictionary[list(dictionary.keys())[0]]}
        for k1 in dictionary:
            for k2 in dictionary[k1]:
                x[k2][k1] = dictionary[k1][k2]
        return x

    @staticmethod
    def check_occurances(a):
        """Check that all elements occur only once in an iterable."""
        x = set()
        for i in a:
            if i not in x:
                x.add(i)
            else:
                raise Exception('{} is more than once in the array'.format(i))
        return

    @jit(nopython=True,fastmath=True)
    def running_average(X,every=1):
        """Compute running average over array X."""
        n = X.shape[0]
        xrun_mean = np.zeros(n)  # running mean sampled every ``every`` steps
        for j in range(0,len(X),every):  # grow window from start up to j
            y = X[:j+1]
            n = y.shape[0]
            xrun_mean[j] = np.sum(y)/n
        return xrun_mean

    def moving_average(a, n=10) :
        """Compute moving average with window size n."""
        mov = np.empty_like(a)
        n2 = int(n/2)
        if n2%2 ==1: n2+=1
        up = a.shape[0]-n2  # last index where a full symmetric window fits
        for i in range(n2):
            mov[i] = a[:2*i+1].mean()  # grow window near the left boundary
        for i in range(n2,up):
            mov[i] = a[i-n2:i+n2+1].mean()  # full-width window in the bulk
        for i in range(up,a.shape[0]):
            j = (a.shape[0]-i)-1
            mov[i] = a[up-j:].mean()  # shrink window near the right boundary
        return mov

    def block_average(a, n=100) :
        """Compute block average with block size n."""
        bv = np.empty(int(a.shape[0]/n)+1,dtype=float)  # final block may be shorter
        for i in range(bv.shape[0]):
            x = a[i*n:(i+1)*n]
            bv[i] = x.mean()
        return bv

    def block_std(a, n=100) :
        """Compute block standard deviation with block size n."""
        bstd = np.empty(int(a.shape[0]/n),dtype=float)  # discard any leftover tail
        for i in range(bstd.shape[0]):
            x = a[i*n:(i+1)*n]
            bstd[i] = x.std()
        return bstd


@jit(nopython=True,fastmath=True,parallel=False)
def distance_kernel(d,coords,c):
    """
    Compute distances between a set of points and a single reference point.
    Parameters
    ----------
    d : np.ndarray, preallocated array to store distances
    coords : np.ndarray, array of n points with m dimensions
    c : np.ndarray, reference point coordinates
    Notes
    -----
    Updates distances in-place in `d` using Euclidean distance.
    """
    nd = coords.shape[1]  # number of dimensions of coordinate vectors
    for i in prange(d.shape[0]):  # loop over points whose distances are stored in d
        d[i] = 0
        for j in range(nd):
            rel = coords[i][j] - c[j]  # displacement in each dimension
            d[i] += rel * rel
        d[i] = d[i] ** 0.5  # final Euclidean distance
    return

@jit(nopython=True,fastmath=True,parallel=True)
def smaller_distance_kernel(d1,d2,c1,c2):
    """
    Compute the minimum distance from each point in c1 to all points in c2.
    Parameters
    ----------
    d1 : np.ndarray, preallocated array to store minimum distances for each point in c1
    d2 : np.ndarray, preallocated array for temporary distances from each c1[i] to c2
    c1 : np.ndarray, first set of n1 points in m dimensions
    c2 : np.ndarray, second set of n2 points in m dimensions
    Notes
    -----
    Updates d1 in-place with the minimum Euclidean distance from each point in c1 to points in c2. Utilizes `distance_kernel` internally.
    """
    for i in prange(c1.shape[0]):  # each iteration: one reference point in c1
        distance_kernel(d2,c2,c1[i])  # reuse d2 as workspace for distances to all c2 points
        d1[i] = 1e16  # initialize with a large number (acts as +inf)
        for j in range(d2.shape[0]):
            if d2[j] < d1[i]:
                d1[i] = d2[j]  # keep the smallest distance
    return




class Energetic_Analysis():
    '''
    Class to analyze GROMACS energy files (.xvg format).
    Currently supports basic reading and plotting; needs improvement for more formats.
    '''
    def __init__(self,file):
        """
        Initialize the Energetic_Analysis object and read the data file.
        Parameters
        ----------
        file : str, path to the .xvg energy file
        """
        self.data_file = file
        if self.data_file[-4:] == '.xvg':
            self.xvg_reader()
        else:
            raise NotImplementedError('Currently only accepting xvg files')

    def xvg_reader(self):
        """
        Read an .xvg GROMACS energy file and store it as a pandas DataFrame.
        Attributes
        ----------
        self.data : pd.DataFrame, columns are 'time' + legends extracted from the file
        """
        with open(self.data_file) as f:
            lines = f.readlines()
            f.closed

        columns = ['time']  # initialize columns with 'time'
        for i,line in enumerate(lines):
            l = line.split()
            if l[0] == '@' and l[1][0] == 's' and l[2] == 'legend':
                columns.append(l[3].strip('"'))  # extract legend names
                last_legend = i
            elif line[0] == '@' or line[0] == '#':
                continue
            else:
                break

        # extract numerical data after last legend line
        data = np.array([line.split() for line in lines[last_legend+1:]], dtype=float)
        self.data = pd.DataFrame(data, columns=columns)
        return

    def simple_plot(self,ycols,xcol='time',size=3.5,dpi=300,
                    title='',func=None,
                    xlabel=['time (ps)'],save_figs=False,fname=None,path=None):
        """
        Plot selected columns from the energy data.
        Parameters
        ----------
        ycols : list of str, columns to plot on y-axis
        xcol : str, column to plot on x-axis (default 'time')
        size : float, figure size multiplier
        dpi : int, figure resolution in dots per inch
        title : str, title of the plot
        func : str, name of function to apply to y values before plotting
        xlabel : list of str, x-axis labels
        save_figs : bool, whether to save figure
        fname : str, filename for saving
        path : str, directory path to save figure
        """
        figsize = (size,size)
        plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor', length=1.5*size)
        plt.tick_params(direction='in', which='major', length=1.5*size)
        plt.xlabel('time (ps)', fontsize=size*3)
        x = self.data[xcol].to_numpy()
        funcs = globals()[func] if func is not None else None  # get function from globals

        # loop over y columns and plot
        for ycol in ycols:
            y = self.data[ycol].to_numpy()
            plt.plot(x, y, label=ycol)
            if funcs is not None:
                plt.plot(x, funcs(y), label='{} - {}'.format(ycol, func))
        plt.legend(frameon=False)

        # save figure if requested
        if fname is not None:
            if path is not None:
                plt.savefig('{}\{}'.format(path, fname), bbox_inches='tight')
            else:
                plt.savefig('{}'.format(fname), bbox_inches='tight')
        plt.show()


class XPCS_communicator():
    '''
    Static methods for handling XPCS relaxation data.
    Includes reading multiple XPCS files, plotting distributions, and saving data for XPCS analysis.
    '''
    @staticmethod
    def ReadPlot_XPCSdistribution(files,**plot_kwargs):
        """
        Read multiple XPCS output files, plot distributions, and return dictionary of data.
        Parameters
        ----------
        files : list of str, paths to XPCS batch files
        plot_kwargs : dict, additional plotting keyword arguments
        Returns
        -------
        datadict : dict, keys are file identifiers, values are XPCS_Reader objects
        """
        datadict = dict()
        for file in files:
            key = file.split('/')[-1].split('.XPCSCONTINbatch')[0].replace('xpcs_','')
            datadict[key] = XPCS_Reader(file)
        plotter.relaxation_time_distributions(datadict,**plot_kwargs)
        return datadict

    @staticmethod
    def write_xy_forXPCS(fname,x,y):
        """
        Write x-y data to a file in XPCS-compatible format with tiny random noise on y.
        Parameters
        ----------
        fname : str, output filename
        x : np.ndarray, x values
        y : np.ndarray, y values
        """
        data = np.zeros((x.shape[0],3),dtype=float)
        data[:,0] = x
        data[:,1] = y
        data[:,2] += np.random.uniform(0,1e-9,x.shape[0])
        np.savetxt(fname,data)
        return

    @staticmethod
    def write_data_forXPCS(datadict,path='XPCS_data',cutf=None,midtime=None,num=100):
        """
        Save XPCS data dictionary to files, optionally applying cutoffs and logarithmic sampling.
        Parameters
        ----------
        datadict : dict, XPCS_Reader data
        path : str, directory to save files
        cutf : dict or None, maximum x value for each dataset
        midtime : float or None, middle time for logarithmic sampling
        num : int, number of points to sample
        """
        if cutf is None:
            cutf = {k:10**10 for k in datadict}
        for k, dy in datadict.items():
            fn = '{:s}\\xpcs_{:}.txt'.format(path,k)
            x = ass.numpy_keys(dy)/1000
            y = ass.numpy_values(dy)
            t = x <= cutf[k]
            x = x[t]
            y = y[t]
            args = plotter.sample_logarithmically_array(x, midtime=midtime, num=num)
            xw = x[args]
            yw = y[args]
            XPCS_communicator.write_xy_forXPCS(fn, xw, yw+1)
        return


class XPCS_Reader():
    '''
    Class to read and process individual XPCS relaxation data files.
    Stores relaxation modes, parameters, fitted curves, and contributions.
    '''
    def __init__(self,fname,fitfunc='freq'):
        """
        Initialize XPCS_Reader and read data from a file.
        Parameters
        ----------
        fname : str, path to XPCS file
        fitfunc : str, fitting function name to use (default 'freq')
        """
        self.relaxation_modes = []
        self.params = []
        self.params_std = []
        self.func = getattr(fitFuncs, fitfunc)

        with open(fname,'r') as f:
            lines = f.readlines()
            f.closed

        # Parse metadata
        for i,line in enumerate(lines):
            l = line.strip()
            if 'Background' in l: self.background = float(l.split(':')[1])
            if 'log Lagrange Multiplier' in l: self.reg = float(l.split(':')[1])
            if 'log(Upsilon)' in line: linum = i+2
            if 'Kohlrausch Exponent' in l: self.ke = float(l.split(':')[1])

        # Read relaxation modes and parameters
        for i,line in enumerate(lines[linum:]):
            l = line.strip().split()
            if '----' in line: break
            self.relaxation_modes.append(float(l[0]))
            self.params.append(float(l[1]))
            self.params_std.append(float(l[2]))

        self.params = np.array(self.params)
        self.params_std = np.array(self.params_std)

        # Compute smoothness and relaxation times
        dlogtau = self.relaxation_modes[1] - self.relaxation_modes[0]
        self.smoothness = smoothness(self.params,dlogtau)
        self.relaxation_modes = 10**np.array(self.relaxation_modes)
        w = self.params
        f = self.relaxation_modes
        self.taus = self.relax()
        self.bounds = (f[0], f[-1])
        self.contributions = w * self.taus
        self.taur = self.tau_relax()
        self.t = np.logspace(-4, np.log10(self.taur)+2, num=10000)
        self.curve = self.fitted_curve(self.t)
        return

    def tau_relax(self):
        """
        Compute total relaxation time as the sum of contributions.
        Returns
        -------
        float, total relaxation time
        """
        return np.sum(self.contributions)

    def relax(self):
        """
        Compute relaxation times for each mode based on Kohlrausch exponent.
        Returns
        -------
        np.ndarray, relaxation times
        """
        if self.ke == 1.0:
            return 1.0/self.relaxation_modes
        elif self.ke == 2.0:
            return 0.5*np.sqrt(np.pi)/np.sqrt(self.relaxation_modes)

    def plot_distribution(self,size=3.5,xlim=(-6,8),title=None):
        """
        Plot relaxation mode distribution (weights and contributions) in log-log scale.
        Parameters
        ----------
        size : float, figure size multiplier
        xlim : tuple of int, x-axis limits for log scale
        title : str or None, plot title
        """
        fig,ax = plt.subplots(figsize=(size,size),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor', length=size*1.5)
        plt.tick_params(direction='in', which='major', length=size*3)
        plt.yscale('log')
        plt.xscale('log')
        if title is not None: plt.title(title)
        xticks = [10**x for x in range(xlim[0], xlim[1]+1)]
        plt.xticks(xticks, fontsize=min(2.5*size, 2.5*size*8/len(xticks)))
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'$f$ / $ns^{-1}$', fontsize=2*size)

        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0.1,1,0.1), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        x = self.relaxation_modes
        w = self.params
        c = self.contributions

        # Plot weights and contributions
        plt.plot(x, w, ls='--', lw=0.25*size, marker='o', markersize=size*1.2, markeredgewidth=0.15*size, fillstyle='none', label='w')
        plt.plot(x, c, ls='--', lw=0.25*size, marker='s', markersize=size*1.2, markeredgewidth=0.15*size, fillstyle='none', label='c')
        plt.legend(frameon=False, fontsize=2.3*size)
        plt.show()
        return

    def fitted_curve(self,x):
        """
        Compute the fitted relaxation curve at given x values using the assigned fitting function.
        Parameters
        ----------
        x : np.ndarray, x values at which to evaluate the curve
        Returns
        -------
        np.ndarray, fitted curve values
        """
        fc = self.func(x, self.bounds[0], self.bounds[1], self.params)
        try: fc += self.background
        except AttributeError: pass
        return fc

class plotter():
    '''
    Class for plotting XPCS and dynamics data.
    Contains color palettes, linestyles, and static methods for various plotting routines.
    '''
    def __init__(self):
        return

    class colors():
        qualitative = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        diverging = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf','#d9ef8b','#a6d96a','#66bd63','#1a9850']
        div6 = ['#d73027','#fc8d59','#fee090','#e0f3f8','#91bfdb','#4575b4']
        sequential = ['#fee0d2','#fc9272','#de2d26']
        safe = ['#1b9e77','#d95f02','#7570b3']
        semisafe = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
        safe2 = ['#a6cee3','#1f78b4','#b2df8a']
        qual6 = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']

    class linestyles():
        lst_map = {
            'loosely dotted': (0, (1, 3)), 'dotted': (0, (1, 1)), 'densely dotted': (0, (2, 1)),
            'loosely dashed': (0, (5, 3)), 'dashed': (0, (4, 2)), 'densely dashed': (0, (3, 1)),
            'loosely dashdotted': (0, (5, 3, 1, 3)), 'dashdotted': (0, (4, 2, 1, 2)), 'densely dashdotted': (0, (3, 1, 1, 1)),
            'dashdotdotted': (0, (5, 2, 1, 2, 1, 2)), 'loosely dashdotdotted': (0, (4, 3, 1, 3, 1, 3)), 'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
        }
        lst3 = [lst_map['densely dotted'], lst_map['loosely dashed'], lst_map['densely dashed']]
        lst4 = [lst_map['densely dotted'], lst_map['loosely dotted'], lst_map['densely dashed'], lst_map['loosely dashed']]
        lst6 = [lst_map['loosely dotted'], lst_map['dotted'], lst_map['dashed'], lst_map['loosely dashdotted'], lst_map['dashdotted'], lst_map['densely dashdotted']]
        lst7 = ['-', '-.', '--', lst_map['loosely dotted'], lst_map['dotted'], lst_map['dashed'], lst_map['loosely dashdotted'], lst_map['dashdotted'], lst_map['densely dashdotted'], lst_map['densely dotted']]

    @staticmethod
    def boldlabel(label):
        """
        Make each word in a label bold using LaTeX formatting.
        Parameters
        ----------
        label : str, input label
        Returns
        -------
        str, bolded label
        """
        label = label.split(' ')
        boldl = ' '.join([r'$\mathbf{'+l+'}$' for l in label])
        return boldl

    @staticmethod
    def relaxation_time_distributions(datadict,fitobject='params',yscale=None,size=3.5,fname=None,title=None,
                                      cmap=None,xlim=(-6,6),pmap=None,ylim=(1e-6,1.0),units='ns',mode='tau'):
        """
        Plot relaxation time distributions from XPCS data.
        Parameters
        ----------
        datadict : dict, keys are identifiers, values are XPCS_Reader objects
        fitobject : str, 'params' or 'eps_imag' for plotting
        yscale : str or None, y-axis scale
        size : float, figure size multiplier
        fname : str or None, save filename
        title : str or None, plot title
        cmap : dict or None, colors for each dataset
        xlim : tuple, x-axis limits in log scale
        pmap : dict or None, marker styles
        ylim : tuple, y-axis limits
        units : str, time units
        mode : str, 'tau' or 'freq' for x-axis label
        """
        if cmap is None:
            c = plotter.colors.semisafe
            try: cmap = {k: c[i] for i,k in enumerate(datadict.keys())}
            except IndexError:
                color_map = matplotlib.cm.get_cmap('viridis')
                cmap = {k: color_map(i/len(datadict)) for i,k in enumerate(datadict.keys())}
        if pmap is None:
            pmap = {k:'o' for k in datadict}
        figsize = (size,size)
        dpi = 300
        fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor', length=size*1.5)
        plt.tick_params(direction='in', which='major', length=size*3)
        plt.xscale('log')
        if yscale is not None:
            plt.yscale(yscale)
            plt.ylim(ylim)
        if title is not None: plt.title(title)
        xticks = [10**x for x in range(xlim[0], xlim[1]+1)]
        plt.xticks(xticks, fontsize=min(2.5*size, 2.5*size*8/len(xticks)))
        plt.yticks(fontsize=2.5*size)
        lab = 'freq' if mode=='freq' else r'\tau'
        units_label = '{:s}^{-1}'.format(units) if mode=='freq' else units
        plt.xlabel(r'${:s}$ / ${:s}$'.format(lab,units_label), fontsize=2*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0.1,1,0.1), numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for i,(k,f) in enumerate(datadict.items()):
            if fitobject=='params': x, y = f.relaxation_modes, f.params
            elif fitobject=='eps_imag': x, y = f.omega, f.eps_imag
            plt.plot(x, y, ls='--', lw=0.25*size, marker=pmap[k], markeredgewidth=0.15*size,
                     fillstyle='none', color=cmap[k], label=k)
        plt.legend(frameon=False, fontsize=2.3*size)
        if fname is not None: plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return

    @staticmethod
    def sample_logarithmically_array(x, midtime=None, num=100, first_ten=True):
        """
        Generate indices to sample an array logarithmically.
        Parameters
        ----------
        x : np.ndarray, array to sample
        midtime : float or iterable of floats, optional intermediate times
        num : int, number of samples
        first_ten : bool, whether to include first 10 points
        Returns
        -------
        np.ndarray, indices of sampled points
        """
        if midtime is not None:
            if not ass.iterable(midtime):
                midtime = [float(midtime)]
            else:
                midtime = [float(m) for m in midtime]
            midtime.append(float(x[-1]))
        else:
            midtime = [float(x[-1])]

        nm = len(midtime)
        args = np.array([0])
        if first_ten: num -= nm*10
        num = int(num/nm)
        i = 0
        for midt in midtime:
            mid = x[x<=midt].shape[0]
            if first_ten:
                fj = int(round(10**i,0))
                args = np.concatenate((args, [j for j in range(fj,fj+10)]+[fj-1]))
            try:
                lgsample = np.logspace(i, np.log10(mid), num=num).round(0)
                args = np.concatenate((args, np.array(lgsample, dtype=int)))
            except ValueError as e:
                logger.warning('Expected ValueError{}\nConsider increasing number of sampling points'.format(e))
            i = np.log10(mid)
        args = np.unique(args[args<x.shape[0]])
        return np.array(args,dtype=int)


class inverseFourier():
    '''
    Class to perform inverse Fourier transforms on a given function.
    Stores time array, Fourier-transformed function, and angular frequencies.
    '''
    def __init__(self, t, ft, omega, omega_0=1e-16, omega_oo=1e16):
        """
        Initialize the inverseFourier object.
        Parameters
        ----------
        t : np.ndarray, time array
        ft : np.ndarray, Fourier-transformed function values
        omega : float or np.ndarray, angular frequency/frequencies
        omega_0 : float, lower reference frequency for normalization
        omega_oo : float, upper reference frequency for normalization
        """
        self.t = t
        self.ft = ft
        self.omega = omega
        self.omega_0 = omega_0
        self.omega_oo = omega_oo
        return

    @staticmethod
    def derft(ft, t):
        """
        Numerical derivative of a function with respect to time.
        Parameters
        ----------
        ft : np.ndarray, function values
        t : np.ndarray, time array
        Returns
        -------
        np.ndarray, derivative of ft with respect to t
        """
        d = np.empty(ft.shape[0], dtype=float)
        d[0] = (ft[1]-ft[0])/(t[1]-t[0])
        d[-1] = (ft[-2]-ft[-1])/(t[-1]-t[-2])
        for i in range(1, ft.shape[0]-1):
            d[i] = (ft[i+1]-ft[i-1])/(t[i+1]-t[i-1])
        return d

    def find_epsilon(self, normalize=True):
        """
        Compute epsilon (inverse Fourier transform) from the derivative of ft.
        Parameters
        ----------
        normalize : bool, whether to normalize the result using omega_0 and omega_oo
        Returns
        -------
        np.ndarray or complex, epsilon values at self.omega
        """
        def ep_epp(t, dft, o):
            I = -simpson(dft * np.exp(-1j*o*t), x=t)
            return I

        dft = self.derft(self.ft, self.t)
        if ass.iterable(self.omega):
            eps = np.array([ep_epp(self.t, dft, o) for o in self.omega])
        else:
            eps = ep_epp(self.t, dft, self.omega)

        if normalize:
            eps0 = ep_epp(self.t, dft, self.omega_0)
            epsoo = ep_epp(self.t, dft, self.omega_oo)
            eps = eps * (eps0 - epsoo) + epsoo
        return eps

class fitLinear():
    '''
    Piecewise linear fitting class using pwlf (piecewise linear fit)
    '''
    def __init__(self, xdata, ydata, nlines=1):
        """
        Initialize fitLinear object.
        Parameters
        ----------
        xdata : np.ndarray, independent variable
        ydata : np.ndarray, dependent variable
        nlines : int, number of linear segments
        """
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.nlines = nlines
        self.fitlines()

    @staticmethod
    def piecewise_linear(x, x0, y0, k1, k2):
        """
        Piecewise linear function with two segments.
        Parameters
        ----------
        x : np.ndarray, independent variable
        x0 : float, break point
        y0 : float, y value at break point
        k1 : float, slope before x0
        k2 : float, slope after x0
        Returns
        -------
        np.ndarray
        """
        return np.piecewise(x, [x < x0],
                            [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

    @staticmethod
    def costF(p, x, y):
        """
        Cost function for fitting.
        Parameters
        ----------
        p : list/tuple, parameters for piecewise_linear
        x : np.ndarray, independent variable
        y : np.ndarray, dependent variable
        Returns
        -------
        float
        """
        yp = fitLinear.piecewise_linear(x, *p)
        return np.sum((y-yp)**2 / y**2)

    def find_slopes(self):
        """
        Calculate slopes and intersections of each linear segment.
        Updates self.slopes and self.intersections
        """
        slopes = []
        intersects = []
        xf = self.xyline[0]
        yf = self.xyline[1]
        a = xf.argsort()
        xf = xf[a]
        yf = yf[a]
        breaks = self.breaks[self.breaks.argsort()]
        for i, b in enumerate(breaks[1:]):
            bm = breaks[i]
            fb = np.logical_and(bm < xf, xf < b)
            x = xf[fb]
            y = yf[fb]
            slope = (y[-1]-y[0])/(x[-1]-x[0])
            ise = y[-1] - slope*x[-1]
            slopes.append(slope)
            intersects.append(ise)
        self.slopes = slopes
        self.intersections = intersects
        return

    def fitlines(self):
        """
        Fit piecewise linear function to data using pwlf.
        Updates self.breaks and self.xyline
        """
        import pwlf
        x = self.xdata
        y = self.ydata
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit(self.nlines)
        self.breaks = breaks
        xd = np.arange(x.min(), x.max(), 0.01)
        self.xyline = (xd, my_pwlf.predict(xd))
        self.find_slopes()
        return


class xifit():
    '''
    Xi fitting class using a tanh function
    '''
    def __init__(self, tau, d, xirho=0.2):
        """
        Initialize xifit object.
        Parameters
        ----------
        tau : np.ndarray, relaxation times
        d : np.ndarray, distances
        xirho : float, reference distance
        """
        self.lntau = np.log(tau)
        self.tau = tau
        self.d = d
        self.xirho = xirho
        self.fit()

    @staticmethod
    def func(xi, c, lntau0, xirho, d):
        """
        Fit function: c * tanh((d-xirho)/xi) + lntau0
        Parameters
        ----------
        xi : float, width parameter
        c : float, amplitude
        lntau0 : float, base log(tau)
        xirho : float, reference distance
        d : np.ndarray, distance array
        Returns
        -------
        np.ndarray
        """
        return c*np.tanh((d-xirho)/xi) + lntau0

    @staticmethod
    def dfunc(xi, c, lntau0, xirho, d):
        """
        Derivative of func with respect to d
        """
        return c / np.cosh((d-xirho)/xi)**2 / xi

    @staticmethod
    def costf(pars, d, xirho, lntau):
        """
        Cost function for xi fit.
        Parameters
        ----------
        pars : list/tuple, fit parameters [xi, c, lntau0]
        d : np.ndarray, distance
        xirho : float, reference distance
        lntau : np.ndarray, log of tau
        Returns
        -------
        float
        """
        pred = xifit.func(*pars, xirho, d)
        r = ((lntau - pred)/lntau)**2
        return r.sum() / r.shape[0]

    def fit(self):
        """
        Fit xi function to data using differential evolution.
        Updates self.p, self.curve, self.dcurve
        """
        bounds = [(0, 3), (-5, 5), (-15, 15)]
        from scipy.optimize import differential_evolution
        opt_res = differential_evolution(self.costf, bounds,
                                        args=(self.d, self.xirho, self.lntau),
                                        maxiter=10000)
        self.opt_res = opt_res
        self.p = opt_res.x
        self.d = np.arange(self.d.min(), self.d.max()+0.01, 0.01)
        self.curve = np.exp(self.func(*self.p, self.xirho, self.d))
        self.dcurve = self.curve * self.dfunc(*self.p, self.xirho, self.d)
        return


class Arrheniusfit():
    '''
    Arrhenius fitting class
    '''
    def __init__(self, temp, tau):
        """
        Initialize Arrheniusfit object
        Parameters
        ----------
        temp : np.ndarray, temperature array
        tau : np.ndarray, relaxation times
        """
        self.tau = tau
        self.temp = temp
        self.fit()
        t = np.arange(temp.min(), temp.max()+0.01, 0.01)
        self.t = t
        self.curve = self.exp(t, *self.opt_res.x)
        return

    @staticmethod
    def exp(temp, A, Ea):
        """
        Arrhenius exponential function
        Parameters
        ----------
        temp : np.ndarray, temperature
        A : float, prefactor
        Ea : float, activation energy
        Returns
        -------
        np.ndarray
        """
        return A * np.exp(-Ea/temp)

    @staticmethod
    def costf(pars, temp, tau):
        """
        Cost function for Arrhenius fit
        """
        tau_pred = Arrheniusfit.exp(temp, *pars)
        return np.sum((np.log10(tau_pred)-np.log10(tau))**2)/tau.size

    def fit(self):
        """
        Fit Arrhenius function using differential evolution
        Updates self.p, self.curve
        """
        bounds = [(0, 1e4), (-1e5, 0)]
        from scipy.optimize import differential_evolution
        opt_res = differential_evolution(self.costf, bounds,
                                        args=(self.temp, self.tau),
                                        maxiter=1000)
        self.opt_res = opt_res
        self.p = opt_res.x
        self.t = np.arange(self.temp.min(), self.temp.max()+0.01, 0.01)
        self.curve = self.exp(self.t, *self.p)
        return


class VFTfit():
    '''
    Vogel-Fulcher-Tammann (VFT) fitting class
    '''
    def __init__(self, temp, tau, pars=[], pmin=None, pmax=None):
        """
        Initialize VFTfit object
        Parameters
        ----------
        temp : np.ndarray, temperature array
        tau : np.ndarray, relaxation times
        pars : list, optional pre-set parameters [A, D, t0]
        pmin : float, minimum temperature for curve
        pmax : float, maximum temperature for curve
        """
        self.tau = tau
        self.temp = temp
        if len(pars) == 0:
            self.fitVFT()
        elif len(pars) == 3:
            self.p = np.array(pars)
        if pmin is None:
            pmin = temp.min()
        if pmax is None:
            pmax = temp.max()
        t = np.arange(pmin, pmax+0.01, 0.01)
        self.t = t
        self.curve = self.vft(t, *self.p)
        return

    @staticmethod
    def vft(temp, A, D, t0):
        """
        VFT function
        Parameters
        ----------
        temp : np.ndarray
        A : float
        D : float
        t0 : float
        Returns
        -------
        np.ndarray
        """
        return A * np.exp(D / (temp - t0))

    @staticmethod
    def costf(pars, temp, tau):
        """
        Cost function for VFT fit
        """
        tau_pred = VFTfit.vft(temp, *pars)
        return np.sum((np.log10(tau_pred)/np.log10(tau) - 1)**2 / tau**0.5)/tau.size

    def fitVFT(self):
        """
        Fit VFT function using differential evolution
        Updates self.p
        """
        from scipy.optimize import differential_evolution
        bounds = [(0, 1e-3), (1e2, 1e4), (0, 180)]
        opt_res = differential_evolution(self.costf, bounds,
                                        args=(self.temp, self.tau),
                                        maxiter=10000)
        self.opt_res = opt_res
        self.p = opt_res.x
        return



class fitData():
    """
    FitData class: Computes the smoothest relaxation times distribution from
    autocorrelation or frequency-domain data using multi-objective optimization.
    It balances fitting accuracy vs smoothness of the relaxation time distribution.

    Attributes
    ----------
    xdata : np.ndarray
        Independent variable (time or frequency)
    ydata : np.ndarray
        Dependent variable (autocorrelation)
    func : str
        Name of the fitting function to use
    method : str
        Fitting method, e.g., 'distribution' for relaxation time distributions
    bounds : tuple
        Min and max bounds for relaxation times
    search_grid : tuple
        Grid dimensions for multi-objective optimization
    reg_bounds : tuple
        Bounds for regularization parameters
    keep_factor : float
        Factor for acceptable residual tolerance
    bias_factor : float
        Factor controlling bias in optimization
    nw : int
        Number of relaxation modes
    maxiter : int
        Maximum iterations for optimizers
    sigmF : list
        List of smoothness factors for optimization
    show_plots : bool
        Flag to show plots
    """
    from scipy.optimize import minimize

    def __init__(self, xdata, ydata, func, method='distribution', bounds=None,
                 search_grid=None, reg_bounds=None, keep_factor=2, bias_factor=1.0,
                 nw=50, bound_res=1e-9, maxiter=200, sigmF=[10,20,30,40,50,60],
                 show_plots=False, **options):
        """
        Initialize fitData object with data, fitting function, and optimization settings.

        Parameters
        ----------
        xdata : np.ndarray
            Independent variable array
        ydata : np.ndarray
            Dependent variable array
        func : str
            Name of fitting function (should exist in fitFuncs)
        method : str
            Fitting method ('distribution' or other)
        bounds : tuple
            Min and max bounds for relaxation times
        search_grid : tuple
            Grid size for multi-objective optimization
        reg_bounds : tuple
            Bounds for regularization
        keep_factor : float
            Multiplier for minimum residual to accept solutions
        bias_factor : float
            Bias factor in optimization
        nw : int
            Number of discrete relaxation modes
        bound_res : float
            Minimum resolution for bounds
        maxiter : int
            Maximum iterations for optimizer
        sigmF : list
            Smoothness factors to explore
        show_plots : bool
            Whether to show plots
        **options : dict
            Optional keyword arguments for weights, init_method, p0, opt_method, etc.
        """
        self.show_plots = show_plots
        self.method = method
        self.ydata = ydata
        self.xdata = xdata
        self.mode = func
        self.func = getattr(fitFuncs, func)

        if method == 'distribution':
            self.kernel = getattr(fitKernels, func)

        self.bounds = bounds if bounds is not None else (1e-7,1e7)
        self.search_grid = search_grid if search_grid is not None else (12,4,4)
        self.reg_bounds = reg_bounds if reg_bounds is not None else (1e-20,1e8)
        self.keep_factor = keep_factor
        self.bias_factor = bias_factor
        self.minimum_res = bound_res
        self.nw = nw
        self.maxiter = maxiter
        self.sigmF = sigmF
        self.sigma_factor = 30

        self.weights = options.get('weights', None)
        self.weighting_method = options.get('weighting_method', 'xy')
        self.is_distribution = options.get('is_distribution', True)
        self.zeroEdge_distribution = options.get('zeroEdge_distribution', True)
        self.show_report = options.get('show_report', True)
        self.init_method = options.get('init_method', 'normal')
        self.p0 = options.get('p0', None)
        self.opt_method = options.get('opt_method', 'SLSQP')
        return

    @property
    def keep_res(self):
        """
        Compute the maximum residual to accept in the Pareto optimization.
        Returns
        -------
        float
        """
        return self.keep_factor * self.minimum_res

    def clean_positive_derivative_data(self):
        """
        Remove data points where derivative of ydata with respect to xdata is positive.
        Ensures monotonic decreasing behavior.
        """
        der = np.empty_like(self.ydata)
        der[1:] = (self.ydata[1:] - self.ydata[:-1]) / (self.xdata[1:] - self.xdata[:-1])
        der[0] = der[1]
        f = der < 0
        self.xdata = self.xdata[f]
        self.ydata = self.ydata[f]
        return

    def clean_non_monotonic_data(self):
        """
        Remove non-monotonic data points to ensure strictly decreasing sequences.
        """
        x = self.xdata
        y = self.ydata
        f = np.ones(x.size, dtype=bool)
        for i in range(x.size):
            if (y[i] < y[i+1:]).any():
                f[i] = False
        self.xdata = self.xdata[f]
        self.ydata = self.ydata[f]
        return

    def clean_data(self):
        """
        Apply all data cleaning routines:
        - Remove positive derivative points
        - Remove non-monotonic points
        - Truncate at first zero point
        """
        self.clean_positive_derivative_data()
        self.clean_non_monotonic_data()
        izero = self.get_arg_t0()
        if izero != self.ydata.size:
            self.ydata = self.ydata[:izero]
            self.xdata = self.xdata[:izero]
        return

    def fitTheModel(self):
        """
        Perform the full multi-objective fit:
        - Clean data
        - Fit initial model
        - Estimate bounds and residuals
        - Perform Pareto optimization to select smoothest solution
        """
        self.clean_data()
        self.justFit()
        self.estimate_minimum_residual()
        self.taulow = self.estimate_taulow()
        self.tauhigh = self.estimate_tauhigh()
        self.refine_bounds(self.tauhigh)
        self.search_best()
        return

    def save_for_XPCS(self, fname):
        """
        Save processed data to file in format compatible with XPCS.
        Parameters
        ----------
        fname : str
            File path to save data
        """
        x = self.xdata
        y = self.ydata
        data = np.zeros((x.shape[0],3), dtype=float)
        data[:,0] = x
        data[:,1] = y + 1
        data[:,2] += np.random.uniform(0,1e-9,x.shape[0])
        np.savetxt(fname, data)
        return

    def estimate_tauhigh(self):
        """
        Estimate the upper bound of relaxation times based on smoother fit.
        Returns
        -------
        float
        """
        smfittau = self.smootherFit()
        tau = smfittau if self.get_arg_t0() == self.ydata.size else self.estimate_taudata()
        return 1e2 * tau

    def estimate_taulow(self):
        """
        Estimate lower bound of relaxation times based on small relaxation fit.
        Returns
        -------
        float
        """
        smalltau = self.smallerTauRelaxFit()
        taudata = self.estimate_taudata()
        tau = min(smalltau, taudata) if smalltau != 0 else taudata
        return 10**(-2*(1 - self.lasty)) * tau

    def estimate_taudata(self):
        """
        Estimate relaxation times from data integral.
        Returns
        -------
        float
        """
        x = self.xdata
        y = self.ydata
        izero = self.get_arg_t0()
        dt = x[1:izero] - x[:izero-1]
        yi = y[:izero-1]
        return np.sum(yi*dt)

    def estimate_minimum_residual(self):
        """
        Estimate minimum data residual for optimization.
        Updates self.minimum_res
        """
        mres = max(self.minimum_res, 1 * self.data_res)
        self.minimum_res = mres
        print('estimated data residual = {:.4e}'.format(mres))
        return

    def get_arg_t0(self):
        """
        Find index where ydata first reaches below threshold (1e-2)
        Returns
        -------
        int
        """
        smallys = np.where(self.ydata <= 1e-2)[0]
        return smallys[0] if len(smallys) > 0 else self.ydata.size


    def refine_bounds(self,th):
        """Refine bounds for tau based on mode and input threshold.
        Parameters
        ----------
        th: float threshold tau to define new bounds
        """
        if self.mode =='freq':
            bh = self.bounds[1]
            bl = 1/(th*1e5)  # lower bound in frequency mode
        else:
            bl = self.bounds[0]
            bh = th*1e5       # upper bound in tau mode
        self.bounds = (bl,bh)
        return


    def get_weights(self):
        """Return weighting vector based on weighting_method.
        Returns
        -------
        w: np.ndarray weights for each data point
        """
        if self.weighting_method == 'high-y':
            w = np.abs(self.ydata)+0.3  # high ydata gives higher weight
        elif self.weighting_method=='xy':
            w = 1  # simple weighting (placeholder for x^0.1 + y)
        else:
            w = np.ones(self.ydata.shape[0])  # uniform weights
        return w


    @property
    def bestreg(self):
        """Return the best regularization parameter based on criteria.
        Returns
        -------
        regbest: float best regularization value
        """
        a = self.bestcrit()  # index of best criterion
        regbest = self.regs[a]
        return regbest


    def search_best(self):
        """Search for best combination of tau and sigma_factor across search_grid.
        Notes
        -----
        - Updates storing_dict with results for each trial
        - Refines tau bounds iteratively
        """
        self.crit = []
        self.storing_list=  ['relaxation_modes','con_res','data_res',
                             'params','prob_distr','loss','smoothness','tau_relax',
                             'bias','sigma_factor']
        self.storing_dict = {k:[] for k in self.storing_list}

        for st in self.storing_list:
            setattr(self,'best_'+st,None)

        tl = self.taulow
        th = self.tauhigh
        print('trelax bounds: [{:.3e} , {:.3e}]'.format(tl,th))
        if tl > th:
            raise Exception('Minimizing the relaxation time gives higher tau than minimizing the smoothness')

        self.bestcr = float('inf')
        tau_bounds = (tl,th)

        for numreg in self.search_grid:
            dilog = (np.log10(tau_bounds[1])-np.log10(tau_bounds[0]))/numreg
            self.search_reg(numreg,tau_bounds)

            f = np.array(self.storing_dict['data_res']) < self.keep_res
            c = np.array(self.crit)[f]
            re = np.array(self.storing_dict['tau_relax'])[f]
            rem = re[c.argmin()]
            th = rem*10**dilog
            tl = rem*10**(-dilog)
            tau_bounds = (tl,th)

        for st in self.storing_list:
            setattr(self,st,getattr(self,'best_'+st))
        return


    def search_reg(self,numreg,taub):
        """Perform regularization search over target taus and sigma factors.
        Parameters
        ----------
        numreg: int number of tau points to evaluate
        taub: tuple (tau_min,tau_max) bounds for search
        Notes
        -----
        - Stores results in storing_dict
        - Updates crit array with combined criterion
        """
        for attr in ['crit','smv','drv','regs']:
            try:
                getattr(self,attr)
            except AttributeError:
                setattr(self,attr,[])  # initialize missing attributes

        target_taus = np.logspace(np.log10(taub[0]),np.log10(taub[1]),base=10,num=numreg)
        for tartau in target_taus:
            self.target_tau = tartau
            self.refine_bounds(tartau)
            for sigmF in self.sigmF:
                self.sigma_factor = sigmF
                self.exactFit(tartau)  # perform exact fit for this tau

                s = self.smoothness ; t = self.tau_relax ; d = self.data_res
                for k in self.storing_list:
                    self.storing_dict[k].append(getattr(self,k))  # store results

                crt = self.criterium(d,s,t)  # compute combined criterion
                self.crit.append(crt)
                self.nsearches = len(self.crit)

        self.refine_keep_res()  # adjust keep_res if needed
        self.select_best_solution()  # pick best solution
        return

    @property
    def lasty(self):
        """Return last value of ydata.
        Returns
        -------
        float last element of ydata
        """
        return self.ydata[-1]


    def criterium(self,d,s,t):
        """Compute combined optimization criterion.
        Parameters
        ----------
        d: float data residual
        s: float smoothness value
        t: float relaxation time
        Returns
        -------
        float combined criterion value
        Notes
        -----
        - Penalizes large residuals, large smoothness, and large/low t depending on lasty
        """
        return d**2*s*t**(1-self.lasty)


    def refine_keep_res(self):
        """Adaptively loosen keep_res until at least one solution satisfies threshold.
        Notes
        -----
        - Multiplies keep_factor iteratively by 1.1
        - Stops once any data_res < keep_res
        - Raises exception if keep_res grows too large
        """
        dres = np.array(self.storing_dict['data_res'])  # all stored residuals
        fd = dres < self.keep_res  # boolean mask of acceptable fits
        i=0
        factor = 1.1  # growth factor for keep_factor

        # iterate until at least one solution satisfies keep_res
        while fd.any() == False:
            self.keep_factor=self.keep_factor*factor  # increase factor
            i+=1
            fd = dres < self.keep_res
            if self.keep_res>0.1:  # safety guard
                raise Exception('Increased keep_res too much and still no solution satysfies it')
        return


    def select_best_solution(self):
        """Select best solution using minimum criterion among acceptable residuals.
        Notes
        -----
        - Filters solutions with data_res < keep_res
        - Picks the entry with minimal criterion
        - Stores all best_* attributes and best_sol dict
        """
        f = np.array(self.storing_dict['data_res']) < self.keep_res  # valid solutions mask
        a = np.array(self.crit)[f].argmin()  # index of best criterion among valid
        best = dict()

        for st in self.storing_list:
            attr = np.array(self.storing_dict[st])[f][a]  # extract best attribute
            if type(attr) is type(np.ones(3)):  # copy arrays to avoid referencing
                attr = attr.copy()
            best[st] = attr
            setattr(self,'best_'+st,attr)
        self.best_sol = best
        return


    @staticmethod
    def get_logtimes(a,b,n):
        """Generate logarithmically spaced tau values.
        Parameters
        ----------
        a: float lower bound
        b: float upper bound
        n: int number of samples
        Returns
        -------
        np.ndarray logarithmically spaced tau array
        """
        tau = np.logspace(np.log10(a),np.log10(b),base=10,num=n)
        return tau


    def initial_params(self):
        """Generate initial parameter guess depending on init_method.
        Returns
        -------
        np.ndarray initial parameter vector
        Notes
        -----
        - Methods: 'uniform','from_previous','ones','zeros','normal'
        - 'normal' uses Gaussian distribution centered at nw/2
        """
        if self.init_method =='uniform':
            pars = np.ones(self.nw)/self.nw  # uniform distribution
        elif self.init_method =='from_previous':
            try:
                pars = self.params  # reuse previous solution
            except:
                pars = np.ones(self.nw)/self.nw
        elif self.init_method =='ones':
            pars = np.ones(self.nw)  # unnormalized equal weights
        elif self.init_method=='zeros':
            pars = np.zeros(self.nw) + 1e-15  # tiny weights to avoid singularities
        elif self.init_method=='normal':
            x = np.arange(0,self.nw,1)
            mu = self.nw/2  # center of Gaussian
            sigma = mu/self.sigma_factor  # width depends on sigma_factor
            pars = np.exp(-(x-mu)**2/(sigma)**2)  # Gaussian distribution
            pars /= pars.sum()  # normalize to 1
        return pars


    def distribution_constraints(self):
        """Build distribution-related optimization constraints.
        Returns
        -------
        list list of constraint dictionaries
        Notes
        -----
        - Adds normalization constraint when is_distribution=True
        - Adds zero-edge constraints when zeroEdge_distribution=True
        """
        constraints = []

        if self.is_distribution:
            # Constraint: sum(w[:-1]) must equal 1
            cdistr = lambda w: 1 - np.sum(w[:-1])

            def dcddw(w):
                # Jacobian: derivative of cdistr wrt weights
                dw = np.zeros(w.shape)
                dw[:-1] = -1.0
                return dw

            constraints.append({'type':'eq','fun':cdistr,'jac': dcddw})

        if self.zeroEdge_distribution:
            # First and last weights must be zero
            w0 = lambda w:  -w[0]
            wn = lambda w:  -w[-2]

            def dw0dw(w):
                # Jacobian for w0 constraint
                x = np.zeros(w.shape)
                x[0] = -1
                return x

            def dwndw(w):
                # Jacobian for wn constraint
                x = np.zeros(w.shape)
                x[-2] = -1
                return x

            constraints.append({'type':'eq','fun':w0,'jac':dw0dw})
            constraints.append({'type':'eq','fun':wn,'jac':dwndw})

        return constraints


    def get_them(self):
        """Assemble data and kernel matrices for inversion.
        Returns
        -------
        int number of weights
        np.ndarray tau values
        float log10 spacing of tau
        np.ndarray weighted y-data
        np.ndarray kernel matrix with weights applied
        Notes
        -----
        - Uses log-spaced tau range from bounds
        - Applies weights to each kernel column
        """
        n = self.nw  # number of weights
        tau = fitData.get_logtimes(self.bounds[0],self.bounds[1],n)  # log-spaced tau
        dlogtau = np.log10(tau[1]/tau[0])  # spacing in log10
        x = self.xdata.copy()  # input x-values
        y = self.ydata.copy()  # measured data
        A = self.kernel(x,tau)  # kernel matrix
        whs = self.get_weights()  # weighting function

        # Apply weights to each column of kernel
        for i in range(A.shape[1]):
            A[:,i] = A[:,i]*whs

        return n,tau,dlogtau,whs*y,A

    def get_params(self):
        """Build initial parameter vector and bounds for optimizer.
        Returns
        -------
        np.ndarray initial parameter guess
        list bounds for each parameter
        Notes
        -----
        - First nw parameters are distribution weights in [0,1]
        - Last parameter is bias term with symmetric bounds
        """
        p0 = self.initial_params()  # generate initial weights
        bounds = [(0,1) for i in range(self.nw)]  # bounds for weights

        p0 = np.concatenate((p0,[0.0]))  # append bias parameter
        ms = self.minimum_res  # minimum residual scale

        # Add bounds for bias parameter
        bounds.append((-1*ms*self.bias_factor,1*ms*self.bias_factor))

        return p0,bounds

    def evaluateNstore(self,opt_res):
        """Store optimization results after fitting.
        Parameters
        ----------
        opt_res : OptimizeResult, result returned from scipy.optimize.minimize
        """
        n, tau, dlogtau, y, A = self.get_them()  # retrieve assembled matrices

        w = opt_res.x[:-1]  # distribution weights
        self.bias = opt_res.x[-1]  # bias parameter

        # Distribution summary information
        isd = 1-w.sum()  # deviation from sum=1
        bl = w[0]        # lower edge weight
        bu = w[-1]       # upper edge weight
        self.prob_distr = {'isd':isd,'blow':bl,'bup':bu}

        # Compute residual and store values
        self.con_res = compute_residual(opt_res.x,A,y)
        self.data_res = self.con_res

        # Save parameters and metadata
        self.params = w
        self.opt_res = opt_res
        self.relaxation_modes = tau
        self.smoothness = smoothness(opt_res.x,dlogtau)
        self.loss = opt_res.fun
        self.tau_relax = self.trelax

        # Optional plotting
        if self.show_plots:
            self.show_relaxation_modes(prefix='')
            self.show_relaxation_modes(prefix='',show_contributions=True,yscale='log')
            self.show_fit()

        # Optional textual report
        if self.show_report:
            self.report()
        return


    def justFit(self):
        """Perform unconstrained fit with distribution constraints only.
        Returns
        -------
        OptimizeResult result of optimization
        Notes
        -----
        - Uses residual-only objective
        - Applies distribution constraints if enabled
        """
        n, tau, dlogtau, y, A = self.get_them()  # build kernel system

        constraints = self.distribution_constraints()  # normalization/edge constraints

        p0,bounds = self.get_params()  # initial parameters and bounds

        costf = compute_residual  # objective function

        # Main optimization
        opt_res = minimize(costf,p0,
                           args=(A,y),
                           method=self.opt_method,
                           constraints=constraints,
                           bounds=bounds,
                           jac=dCRdw,
                           options={'maxiter':int(self.maxiter/2),'disp':self.show_report},
                           tol=1e-16)

        self.evaluateNstore(opt_res)  # store results
        return opt_res


    def smallerTauRelaxFit(self):
        """Fit enforcing a smaller relaxation time via inequality constraint.
        Returns
        -------
        float relaxation time achieving minimum value
        Notes
        -----
        - Adds inequality constraint that enforces low τ_relax
        - Also applies distribution constraints
        """
        n, tau, dlogtau, y, A = self.get_them()  # prepare system

        # Inequality constraint for minimizing relaxation time
        constraints = [{'type':'ineq',
                        'fun':constraint,
                        'jac':dCdw,
                        'args':(A,y,self.minimum_res)}]

        constraints.extend(self.distribution_constraints())  # add distribution rules

        p0,bounds = self.get_params()  # initial guess and bounds

        costf = FrelaxCost  # relaxation-based objective

        # Optimization
        opt_res = minimize(costf,p0,
                           args=(tau,),
                           method=self.opt_method,
                           constraints=constraints,
                           bounds=bounds,
                           #jac=dFdw,
                           options={'maxiter':self.maxiter,'disp':self.show_report},
                           tol=1e-6)

        self.evaluateNstore(opt_res)  # store fit results
        return self.trelax

    def smootherFit(self):
        """Fit by minimizing the smoothness functional.
        Parameters
        ----------
        None
        Returns
        -------
        float, relaxation time after smoothing-based optimization
        Notes
        -----
        - Minimizes smoothness while enforcing feasibility and distribution constraints
        """
        n, tau, dlogtau, y, A = self.get_them()  # compute kernel/system

        # Inequality constraint ensuring minimal residual
        constraints = [{'type':'ineq',
                        'fun':constraint,
                        'jac':dCdw,
                        'args':(A,y,self.minimum_res)}]

        constraints.extend(self.distribution_constraints())  # add distribution constraints

        p0,bounds = self.get_params()  # initial parameters and bounds

        costf = smoothness  # objective: minimize smoothness

        # Optimization step
        opt_res = minimize(costf,p0,
                           args=(dlogtau,),
                           method=self.opt_method,
                           constraints=constraints,
                           bounds=bounds,
                           jac=dSdw,
                           options={'maxiter':self.maxiter,'disp':self.show_report},
                           tol=1e-6)

        self.evaluateNstore(opt_res)  # store results
        return self.trelax


    def exactFit(self,target_tau):
        """Fit enforcing an exact target relaxation time.
        Parameters
        ----------
        target_tau : float, relaxation time to be exactly enforced
        Returns
        -------
        OptimizeResult, result returned by minimizer
        Notes
        -----
        - Adds equality constraint to force tau_relax = target_tau
        - Includes feasibility, distribution, and contribution constraints
        """
        n, tau, dlogtau, y, A = self.get_them()  # kernel system

        # Primary inequality + relaxation equality constraint
        constraints = [{'type':'ineq',
                        'fun':constraint,
                        'jac':dCdw,
                        'args':(A,y,self.minimum_res)},
                       {'type':'eq',
                        'fun':FrelaxCon,
                        'jac':dFCdw,
                        'args':(tau,target_tau)}]

        constraints.extend(self.distribution_constraints())  # normalization/edges
        constraints.extend(self.contribution_constraints(target_tau,tau))  # custom contrib constraints

        p0,bounds = self.get_params()  # initial values and bounds

        costf = FitCost  # global fit objective

        # Main optimization
        opt_res = minimize(costf,p0,
                           args=(dlogtau,),
                           method=self.opt_method,
                           constraints=constraints,
                           bounds=bounds,
                           jac=dFitCostdw,
                           options={'maxiter':self.maxiter,'disp':self.show_report},
                           tol=1e-6)

        self.evaluateNstore(opt_res)  # store all results
        return opt_res


    def contribution_constraints(self,target_tau,tau):
        """Return additional contribution constraints.
        Parameters
        ----------
        target_tau : float, target relaxation time
        tau : array, relaxation modes used for contributions
        Returns
        -------
        list, constraint dictionaries (currently empty)
        Notes
        -----
        - Placeholder for future contribution-based constraints
        """
        constraints = []  # currently no additional rules
        return constraints

    def show_residual_distribution(self,fname=None,size=3.5,title=None):
        """Show histogram of residual values collected during the search.
        Parameters
        ----------
        fname : str or None, output file name (unused)
        size : float, figure size in inches
        title : str or None, optional title for the plot
        Returns
        -------
        None
        Notes
        -----
        - Displays a log-scale histogram of all stored data residuals
        """
        figsize = (size,size)
        dpi = 300
        fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
        plt.xscale('log')
        if title is not None:
            plt.title(title)
        plt.ylabel(r'number of occurances')
        plt.xlabel(r"$residual$")
        plt.minorticks_on()
        # tick lengths proportional to figure size
        plt.tick_params(direction='in',which='minor',length=size*1.5)
        plt.tick_params(direction='in',which='major',length=size*3)
        plt.hist(self.storing_dict['data_res'],bins=100,color='k')  # residual histogram
        plt.show()
        return


    def show_tstar(self,tmax,n=1000,size=3.5,title=None):
        """Plot t* vs t*_relax over a chosen range.
        Parameters
        ----------
        tmax : float, maximum time to compute t*
        n : int, number of sampling points
        size : float, figure size in inches
        title : str or None, optional plot title
        Returns
        -------
        None
        Notes
        -----
        - Computes t* and corresponding relaxation t* values
        """
        figsize = (size,size)
        dpi = 300
        fig,ax = plt.subplots(figsize=figsize,dpi=dpi)

        dt = tmax/n
        t = np.arange(0,tmax,dt)  # time grid
        tstar = [self.tstar(ts) for ts in t]  # compute t* relaxation

        plt.yscale('log')
        plt.xscale('log')
        if title is not None:
            plt.title(title)
        plt.xlabel(r'$t^*$')
        plt.ylabel(r"$t^{*}_{relax}$")
        plt.minorticks_on()
        plt.tick_params(direction='in',which='minor',length=size*1.5)
        plt.tick_params(direction='in',which='major',length=size*3)
        plt.plot(t,tstar,color='k',marker='.')
        plt.show()
        return


    def get_wrm(self):
        """Return current distribution weights and relaxation modes.
        Parameters
        ----------
        None
        Returns
        -------
        tuple, (weights, relaxation_modes)
        Notes
        -----
        - Raises if parameters or modes are unavailable
        """
        try:
            w = self.params  # weights
        except AttributeError as err:
            raise err
        try:
            rm = self.relaxation_modes  # relaxation times
        except AttributeError as err:
            raise err
        return w,rm


    def phit(self,ts):
        """Return φ(t) or φ*(t) depending on mode.
        Parameters
        ----------
        ts : float, time value
        Returns
        -------
        float, computed φ or φ* value
        Notes
        -----
        - Uses exp(-ts*rm) or exp(-ts/rm) depending on freq/time mode
        """
        w, rm = self.get_wrm()
        f = lambda w,rm: np.sum(w*np.exp(-ts*rm))
        t = lambda w,rm: np.sum(w*np.exp(-ts/rm))
        p = f(w,rm) if self.mode == 'freq' else t(w,rm)
        return p


    def dertstar(self,ts):
        """Derivative of t* with respect to t.
        Parameters
        ----------
        ts : float, time value
        Returns
        -------
        float, derivative dt*/dt
        Notes
        -----
        - Formula: 1 - φ(t)
        """
        dts = 1 - self.phit(ts)
        return dts


    def dtstar(self,ts):
        """Return t*(t) without mode-dependent scaling factors.
        Parameters
        ----------
        ts : float, time value
        Returns
        -------
        float, t* value
        Notes
        -----
        - Computes ∑ w(1 − e^{-ts*rm}) or ∑ w(1 − e^{-ts/rm})
        """
        w, rm = self.get_wrm()
        f = lambda w,rm: np.sum((1.0-np.exp(-ts*rm))*w)
        t = lambda w,rm: np.sum(w*(1.0-np.exp(-ts/rm)))
        tr = f(w,rm) if self.mode == 'freq' else t(w,rm)
        return tr


    def tstar(self,ts):
        """Return relaxation-weighted t* value.
        Parameters
        ----------
        ts : float, time value
        Returns
        -------
        float, weighted t*
        Notes
        -----
        - Computes ∑ w(1 − e^{-ts*rm})/rm or ∑ rm w(1 − e^{-ts/rm})
        """
        w, rm = self.get_wrm()
        f = lambda w,rm: np.sum((1.0-np.exp(-ts*rm))*w/rm)
        t = lambda w,rm: np.sum(rm*w*(1.0-np.exp(-ts/rm)))
        tr = f(w,rm) if self.mode == 'freq' else t(w,rm)
        return tr


    @property
    def tmax(self):
        '''
        Returns the characteristic maximum relaxation time tmax.
        In frequency mode this is 1/rm[argmax(w)], otherwise rm[argmax(w)].
        Returns: float, computed tmax value
        '''
        w, rm = self.get_wrm()  # get weights and relaxation modes
        a = w.argmax()  # index of dominant weight
        tr = 1/rm[a]  if self.mode =='freq' else rm[a]  # compute tmax depending on mode
        return tr

    @property
    def trelax(self):
        '''
        Computes the global relaxation time trelax via Frelax or Trelax depending on mode.
        Returns: float, relaxation time
        '''
        w, rm = self.get_wrm()  # load weights and modes
        tr = Frelax(w,rm) if self.mode =='freq' else Trelax(w,rm)  # call correct relaxation function
        return tr

    @property
    def print_trelax(self):
        '''
        Prints a formatted string showing the current relaxation time trelax and bounds.
        Returns: None, prints output to console
        '''
        print('For bounds {}: --> trelax = {:.4e} ns'.format(self.bounds,self.trelax))  # formatted print

    def report(self):
        '''
        Prints a summary report of key results stored in the object.
        Returns: None, prints available attributes defined in report list
        '''
        for k in ['prob_distr','con_res','data_res',
                  'smoothness','loss','trelax',
                  'target_tau','nsearches','bias']:
            try:
                a = getattr(self,k)  # attempt to load attribute
            except:
                pass  # silently skip missing attributes
            else:
                print('{:s} = {}'.format(k,a))  # print attribute value
        return

    def print_reg(self):
        '''
        Prints the best regularization parameter.
        Returns: None, prints bestreg to console
        '''
        print('best reg = {:.6e}'.format(self.bestreg))  # formatted output of best regularization value
        return

    def show_Pareto_front(self,size=3.5,
                              title=None,color='magenta',fname=None):
        '''Plots the Pareto front of smoothness vs relaxation time.
        size: float, base figure size
        title: str or None, optional plot title
        color: str, default plot color for main curve
        fname: str or None, if provided saves figure to file
        Returns: None, displays Pareto front plot
        '''
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)  # create figure
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        d = np.array(self.storing_dict['data_res'])  # data residuals
        si = np.array(self.storing_dict['smoothness'])  # smoothness values
        ti = np.array(self.storing_dict['tau_relax'])  # relaxation times

        filt = d <= self.keep_res  # accepted points mask
        s = si[filt]
        t = ti[filt]

        nf = np.logical_not(filt)  # rejected mask
        ns = si[nf]
        ts = ti[nf]

        plt.xscale('log')
        plt.yscale('log')
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'smoothness',fontsize=3*size)
        plt.ylabel(r'$\tau_{relax}$',fontsize=3*size)
        if title is None:
            plt.title('Pareto Front')
        else:
            plt.title(title)

        plt.plot(s,t,label='accepted',
                 ls='none',marker='o',color='green',markersize=1.7*size,fillstyle='none')

        pareto = []  # collect Pareto optimal indices
        for i,(si,ti) in enumerate(zip(s,t)):
            fs = si > s
            ft = ti > t
            f = np.logical_and(fs,ft)
            if f.any(): continue  # skip if dominated
            pareto.append(i)

        p = np.array(pareto,dtype=int)
        sp = s[p] ; tp = t[p]
        ser = sp.argsort()  # order by smoothness
        sp = sp[ser] ; tp = tp[ser]

        plt.plot(ns,ts,color='red',label='rejected',ls='none',
                 marker='o',fillstyle='none',markersize=1.7*size)
        plt.plot(sp,tp,ls='--',color='blue',lw=size/5,label='Opt. front')

        plt.plot([self.best_smoothness],[self.best_tau_relax],marker='o',
                 color='blue',markersize=1.7*size,label='selected')

        plt.xticks(fontsize=2.5*size)
        plt.legend(frameon=False,fontsize=1.5*size)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')  # save if requested
        plt.show()
        return

    def eps_omega(self,omega=[]):
        '''Computes dielectric loss spectrum e'(ω) and e''(ω).
        omega: array-like, angular frequencies
        Returns: tuple of arrays (eps_real, eps_imag)
        '''
        n = len(omega)
        eps_real = np.empty(len(omega),dtype=float)
        eps_imag = np.empty(len(omega),dtype=float)
        w,f = self.get_wrm()  # weights and modes
        if self.mode != 'freq':
            f = 1/f  # convert if needed
        for i in range(n):
            eps_real[i] = np.sum(w*f/(omega[i]**2+f**2))
            eps_imag[i] = -np.sum(-w*omega[i]/(omega[i]**2+f**2))
        tr = self.trelax  # normalization by relaxation time
        eps_real /= tr
        eps_imag /= tr
        self.omega = omega
        self.eps_real = eps_real
        self.eps_imag = eps_imag
        return eps_real,eps_imag

    def omega_peak(self,omega):
        '''Finds peak frequency (or relaxation time) where derivative of e'' crosses zero.
        omega: array-like, angular frequencies
        Returns: float, peak location (omega or tau)
        '''
        eps_real, eps_imag = self.eps_omega(omega)  # compute dielectric spectrum
        der = np.empty_like(eps_imag)
        der[0] = eps_imag[1]-eps_imag[0]
        der[-1] = eps_imag[-2] - eps_imag[-1]
        der[1:] = eps_imag[1:]-eps_imag[:-1]

        sign_change = []  # detect derivative zero-crossings
        for i in range(1,der.size):
            if der[i-1]*der[i] < 0:
                sign_change.append(i)

        if self.mode =='freq': arg = 0
        else: arg = -1

        if len(sign_change)==0:
            return omega[0]  # fallback

        return 1/omega[sign_change[arg]]

    def show_eps_omega(self,omega,e='imag',size=3.5,units='ns',yscale=None,
                              title=None,color='red',fname=None,
                              prefix='best_'):
        '''Plots dielectric spectrum e'(ω) or e''(ω).
        omega: array-like, angular frequencies
        e: str, 'imag' or 'real' component to plot
        size: float, figure size
        units: str, displayed x-axis units
        yscale: str or None, linear or log scale
        title: str or None, optional title
        color: str, plot color
        fname: str or None, save figure filename
        prefix: str, parameter prefix
        Returns: None, displays plot
        '''
        eps_real,eps_imag = self.eps_omega(omega)
        if e=='imag':
            eps = eps_imag
        elif e=='real':
            eps = eps_real
            yscale=None
        else:
            raise ValueError('option e = "{:s}" not known'.format(e))

        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        if yscale is not None:
            plt.yscale(yscale)
        if yscale =='log':
            y0 = -5
            ym = int(np.log10(max(eps))+1)
            plt.yticks([10**y for y in range(y0,ym )])
            plt.ylim(10**y0,10**ym)
        plt.yticks(fontsize=2.5*size)

        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        if e=='imag':
            ylabel=r'$e^{\prime\prime}(\omega)$'
        elif e =='real':
            ylabel = r"$e^{\prime}(\omega)$"
        plt.ylabel(ylabel,fontsize=3*size)

        if self.mode=='freq':
            units = r'${:s}^{:s}$'.format(units,'{-1}')
            lab = r'$f$ / {:s}'.format(units)
        elif self.mode =='tau':
            lab = r'$\tau$ / {:s}'.format(units)
        plt.xlabel(lab,fontsize=3*size)

        if title is not None:
            plt.title(title)

        plt.plot(omega,eps,
                 ls='-',marker='o',color=color,markersize=1.3*size,lw=0.2*size,fillstyle='none')

        xticks = [10**x for x in range(-10,20) if omega[0]<=10**x<=omega[-1] ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))

        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return

    def show_fit(self,size=3.5,units='ns',yscale=None,
                              title=None,color='red',fname=None,
                              prefix='best_'):
        '''Plots fitted decay curve P1(t) against data.
        size: float, figure size
        units: str, time axis units
        yscale: str or None, scaling of y-axis
        title: str or None, optional title
        color: str, curve color
        fname: str or None, save filename
        prefix: str, parameter prefix
        Returns: None, shows plot
        '''
        xlim = 10
        xf = np.logspace(-4,xlim,base=10,num=10000)  # high-resolution x-grid
        yee = self.fitted_curve(xf)  # model curve

        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')

        xticks = [10**x for x in range(-4,xlim+1)]
        plt.xticks(xticks,fontsize=2.5*size)
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'$t (ns)$',fontsize=3*size)
        plt.ylabel(r'$P_1(t)$',fontsize=3*size)

        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        plt.plot(xf,yee,ls='-.',color=color,label='fit')
        plt.ylim((-0.05,1))

        plt.plot(self.xdata,self.ydata,ls='none',marker='o',
            markersize=size*0.8,fillstyle='none',color=color,label='data')

        plt.legend(frameon=False,bbox_to_anchor=(1,1),fontsize=2.3*size)

        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return

    def show_relaxation_modes(self,size=3.5,units='ns',yscale=None,
                              title=None,color='red',fname=None,
                              show_contributions=False,prefix='best_'):
        '''Plots relaxation modes in either freq or tau representation.
        size: float, figure size
        units: str, x-axis unit string
        yscale: str or None, y-axis scaling
        title: str or None, optional title
        color: str, plot color
        fname: str or None, save filename
        show_contributions: bool, whether to plot weighted contributions
        prefix: str, parameter prefix
        Returns: None, displays relaxation mode distribution
        '''
        rm = getattr(self,prefix+'relaxation_modes')  # relaxation modes
        pars = getattr(self,prefix+'params')  # weights

        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')

        if yscale is not None:
            plt.yscale(yscale)

        if yscale =='log':
            y0 = -5
            if show_contributions:
                contr = pars/rm if self.mode=='freq' else pars*rm
                ym = int(np.log10(max(contr))+1)
            plt.yticks([10**y for y in range(y0,ym)])
            plt.ylim(10**y0,10**ym)

        plt.yticks(fontsize=2.5*size)

        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        plt.ylabel('w',fontsize=3*size)

        if self.mode=='freq':
            units = r'${:s}^{:s}$'.format(units,'{-1}')
            lab = r'$f$ / {:s}'.format(units)
        elif self.mode =='tau':
            lab = r'$\tau$ / {:s}'.format(units)
        plt.xlabel(lab,fontsize=3*size)

        if title is None:
            plt.title('Relaxation times distribution')
        else:
            plt.title(title)

        plt.plot(rm,pars,
                 ls='-',marker='o',color=color,markersize=1.3*size,lw=0.2*size,fillstyle='none')

        if show_contributions:
            contr = pars/rm if self.mode == 'freq' else pars*rm
            plt.plot(rm,contr,label='contribution',color='k',
                     ls='-',lw=0.1*size,marker='.',fillstyle='none',markersize=1.5*size,
                     markeredgewidth=size*0.5)

        xticks = [10**x for x in range(-10,20) if self.bounds[0]<=10**x<=self.bounds[1]]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))

        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return

    def fitted_data(self):
        '''Returns fitted decay values at xdata positions.
        Returns: array-like, fitted P1(t) values
        '''
        return self.fitted_curve(self.xdata)

    def fitted_curve(self,x):
        '''Evaluates model curve at given x positions.
        x: array-like, evaluation points
        Returns: array-like, model values
        '''
        if self.method =='distribution':
            return self.func(x,self.bounds[0],self.bounds[1],self.params) + self.bias
        else:
            return self.func(x,*self.params)



class fitKernels():
	@staticmethod
	def freq(t,f):
		'''Compute frequency-domain exponential kernel
		Parameters
		t : array-like, time values
		f : array-like, frequency values
		Returns
		array-like, exp(-t ⊗ f) kernel'''
		return np.exp(-np.outer(t,f))   # outer product → kernel matrix

	@staticmethod
	def tau(t,tau):
		'''Compute relaxation-time kernel
		Parameters
		t : array-like, time values
		tau : array-like, relaxation times
		Returns
		array-like, exp(-t ⊗ 1/tau) kernel'''
		return np.exp(-np.outer(t,(1/tau)))   # convert τ to rate = 1/τ


class fitFuncs():
	@staticmethod
	def gauss(t,a,b,w):
		'''Compute Gaussian-distributed relaxation function
		Parameters
		t : array-like, time values
		a : float, lower log-bound for frequencies
		b : float, upper log-bound for frequencies
		w : array-like, weights for each mode
		Returns
		array-like, relaxation curve'''
		frqs = fitData.get_logtimes(a,b,len(w))   # log-spaced frequencies
		s=0
		for i,fr in enumerate(frqs):
			s+= w[i]*np.exp(-t*t*fr)   # Gaussian-type dependence
		return s

	@staticmethod
	def freq(t,a,b,w):
		'''Compute sum of exponentials in frequency domain
		Parameters
		t : array-like, time values
		a : float, lower log-bound for frequencies
		b : float, upper log-bound for frequencies
		w : array-like, weights for each mode
		Returns
		array-like, relaxation curve'''
		frqs = fitData.get_logtimes(a,b,len(w))
		s=0
		for i,fr in enumerate(frqs):
			s+= w[i]*np.exp(-t*fr)   # exp(-t f)
		return s

	@staticmethod
	def tau(t,a,b,w):
		'''Compute sum of exponentials in τ-domain
		Parameters
		t : array-like, time values
		a : float, lower log-bound for τ
		b : float, upper log-bound for τ
		w : array-like, weights for each mode
		Returns
		array-like, relaxation curve'''
		taus = fitData.get_logtimes(a,b,len(w))
		s=0
		for i,ta in enumerate(taus):
			s+= w[i]*np.exp(-t*ta)   # exp(-t / τ_effective)
		return s

	@staticmethod
	def KWW(t,tww,beta,A=1):
		'''Compute stretched-exponential (KWW) function
		Parameters
		t : array-like, time values
		tww : float, characteristic relaxation time
		beta : float, stretching exponent
		A : float, prefactor
		Returns
		array-like, KWW relaxation curve'''
		# beta controls curve shape; beta→1 is exponential; beta<1 stretches
		phi = A*np.exp( -(t/tww)**beta )
		return phi

	@staticmethod
	def KWW2(t,tw1,tw2,b1,b2,A1=1,A2=1):
		'''Compute two-component KWW relaxation function
		Parameters
		t : array-like, time values
		tw1 : float, first relaxation time
		tw2 : float, second relaxation time
		b1 : float, first stretching exponent
		b2 : float, second stretching exponent
		A1 : float, amplitude of first component
		A2 : float, amplitude of second component
		Returns
		array-like, sum of two KWW curves'''
		phi1 = fitFuncs.KWW(t,tw1,b1,A1)
		phi2 = fitFuncs.KWW(t,tw2,b2,A2)
		return phi1 + phi2   # superposition

	@staticmethod
	def exp(t,t0,A=1):
		'''Compute single exponential relaxation
		Parameters
		t : array-like, time values
		t0 : float, decay time constant
		A : float, amplitude
		Returns
		array-like, exponential decay'''
		# A shifts starting amplitude
		phi = A*np.exp(-t/t0)
		return phi


class Analytical_Expressions():

	@staticmethod
	def expDecay_sum(t,t0v):
		'''Compute average of multiple simple exponential decays
		Parameters
		t : array-like, time values
		t0v : array-like, decay constants
		Returns
		array-like, mean of exponentials'''
		s = np.zeros(t.shape[0])
		t0v = np.array(t0v)
		for i,t0 in enumerate(t0v):
			s+=Analytical_Expressions.expDecay_simple(t,t0)   # sum each decay
		return s/t0v.shape[0]   # average

	@staticmethod
	def expDecay_simple(t,t0):
		'''Compute single simple exponential decay
		Parameters
		t : array-like, time values
		t0 : float, decay constant
		Returns
		array-like, exp(-t/t0)'''
		phi =  np.exp(-t/t0)
		return phi

	@staticmethod
	def expDecay(t,A,t0):
		'''Exponential decay with offset for endpoint
		Parameters
		t : array-like, time values
		A : float, shift factor for endpoint
		t0 : float, decay constant
		Returns
		array-like, 1 + A*(exp(-t/t0)-1)'''
		phi = 1 + A*( np.exp(-t/t0) - 1 )   # shift endpoint
		return phi

	@staticmethod
	def expDecay2(t,A,t0):
		'''Exponential decay with starting amplitude shift
		Parameters
		t : array-like, time values
		A : float, starting amplitude
		t0 : float, decay constant
		Returns
		array-like, A*exp(-t/t0)'''
		phi = A*np.exp(-t/t0)
		return phi

	@staticmethod
	def expDecay3(t,A,t0):
		'''Exponential decay shifted to endpoint -A
		Parameters
		t : array-like, time values
		A : float, amplitude shift
		t0 : float, decay constant
		Returns
		array-like, A*(exp(-t/t0)-1)'''
		phi = A*(np.exp(-t/t0)-1)
		return phi

	@staticmethod
	def expDecay4(t,As,Ae,t0):
		'''Exponential decay with independent start and endpoint shifts
		Parameters
		t : array-like, time values
		As : float, starting amplitude shift
		Ae : float, endpoint shift
		t0 : float, decay constant
		Returns
		array-like, As*exp(-t/t0)-Ae'''
		phi = As*np.exp(-t/t0)-Ae
		return phi

	@staticmethod
	def expDecay_KWW(t,A1,A2,tc,t0,beta,tww):
		'''Combined exponential and KWW (stretched exponential) decay
		Parameters
		t : array-like, time values
		A1 : float, amplitude of initial exponential
		A2 : float, amplitude for KWW (updated for continuity)
		tc : float, crossover time
		t0 : float, decay constant for exponential
		beta : float, stretching exponent for KWW
		tww : float, characteristic KWW relaxation time
		Returns
		array-like, combined exponential + KWW decay curve'''
		tl =  t[t<tc]   # times before crossover
		tup = t[t>=tc]  # times after crossover
		phil = Analytical_Expressions.expDecay(tl,A1,t0)   # exponential part
		A2 = Analytical_Expressions.expDecay(tc,A1,t0)   # continuity at tc
		phiup = Analytical_Expressions.KWW(tup,A2,tc,beta,tww)   # KWW part
		return np.concatenate( (phil,phiup) )   # full curve


@jitclass
class Analytical_Functions():
	'''
	This class is used exclusively to add hydrogens on correct locations
	and compute all atom properties like dipole moment vectors (dielectric data)
	Currently works well for PB. It is NOT tested for anything else.
	It needs attention to geometry calculations and should be used
	with the add_atoms class.
	'''

	def __init__(self):
		pass

	@staticmethod
	def Rz(th):
		'''Rotation matrix around z-axis
		Parameters
		th : float, angle in radians
		Returns
		3x3 np.array, rotation matrix'''
		R = np.array([[1, 0 ,0],
			[0, np.cos(th), -np.sin(th)],
			[0, np.sin(th), np.cos(th)]])
		return R

	@staticmethod
	def Ry(th):
		'''Rotation matrix around y-axis
		Parameters
		th : float, angle in radians
		Returns
		3x3 np.array, rotation matrix'''
		R = np.array([[np.cos(th), 0, np.sin(th)] ,
					   [0, 1, 0],
					   [-np.sin(th), 0, np.cos(th)]])
		return R

	@staticmethod
	def Rx(th):
		'''Rotation matrix around x-axis
		Parameters
		th : float, angle in radians
		Returns
		3x3 np.array, rotation matrix'''
		R = np.array([[np.cos(th), -np.sin(th), 0],
					 [np.sin(th), np.cos(th), 0],
					 [0,0,1]])
		return R

	@staticmethod
	def q_mult(q1,q2):
		'''Multiply two quaternions
		Parameters
		q1 : array-like (4,), first quaternion
		q2 : array-like (4,), second quaternion
		Returns
		np.array (4,), resulting quaternion'''
		w1 = q1[0] ; x1 = q1[1] ; y1 = q1[2] ; z1 = q1[3]
		w2 = q2[0] ; x2 = q2[1] ; y2 = q2[2] ; z2 = q2[3]
		w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
		x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
		y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
		z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
		return np.array((w, x, y, z))

	@staticmethod
	def quaternionConjugate(q):
		'''Compute conjugate of a quaternion
		Parameters
		q : array-like (4,), quaternion
		Returns
		np.array (4,), conjugated quaternion'''
		w = q[0] ; x = q[1] ; y = q[2] ; z = q[3]
		return np.array((w, -x, -y, -z))

	@staticmethod
	def rotate_around_an_axis(axis,r,theta):
		'''Rotate a vector around arbitrary axis using quaternions
		Parameters
		axis : array-like (3,), rotation axis
		r : array-like (3,), vector to rotate
		theta : float, rotation angle in radians
		Returns
		np.array (3,), rotated vector'''
		q1 = (0,r[0],r[1],r[2])
		th2 = 0.5*theta
		c = np.cos(th2)
		s = np.sin(th2)
		q2 = np.array((c,s*axis[0],s*axis[1],s*axis[2]))
		q2c = Analytical_Functions().quaternionConjugate(q2)
		q = Analytical_Functions().q_mult(q1,q2c)
		q3 = Analytical_Functions().q_mult(q2,q)
		return np.array((q3[1],q3[2],q3[3]))

	@staticmethod
	def rotate_to_theta_target(rp,r0,rrel,theta_target):
		'''Rotate rrel to achieve target angle between rp-r0 and r0+rrel
		Parameters
		rp : array-like (3,), reference point
		r0 : array-like (3,), center point
		rrel : array-like (3,), vector to rotate
		theta_target : float, target angle in radians
		Returns
		rrot : array-like (3,), rotated vector
		newth : float, resulting angle'''
		th0 = calc_angle(rp,r0,r0+rrel)
		theta = theta_target - th0
		naxis = np.cross((rp-r0)/norm2(rp-r0),rrel/norm2(rrel))
		rrot = Analytical_Functions().rotate_around_an_axis(naxis, rrel, theta)
		newth = calc_angle(rp,r0,r0+rrot)
		return rrot,newth

	@staticmethod
	def position_hydrogen_analytically_endgroup(bondl,theta,r1,r0,nh=1):
		'''Place hydrogen analytically for endgroup
		Parameters
		bondl : float, bond length
		theta : float, bond angle in radians
		r1 : array-like (3,), coordinate of neighbor atom
		r0 : array-like (3,), coordinate of central atom
		nh : int, number of hydrogens to place (1 or more)
		Returns
		np.array (3,) or array-like, coordinates of added hydrogen(s)'''
		r01 = r1-r0
		ru01 = r01/norm2(r01)
		rp = r0 + bondl*ru01
		dhalf = bondl/np.sqrt(3)
		s = -np.sign(r01)
		rrel = np.array([s[0]*dhalf,s[1]*dhalf,-s[2]*dhalf])
		newth = theta +1
		af = Analytical_Functions()
		while np.abs(newth - theta)>1.e-4:
			rrel,newth = af.rotate_to_theta_target(rp, r0, rrel, theta)
		r2 = r0 + rrel
		if nh == 1:
			return r2
		rn = af.position_hydrogen_analytically(bondl,theta,rp,r0,r2,nh-1)
		return rn

	@staticmethod
	def position_hydrogen_analytically_cis(bondl,theta,r1,r0,r2,nh=1):
		'''Place hydrogen analytically for cis geometry
		Parameters
		bondl : float, bond length
		theta : float, bond angle in radians
		r1 : array-like (3,), left atom
		r0 : array-like (3,), central atom
		r2 : array-like (3,), right atom
		nh : int, number of hydrogens
		Returns
		np.array (3,), coordinates of added hydrogen'''
		r01 = r1-r0
		r02 = r2-r0
		r1 = r0 + bondl*(r01)/norm2(r01)
		r2 = r0 + bondl*(r02)/norm2(r02)
		rm = 0.5*(r1+r2)
		ru2 = r0 - rm ; u2 = ru2/norm2(ru2)
		rn = r0 + bondl*u2
		return rn

	@staticmethod
	def position_hydrogen_analytically(bondl_h,theta,r1,r0,r2,nh=1):
		'''Analytical hydrogen placement for CH2 or CH3
		Parameters
		bondl_h : float, bond length
		theta : float, bond angle in radians
		r1 : array-like (3,), left atom
		r0 : array-like (3,), central atom
		r2 : array-like (3,), right atom
		nh : int, which hydrogen (1 or 2)
		Returns
		rh : array-like (3,), coordinates of added hydrogen'''
		r01 = r1-r0
		r02 = r2-r0
		r1 = r0 + bondl_h*(r01)/norm2(r01)
		r2 = r0 + bondl_h*(r02)/norm2(r02)
		rm = 0.5*(r1+r2)
		ru1 = r2 - r1 ; u1 = ru1/norm2(ru1)
		ru2 = r0 - rm ; u2 = ru2/norm2(ru2)
		u3 = np.cross(u1,u2)
		a = theta/2
		if nh ==1:
			rh = r0 + bondl_h*(np.cos(a)*u2 + np.sin(a)*u3)
		elif nh ==2:
			rh = r0 + bondl_h*(np.cos(a)*u2 - np.sin(a)*u3)
		return rh


class add_atoms():
    '''
    This class is used to add atoms to the system.
    Currently works well to add hydrogens to PB.
    It needs modification to achieve generality.
    '''

    def __init__(self):
        pass

    # Predefined hydrogen mapping for different atom types: [number, bond_length, bond_angle, attachment_indices, type]
    hydrogen_map = {'CD':[1,0.11,116,(1,),'_cis'],
                    'C':[2,0.11,109.47,(1,2),''],
                    'CE':[3,0.109,109.47,(1,2,3),'_endgroup']}

    @staticmethod
    def add_ghost_atoms(self, system2, gconnectivity=dict()):
        '''Add ghost atoms from another system
        Parameters
        self : add_atoms instance
        system2 : system object, source of ghost atoms
        gconnectivity : dict, optional, connectivity for ghost atoms
        Returns
        None'''

        # Copy atom metadata from system2
        for s in ['at_ids','at_types','mol_names','mol_ids']:
            a1 = getattr(system2, s)
            setattr(self, '_'.join(('ghost', s)), a1)

        # Copy coordinates for each timeframe
        for frame in system2.timeframes:
            c1 = system2.timeframes[frame]['coords']
            self.timeframes[frame]['_'.join(('ghost', 'coords'))] = c1

        # Set ghost connectivity and update mass map
        self.ghost_connectivity = gconnectivity
        self.mass_map.update(system2.mass_map)
        return

    @staticmethod
    def append_atoms(self, k='ghost'):
        '''Append ghost atoms to real atoms
        Parameters
        self : add_atoms instance
        k : str, prefix for ghost atoms
        Returns
        None'''

        t0 = perf_counter()  # track total initialization time

        # Concatenate metadata arrays
        for s in ['at_ids','at_types','mol_names','mol_ids']:
            a1 = getattr(self, s)
            a2 = getattr(self, '_'.join([k, s]))
            n12 = np.concatenate((a1, a2))
            setattr(self, s, n12)

        # Concatenate coordinates per frame
        for frame in self.timeframes:
            c1 = self.timeframes[frame]['coords']
            c2 = self.timeframes[frame]['_'.join((k,'coords'))]
            c12 = np.concatenate((c1, c2))
            self.timeframes[frame]['coords'] = c12

        # Update connectivity and reinitialize topology
        self.connectivity.update(self.ghost_connectivity)
        self.topology_initialization()

        # Special initialization for confined systems
        if self.__class__.__name__ == 'Analysis_Confined':
            self.confined_system_initialization()

        # Log elapsed time
        ass.print_time(perf_counter() - t0,
                       inspect.currentframe().f_code.co_name, frame + 1)
        return

    @staticmethod
    def add_ghost_hydrogens(self, types, noise=None, pickleframes=False):
        '''Add ghost hydrogens to system
        Parameters
        self : add_atoms instance
        types : list of atom types for H addition
        noise : optional, coordinate noise
        pickleframes : bool, whether to save frames to pickle
        Returns
        None'''

        t0 = perf_counter()  # full cluster-detection and statistics pass

        # Compute new atom information
        new_finfo = add_atoms.get_new_atoms_info(self, 'h', types)
        self.ghost_atoms_info = new_atoms_info

        # Set connectivity and topology for ghost atoms
        add_atoms.set_ghost_connectivity(self, new_atoms_info)
        add_atoms.assign_ghost_topol(self, new_atoms_info)

        # Set coordinates for all ghost atoms
        add_atoms.set_all_ghost_coords(self, new_atoms_info, noise, pickleframes=pickleframes)

        # Log elapsed time
        tf = perf_counter() - t0
        ass.print_time(tf, inspect.currentframe().f_code.co_name)
        return

    @staticmethod
    def set_ghost_connectivity(self, info):
        '''Set connectivity for ghost atoms
        Parameters
        self : add_atoms instance
        info : dict, atom information
        Returns
        None'''

        gc = dict()
        for j, v in info.items():
            # Store connectivity as (atom_type_index, hydrogen_type)
            gc[(v['bw'], j)] = (self.at_types[v['bw']], v['ty'])
        self.ghost_connectivity = gc
        return


    @staticmethod
    def set_all_ghost_coords(self, info, noise=None, pickleframes=False):
        '''Compute and set coordinates for all ghost atoms
        Parameters
        self : add_atoms instance
        info : dict, atom information
        noise : optional, random noise
        pickleframes : bool, save coordinates to pickle
        Returns
        None'''

        # Serialize atom info for parallel processing
        f, l, th, s, ir1, ir0, ir2 = add_atoms.serialize_info(info)
        run = False

        # Pickle file path for caching
        fpickle = '{:s}_N{:d}_.pickle'.format(self.trajectory_file, self.nframes)
        try:
            with open(fpickle, 'rb') as handle:
                timeframes = pickle.load(handle)
                logger.info('Done: Read from {} '.format(fpickle))
                self.timeframes = timeframes
        except:
            run = True

        if run or pickleframes == False:
            # Loop over frames to calculate ghost coordinates
            for frame in self.timeframes:
                ghost_coords = np.empty((len(info), 3), dtype=float)
                coords = self.get_coords(frame)
                add_atoms.set_ghost_coords_parallel(f, l, th, s, ir1, ir0, ir2,
                                                    coords, ghost_coords)
                self.timeframes[frame]['ghost_coords'] = ghost_coords
                if frame % 1000 == 0:
                    logger.info('Done: setting ghost coords frame {}'.format(frame))

            # Pickle results if required
            if pickleframes:
                logger.info('Done: pickling to {} '.format(fpickle))
                with open(fpickle, 'wb') as handle:
                    try:
                        pickle.dump(self.timeframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        os.system('rm {}'.format(fpickle))

        return


    @staticmethod
    def serialize_info(info):
        '''Serialize atom info for parallel processing
        Parameters
        info : dict
            Dictionary containing atom information
        Returns
        f : int array, frame type
        l : float array, bond lengths
        th : float array, bond angles
        s : int array, additional sign info
        ir1 : int array, first reference atom index
        ir0 : int array, central atom index
        ir2 : int array, second reference atom index (0 if not present)'''

        n = len(info)
        f = np.empty(n,dtype=int)
        l = np.empty(n,dtype=float)
        th = np.empty(n,dtype=float)
        s = np.empty(n,dtype=int)
        ir1 = np.empty(n,dtype=int)
        ir0 = np.empty(n,dtype=int)
        ir2 = np.zeros(n,dtype=int)  # default 0 for missing third reference

        # Populate arrays from info dict
        for j,(k,v) in enumerate(info.items()):
            f[j] = v['f']
            l[j] = v['l']
            th[j] = v['th']
            s[j] = v['s']
            ir1[j] = v['ir'][0]
            ir0[j] = v['ir'][1]
            if len(v['ir'])==3:
                ir2[j] = v['ir'][2]

        return f,l,th,s,ir1,ir0,ir2

    @staticmethod
    @jit(nopython=True,fastmath=True,parallel=True)
    def set_ghost_coords_parallel(f,l,th,s,ir1,ir0,ir2,
                                  coords,ghost_coords):
        '''Compute ghost atom coordinates in parallel
        Parameters
        f : int array, frame type
        l : float array, bond lengths
        th : float array, angles
        s : int array, sign information
        ir1 : int array, first reference atom index
        ir0 : int array, central atom index
        ir2 : int array, second reference atom index
        coords : float array, coordinates of all atoms in frame
        ghost_coords : float array, output coordinates of ghost atoms
        Returns
        None'''

        N = f.shape[0]
        af = Analytical_Functions()  # instance of analytical functions

        # Loop over each ghost atom in parallel
        for j in prange(N):
            cr1 = coords[ir1[j]]
            cr0 = coords[ir0[j]]
            cr2 = coords[ir2[j]]

            # Determine placement method based on type
            if   f[j] == 1:  # cis placement
                rn = af.position_hydrogen_analytically_cis(l[j],th[j],cr1,cr0,cr2,s[j])
            elif f[j] == 2:  # general CH2 placement
                rn = af.position_hydrogen_analytically(l[j],th[j],cr1,cr0,cr2,s[j])
            elif f[j] == 3:  # endgroup placement
                rn = af.position_hydrogen_analytically_endgroup(l[j],th[j],cr1,cr0,s[j])

            ghost_coords[j] = rn
        return

    @staticmethod
    def set_ghost_coords(self,info):
        '''Compute ghost coordinates for a single frame
        Parameters
        self : add_atoms instance
        info : dict, ghost atom info
        Returns
        None'''

        N = len(info)
        frame = self.current_frame
        ghost_coords = np.empty((N,3),dtype=float)
        coords = self.get_whole_coords(frame)  # get full frame coordinates

        # Loop over all ghost atoms and compute positions
        for j,(k,v) in enumerate(info.items()):
            cr = coords[v['ir']]
            rn = v['func'](v['l'],v['th'],*cr,v['s'])
            ghost_coords[j] = rn

        self.timeframes[frame]['ghost_coords'] = ghost_coords
        return

    @staticmethod
    def assign_ghost_topol(self,info):
        '''Assign topology information for ghost atoms
        Parameters
        self : add_atoms instance
        info : dict, ghost atom information
        Returns
        None'''

        n = len(info)
        gtypes = np.empty(n,dtype=object)
        gmol_names = np.empty(n,dtype=object)
        gat_ids = np.empty(n,dtype=int)
        gmol_ids = np.empty(n,dtype=int)

        # Fill arrays from info dict
        for i,(k,v) in enumerate(info.items()):
            gtypes[i] = v['ty']         # atom type
            gmol_names[i] = v['res_nm'] # residue name
            gmol_ids[i] = v['res']      # residue ID
            gat_ids[i] = k              # atom ID

        # Assign to class variables
        self.ghost_at_types = gtypes
        self.ghost_mol_names = gmol_names
        self.ghost_at_ids = gat_ids
        self.ghost_mol_ids = gmol_ids
        return

    @staticmethod
    def get_new_atoms_info(self,m,types):
        '''Compute information for new atoms to be added
        Parameters
        self : add_atoms instance
        m : str, atom symbol (only 'h' supported)
        types : list of types to add
        Returns
        new_atoms_info : dict
            Contains ir, bond length, angle, sign, type, residue info, placement function, and frame type'''

        if m =='h':
            type_map = add_atoms.hydrogen_map
            self.mass_map.update({m+ty:maps.elements_mass['H'] for ty in types})
        else:
            raise NotImplementedError('Currently the only option is to add hydrogens')

        # Retrieve bond and angle connectivity
        bond_ids = np.array(list(self.connectivity.keys()))
        ang_ids = np.array(list(self.angles.keys()))

        at_types = self.at_types
        residues = self.mol_ids
        res_nms = self.mol_names

        jstart = self.at_types.shape[0]  # starting index for new atoms
        new_atoms_info = dict()
        fu = 'position_hydrogen_analytically'

        # Loop over existing atoms to determine where to add new hydrogens
        for t,ty in enumerate(at_types):
            if ty not in types:
                continue
            tm = type_map[ty]
            i = self.at_ids[t]

            # Determine reference atoms
            try:
                ir = ang_ids[np.where(ang_ids[:,1] == i)][0]
            except IndexError:
                try:
                    ir = bond_ids[np.where(bond_ids[:,0]==i)][0]
                    ir = np.array([ir[1],ir[0]])
                except IndexError:
                    try:
                        ir = bond_ids[np.where(bond_ids[:,1]==i)][0]
                    except IndexError as exc:
                        raise Exception('{}: Could not find bonds for atom {:d}, type = {:s}'.format(exc,i,ty))
            finally:
                func = getattr(Analytical_Functions(),fu+tm[4])

            # Determine frame type based on hydrogen placement
            if tm[4] == '_cis': f = 1
            elif tm[4] =='' : f = 2
            elif tm[4] =='_endgroup': f = 3

            # Add entries for all hydrogens for this atom
            for j in range(tm[0]):
                new_atoms_info[jstart] = {'ir':ir,'l':tm[1],'th':tm[2]*np.pi/180.0,'s':tm[3][j], 'bw': i,
                                         'ty':m+ty,'res':residues[i],'res_nm':res_nms[i],
                                         'func':func,'f':f}
                jstart+=1

        return new_atoms_info


class Distance_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    These class functions are used to calculate
    the Distance between coords and a center position (usually particle center of mass is passed)
    '''
    @staticmethod
    def zdir(coords,cref):
        return np.abs(coords[:,2] - cref[2])
    @staticmethod
    def ydir(coords,cref):
        return np.abs(coords[:,1] - cref[1])
    @staticmethod
    def xdir(coords,cref):
        return np.abs(coords[:,0] - cref[0])

    @staticmethod
    def spherical(coords,cref):
        d = np.zeros(coords.shape[0],dtype=float)
        distance_kernel(d,coords,cref)
        #r = coords -cref
        #d = np.sqrt(np.sum(r*r,axis=1))
        return d

    @staticmethod
    def minimum_distance(coords1,coords2):
        d1 = np.empty(coords1.shape[0])
        d2 = np.empty(coords2.shape[0])
        smaller_distance_kernel(d1,d2,coords1,coords2)
        return d1

    @staticmethod
    def zcylindrical(self,coords,cref):
         r = coords[:,0:2]-c[0:2]
         d = np.sqrt(np.sum(r*r,axis=1))
         return d

class Box_Additions():
    '''
    Depending on the confinement type one of these functions
    will be called. These class functions are used to calculate
    the minimum image distance or box-related vectors.
    '''

    @staticmethod
    def zdir(box):
        '''
        Returns the displacement along z-axis considering box boundaries
        box : array-like
            Box dimensions [Lx,Ly,Lz]

        Returns
        -------
        list of float
            [positive_z, 0, negative_z] displacements along z
        '''
        return [box[2],0,-box[2]]  # +Lz, 0, -Lz

    @staticmethod
    def ydir(box):
        '''
        Returns the displacement along y-axis considering box boundaries
        box : array-like
            Box dimensions [Lx,Ly,Lz]

        Returns
        -------
        list of float
            [positive_y, 0, negative_y] displacements along y
        '''
        return [box[1],0,-box[1]]  # +Ly, 0, -Ly

    @staticmethod
    def xdir(box):
        '''
        Returns the displacement along x-axis considering box boundaries
        box : array-like
            Box dimensions [Lx,Ly,Lz]

        Returns
        -------
        list of float
            [positive_x, 0, negative_x] displacements along x
        '''
        return [box[0],0,-box[0]]  # +Lx, 0, -Lx

    @staticmethod
    def minimum_distance(box):
        '''
        Computes all 27 combinations of box translations for minimum image convention
        box : array-like
            Box dimensions [Lx,Ly,Lz]

        Returns
        -------
        list of ndarray
            List of 27 vectors representing shifts along x,y,z
        '''
        zd = Box_Additions.zdir(box)
        yd = Box_Additions.ydir(box)
        xd = Box_Additions.xdir(box)
        # Cartesian product of shifts along x, y, z
        lst_L = [np.array([x,y,z]) for x in xd for y in yd for z in zd]
        return lst_L

    @staticmethod
    def spherical(box):
        '''
        Computes box displacement vectors for a spherical particle
        box : array-like
            Box dimensions [Lx,Ly,Lz]

        Returns
        -------
        ndarray
            Array of 27 vectors representing shifts along x,y,z
        '''
        zd = Box_Additions.zdir(box)
        yd = Box_Additions.ydir(box)
        xd = Box_Additions.xdir(box)
        # Cartesian product of shifts along x, y, z as numpy array
        lst_L = np.array([np.array([x,y,z]) for x in xd for y in yd for z in zd])
        return lst_L

    @staticmethod
    def zcylindrical(box):
        '''
        Returns displacement along z for cylindrical confinement
        box : array-like
            Box dimensions [Lx,Ly,Lz]

        Returns
        -------
        list of float
            [0] as only z shift is relevant
        '''
        return [0]  # Only z is relevant in cylindrical confinement



class bin_Volume_Functions():
    '''
    Depending on the confinement type one of these functions
    will be called. These class functions are used to calculate
    the volume of each bin when needed (e.g., for density profile calculations).
    '''

    @staticmethod
    def zdir(self,bin_low,bin_up):
        '''
        Compute bin volume for slab along z-axis.
        bin_low : float
            Lower boundary of the bin
        bin_up : float
            Upper boundary of the bin

        Returns
        -------
        float
            Volume of the bin
        '''
        box = self.get_box(self.current_frame)  # Get current simulation box dimensions
        binl = bin_up - bin_low  # Bin width along z
        return  box[0] * box[1] * binl  # Volume = area * height

    @staticmethod
    def ydir(self,bin_low,bin_up):
        '''
        Compute bin volume for slab along y-axis.
        '''
        box = self.get_box(self.current_frame)
        binl = bin_up - bin_low
        return  box[0] * box[2] * binl  # Volume = area * height

    @staticmethod
    def xdir(self,bin_low,bin_up):
        '''
        Compute bin volume for slab along x-axis.
        '''
        box = self.get_box(self.current_frame)
        binl = bin_up - bin_low
        return box[1] * box[2] * binl  # Volume = area * height

    @staticmethod
    def zcylindrical(self,bin_low,bin_up):
        '''
        Compute bin volume for a cylinder along z-axis.
        '''
        box = self.get_box(self.current_frame)
        # Volume of cylindrical shell: π*(R_outer^2 - R_inner^2)*height
        return np.pi * (bin_up**2 - bin_low**2) * box[2]

    @staticmethod
    def spherical(self,bin_low,bin_up):
        '''
        Compute bin volume for spherical shells.
        '''
        # Volume of spherical shell: 4/3 * π * (R_outer^3 - R_inner^3)
        v = 4 * np.pi * (bin_up**3 - bin_low**3) / 3
        return v

class unit_vector_Functions():
    '''
    Depending on the confinement type one of these functions
    will be called. A unit vector for computing bond order is defined.
    '''

    @staticmethod
    def zdir(self, r, c):
        '''
        Unit vectors along z-direction.
        r : ndarray (N,3)
            Atomic positions
        c : ndarray (3,)
            Reference center (unused here)

        Returns
        -------
        uv : ndarray (N,3)
            Unit vectors along z
        '''
        uv = np.zeros((r.shape[0], 3))
        uv[:, 2] = 1  # Set z-component to 1
        return uv

    @staticmethod
    def ydir(self, r, c):
        '''
        Unit vectors along y-direction.
        '''
        uv = np.zeros((r.shape[0], 3))
        uv[:, 1] = 1  # Set y-component to 1
        return uv

    @staticmethod
    def xdir(self, r, c):
        '''
        Unit vectors along x-direction.
        '''
        uv = np.zeros((r.shape[0], 3))
        uv[:, 0] = 1  # Set x-component to 1
        return uv

    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True)
    def spherical_inner(cperiodic, r, c, box_add):
        '''
        Finds the closest periodic image of a particle for a spherical confinement.
        cperiodic : ndarray (N,3)
            Array to store adjusted particle positions
        r : ndarray (N,3)
            Original particle positions
        c : ndarray (3,)
            Reference center
        box_add : list of ndarray
            List of box translations for minimum image convention
        '''
        for i in prange(cperiodic.shape[0]):
            dist_min = 1e16
            for j, L in enumerate(box_add):
                rr = r[i] - (c + L)
                dist = np.sum(rr * rr)  # Squared distance
                if dist < dist_min:
                    jmin = j
                    dist_min = dist
            cperiodic[i] = c + box_add[jmin]
        return

    @staticmethod
    def spherical(self, r, c):
        '''
        Unit vectors for spherical particle confinement.
        r : ndarray (N,3)
            Atomic positions
        c : ndarray (3,)
            Particle center
        '''
        box = self.get_box(self.current_frame)  # Get box dimensions
        box_add = Box_Additions.spherical(box)  # List of periodic translations
        cperiodic = np.empty(r.shape)
        # Adjust particle positions for minimum image
        unit_vector_Functions.spherical_inner(cperiodic, r, c, box_add)
        uv = r - cperiodic  # Vector from adjusted center
        return uv

    @staticmethod
    def zcylindrical(self, r, c):
        '''
        Unit vectors for cylindrical confinement along z-axis.
        '''
        uv = np.ones((r.shape[0], 3))
        uv[:, 2] = 0  # Remove z-component for cylindrical symmetry
        return uv


class add_sudo_atoms:
    '''
    Adds "sudo" atoms to a system object in 3D space.
    Can place them randomly, uniformly, or with separation constraints.
    '''
    def __init__(self, obj1, num, sigma, frame=0, sep_dist=3.0, decrease_rate=0.9, positions=None,
                 r='no', rd='z', rl=0, rh=0):
        self.obj1 = obj1  # System object to add atoms to
        self.nsudos = num  # Number of sudo atoms
        self.sigma = sigma  # Sigma parameter (unused for placement here)
        self.frame = frame  # Frame index
        self.r = r  # Restriction type for placement
        self.rd = rd  # Restriction direction
        self.rl = rl  # Restriction lower bound
        self.rh = rh  # Restriction upper bound
        self.sep_dist = sep_dist  # Minimum separation distance
        self.decrease_rate = decrease_rate  # Rate to reduce separation if placement fails
        self.grid = self.d3grid()  # Grid for uniform placement

        if positions is None:
            self.find_positions()  # Generate positions automatically
        else:
            self.positions = positions  # Use provided positions
        self.update_topology()  # Merge sudo atoms into system topology
        return

    def d3grid(self):
        '''
        Generate a 3D grid based on the system box and number of atoms.
        Returns
        -------
        grid : ndarray (3,)
            Number of divisions along each box axis
        '''
        boxsort = self.obj1.get_box(self.frame).argsort()[::-1]  # Sort box dimensions descending
        grid = np.array([0, 0, 0])
        i = 0
        while grid.prod() < self.nsudos:  # Ensure grid can fit all atoms
            j = i % 3
            grid[j] += 1
            i += 1
        grid[(j + 1) % 3] += 1
        return grid[boxsort]

    @property
    def position_distances_nonperiodic(self):
        '''
        Compute distances between sudo atoms ignoring periodic boundaries.
        Returns
        -------
        dists : ndarray
            Pairwise distances
        '''
        dists = []
        for i, c1 in enumerate(self.positions):
            c2 = self.positions[i + 1:]
            r = c2 - c1
            d = np.sum(r * r, axis=1) * 0.5  # Squared distance scaled
            dists.extend(d)
        return np.array(dists)

    @property
    def position_distances(self):
        '''
        Compute distances between sudo atoms using minimum image convention.
        Returns
        -------
        dists : ndarray
            Pairwise distances considering periodicity
        '''
        dists = []
        box = self.obj1.get_box(self.frame)
        for i, c1 in enumerate(self.positions):
            c2 = self.positions[i + 1:]
            d = React_two_systems.minimum_image_distance(c2, c1, box)
            dists.extend(d)
        return np.array(dists)

    def find_positions(self):
        '''
        Wrapper function to generate positions with separation constraints.
        '''
        self.separation_positions()
        return

    def random_positions(self):
        '''
        Place sudo atoms randomly within the box.
        '''
        box = self.obj1.get_box(self.frame)
        self.positions = np.random.uniform([0, 0, 0], box, (self.nsudos, 3))
        return

    def separation_positions(self):
        '''
        Place sudo atoms with minimum separation distance and optional restrictions.
        '''
        box = self.obj1.get_box(self.frame)
        sudo_coords = []
        sep_dist = self.sep_dist
        r, rd, rl, rh = self.r, self.rd, self.rl, self.rh
        restricted = True

        if r == 'no':
            restricted = False
        elif r == 'center':
            cr = box / 2
        elif r in self.obj1.at_types or r in self.obj1.mol_names:
            if r in self.obj1.at_types:
                fc = self.obj1.at_types == r
            elif r in self.obj1.mol_names:
                fc = self.obj1.mol_names == r
            c = self.obj1.get_coords(self.frame)[fc]
            am = self.obj1.atom_mass[fc]
            cr = CM(c, am)  # Center of mass
        else:
            raise Exception('r = {:s} is not valid; use center, atom type, or molecule name'.format(r))

        for jadd in range(self.nsudos):
            accepted = False
            jfailed = 0
            while not accepted:
                pos = np.random.uniform([0, 0, 0], box)
                if restricted:
                    rc = pos - cr
                    if rd == 'x' and rl <= rc[0] <= rh: continue
                    if rd == 'y' and rl <= rc[1] <= rh: continue
                    if rd == 'z' and rl <= rc[2] <= rh: continue
                    if rd == 'd':
                        d = np.sum(rc * rc) ** 0.5
                        if rl <= d <= rh: continue
                if len(sudo_coords) == 0:
                    accepted = True
                else:
                    c_comp = np.array(sudo_coords)
                    dists = React_two_systems.minimum_image_distance(c_comp, pos, box)
                    if not (dists > sep_dist).all():
                        jfailed += 1
                    else:
                        accepted = True
                if jfailed >= 30:
                    jfailed = 0
                    print('Separation {:4.3f} too big for sudo atom {:d}. Reducing to {:4.3f}'.format(
                        sep_dist, jadd, sep_dist * self.decrease_rate))
                    sep_dist *= self.decrease_rate

            sudo_coords.append(pos)

        print(sudo_coords)
        self.positions = np.array(sudo_coords)
        return

    def uniform_positions(self):
        '''
        Place sudo atoms uniformly on a grid inside the box.
        '''
        box = self.obj1.get_box(self.frame)
        dx, dy, dz = box / self.grid
        gx, gy, gz = int(self.grid[0]), int(self.grid[1]), int(self.grid[2])
        positions = []
        num = 0
        for i in range(gx):
            x = (i + 1) * dx / 2
            for j in range(gy):
                y = (j + 1) * dy / 2
                for k in range(gz):
                    if num >= self.nsudos:
                        break
                    z = (k + 1) * dz / 2
                    positions.append(np.array([x, y, z]))
                    num += 1
        self.positions = np.array(positions)
        return

    def update_topology(self):
        '''
        Merge sudo atoms into the system topology as a new Topology object.
        '''
        box = self.obj1.get_box(self.frame)
        timeframes = {0: {'coords': self.positions, 'boxsize': box, 'time': 0, 'step': 0}}
        natoms = self.positions.shape[0]
        ty = 'SUDO'
        tyff = {ty: (ty, '10000.000', '0.0', 'A', '0.01', '1E-06')}  # Force field info
        obj2 = Topology(natoms, at_types=ty, atom_code=ty, timeframes=timeframes, atomtypes=tyff)
        self.obj2 = obj2
        self.obj1.merge_system(self.obj2)  # Merge into original system
        return




class React_two_systems:
    """Reactive merging workflow for two molecular systems.

    This class encapsulates the full protocol used to:

    1. Select and break one bond in each input topology.
    2. Remove the trailing fragments to create reactive radicals.
    3. Optimally place the second system relative to the first via
       a global optimization on a pseudo–energy surface that
       penalizes both bond stretching and steric overlaps.
    4. Merge the two systems, create the new reactive bond and
       reconstruct the resulting angle and dihedral topology.
    5. Refine atomic charges on the reactive atoms so that the
       final merged system remains charge neutral.

    The constructor runs the whole workflow for the currently
    supported method ``"breakBondsMerge"`` and updates ``obj1`` in
    place so that it contains the final merged topology and
    coordinates.

    Parameters
    ----------
    obj1, obj2
        ``Topology`` instances representing the two reacting systems.
        ``obj1`` is used as the reference / primary system and is
        modified in place.
    bb1, bb2
        Bond specifications for the bonds to be broken in ``obj1``
        and ``obj2``. Each can either be a pair of atom indices or a
        pair of atom *types*; in the latter case a random matching
        bond of that type is selected.
    react1, react2 : int, optional
        Orientation flags (0 or 1) that indicate which side of the
        selected bond is kept as the product. ``1`` corresponds to a
        reversed (swapped) bond direction.
    rcut : float, optional
        Cutoff radius used to define the neighbourhood around the
        reaction site during placement of ``obj2``.
    method : {"breakBondsMerge"}, optional
        High–level protocol to execute. Currently only
        ``"breakBondsMerge"`` is implemented.
    frame : int, optional
        Index of the coordinate frame used for the operation.
    seed1, seed2 : int or None, optional
        Random seeds used when sampling candidate bonds in ``obj1``
        and ``obj2`` respectively.
    shape : str, optional
        Geometric selection mode for reactive sites when sampling
        bonds (e.g. ``"flat"`` surface selection).
    morse_bond, morse_overlaps : tuple, optional
        Parameters for the Morse–like attractive term for the
        reaction bond and the repulsive term for overlaps used in the
        placement cost function.
    use_bounds : bool, optional
        Whether to iteratively refine the placement if the resulting
        ``z``–extent stays within the user–defined ``bounds_z``.
    bounds_z : (float, float), optional
        Lower and upper bounds on the mean ``z`` position of the
        placed object used to trigger a secondary refinement.
    maptypes : dict, optional
        Optional mapping from original atom types to new reactive
        atom types that should be used at the reaction centres.
    updown_method : str or None, optional
        Strategy used to alternate between placing reactive sites
        above / below the reference surface when ``shape == "flat"``.
    cite_method : str, optional
        Site–selection method name used when choosing a bond in
        ``obj1`` based on the provided bond type ``bb1``.
    cite_method_kwargs : dict, optional
        Extra keyword arguments passed to the site–selection method.
    iapp : str or None, optional
        Optional suffix used to distinguish newly created reactive
        atom types when updating force–field parameters.
    """

    def __init__(self,obj1,obj2,bb1,bb2,react1=0,react2=0,
                 rcut=3.5,method='breakBondsMerge',
                 frame=0,seed1=None,seed2=None,shape='flat',
                 morse_bond=(100,0.16,2),morse_overlaps=(0.2,5),use_bounds=True,bounds_z=(-1.5,1.5),
                 maptypes=dict(),updown_method=None,cite_method='random',cite_method_kwargs=dict(),
                 iapp=None):
        self.cite_method = cite_method
        for k,v in cite_method_kwargs.items():
            setattr(self,k,v)
        self.obj1 = obj1
        self.obj2 = obj2
        self.seed1 = seed1
        self.seed2 = seed2
        self.react1 = react1
        self.react2 = react2
        self.shape=shape
        self.updown_method = updown_method
        self.bb1 = bb1
        self.bb2 = bb2
        self.set_break_bond_id('1')
        self.set_break_bond_id('2')
        self.rcut = rcut
        self.frame = frame
        self.bond_to_create = (self.break_bondid1[react1],self.break_bondid2[react1])
        self.maptypes = maptypes
        self.morse_bond = morse_bond
        self.morse_overlaps = morse_overlaps
        self.bounds_z = bounds_z
        self.use_bounds = use_bounds
        if method == 'breakBondsMerge':
            # 1) Break the selected bonds and remove the trailing
            #    fragments to form reactive radicals on each system.
            self.break_bonds()

            # Map the original atom ids to the new ids that are
            # consistent with the potentially pruned topologies.
            self.react_id1 = self.obj1.old_to_new_ids[self.bond_to_create[0]]
            self.react_id2 = self.obj2.old_to_new_ids[self.bond_to_create[1]]



            # 2) Optimally place obj2 around the reactive site of
            #    obj1 using a global optimization on the pseudo
            #    energy surface defined in ``cost_overlaps``.
            self.place_obj2()

            natoms1 = self.obj1.natoms
            self.obj2.mol_names[:] = self.obj1.mol_names[0]
            if iapp is None:
                self.obj1.merge_system(self.obj2)
            else:
                self.obj1.merge_system(self.obj2,add=str(iapp))

            # 3) Create the new bond between the two reactive atoms
            #    and reconstruct the resulting angle / dihedral
            #    topology.
            self.reacted_id2 = natoms1 +self.react_id2 #renweing the id
            self.reacted_id1 = self.react_id1
            self.change_type(self.reacted_id1)
            self.change_type(self.reacted_id2)
            conn_id,c_type = self.obj1.sorted_id_and_type((self.reacted_id1,self.reacted_id2))
            self.obj1.connectivity[conn_id] = c_type

            new_angles = self.obj1.find_new_angdihs(conn_id)

            self.obj1.angles.update(new_angles)
            for newa in new_angles:
                new_dihs = self.obj1.find_new_angdihs(newa)
                self.obj1.dihedrals.update(new_dihs)
            # 4) Adjust charges on the reactive atoms and update
            #    force–field parameters to maintain charge neutrality.
            self.refine_charge()


        else:
            raise NotImplementedError('There is no such method as "{:s}"'.format(method))
        return

    def break_bonds(self):
        """Break the selected bonds and remove trailing fragments.

        The method determines the orientation of each bond based on
        ``react1`` and ``react2`` and then calls ``remove_trails`` to
        identify and remove the atoms that belong to the product side
        of the bond. The resulting information about the removed
        fragments (free radicals) is stored in ``radicals1`` and
        ``radicals2``.
        """
        # Remove the product side of each chosen bond and keep the
        # reactive atom that will later form the new bond.
        if self.react1 ==1:
            rbb1 = (self.break_bondid1[1],self.break_bondid1[0])
        else:
            rbb1 = self.break_bondid1
        if self.react2 ==1:

            rbb2 = (self.break_bondid2[1],self.break_bondid2[0])
        else:
            rbb2 = self.break_bondid2
        self.radicals1 = self.remove_trails(self.obj1,*rbb1)
        self.radicals2 = self.remove_trails(self.obj2,*rbb2)
        return
    def refine_charge(self):
        """Redistribute charge on the reactive atoms and update FF.

        The total charge of the merged system is computed and half of
        this value (``tch``) is subtracted from each of the two
        reactive atoms. The corresponding atom type parameters in the
        force field are updated to be consistent with the new atomic
        charges. A small tolerance is enforced on the final total
        charge; if violated an exception is raised.
        """
        id1 = self.reacted_id1
        id2 = self.reacted_id2
        tch = 0.5*self.obj1.total_charge
        ty1 = self.obj1.at_types[id1]
        ty2 = self.obj1.at_types[id2]
        for t,i in zip([ty1,ty2],[id1,id2]):
            newc = self.obj1.atom_charge[i] - tch
            self.obj1.atom_charge[i] = newc
            val = np.array(self.obj1.ff.atomtypes[t])
            val[2] = str(newc)
            self.obj1.ff.atomtypes[t] = tuple(val)
        totc = self.obj1.total_charge
        if abs(totc)>1e-10:
            raise Exception('Total charge is not newtral, total charge = {:.8e}'.format(totc))

        return
    def change_type(self,at_id,iapp=''):
        """Change the atom type of a reactive atom and sync FF terms.

        The atom type and atom code of the selected atom are changed
        either according to the user supplied ``maptypes`` or by
        appending ``'R'`` (and the optional ``iapp`` suffix) to the
        original type. All occurrences of the old atom type in local
        topology dictionaries (bonds, angles, dihedrals) and in the
        force–field parameter tables (bondtypes, angletypes,
        dihedraltypes, atomtypes) are updated accordingly so that
        the new reactive type reuses the same parameters.
        """
        #from copy import copy

        ty = self.obj1.at_types[at_id]
        if ty in self.maptypes:
            newty = self.maptypes[ty]
        else:
            newty = self.obj1.at_types[at_id] +'R' +iapp
        self.obj1.at_types[at_id] = newty
        self.obj1.atom_code[at_id] = newty

        for attr_name in ['connectivity','angles','dihedrals']:
            attr = getattr(self.obj1,attr_name)
            for c in list(attr.keys()):

                if at_id in c:
                    t = [i for i in attr[c]]
                    i = np.where(np.array(t)==ty)[0][0]
                    t[i] = newty
                    logger.debug('changed {} to {}'.format(attr[c],tuple(t)))
                    attr[c] = tuple(t)


        for attr_name in ['bondtypes','angletypes','dihedraltypes']:
            attr = getattr(self.obj1.ff,attr_name)
            for k in list(attr.keys()):
                if ty in k:
                    arr = [i for i in k]
                    i = np.where(np.array(arr)==ty)[0][0]
                    arr[i] = newty
                    ty_new  = tuple(arr)
                    logger.debug('made type {} same as  {}'.format(k,newty))
                    val = list(attr[k])

                    code = '  '.join(ty_new)
                    val[0] = code
                    attr[ty_new] = tuple(val)
                    #del attr[k]
            #setattr(self.obj1.ff,attr_name,attr)
        val = list(self.obj1.ff.atomtypes[ty])
        val[0] = newty
        self.obj1.ff.atomtypes[newty] = tuple(val)
        #self.obj1.filter_ff()
        return

    def set_break_bond_id(self,prefix):
        """Resolve bond specification for ``obj1`` or ``obj2``.

        Depending on ``prefix`` being ``'1'`` or ``'2'`` this method
        interprets ``bb1`` / ``bb2`` as either:

        * a pair of atom types (strings) – a random bond of that type
          is selected via :meth:`find_random_bond_of_type`, or
        * a pair of atom indices (ints) – used directly as the bond
          to break.

        The resolved bond id is stored on the instance as
        ``break_bondid1`` or ``break_bondid2`` respectively.
        """
        obj = getattr(self,'obj'+prefix)
        bb = getattr(self,'bb'+prefix)
        seed = getattr(self,'seed'+prefix)
        react = getattr(self,'react'+prefix)

        if not ( react == 0 or react == 1):
            raise ValueError('react'+prefix +' must be zero or one')

        name = 'break_bondid' + prefix

        if type(bb[0]) is str and type(bb[1]) is str:
            if prefix =='1':
                pm = self.cite_method
            else:
                pm = 'random'
            self.prefix_break_id =prefix
            bond_id  = self.find_random_bond_of_type(obj,bb,seed,react,pm)
            assert bb == obj.connectivity[bond_id],'the bond id {} does not give the specified type {}'.format(bond_id,bb)
            #if react == 1:
                #bond_id = (bond_id[1],bond_id[0])

        elif type(bb[0]) is int and type(bb[1]) is int:
            bond_id = bb
        else:
            raise NotImplementedError('value {} is not understood for {:s}'.format(bb,'bb'+prefix) )
        setattr(self,name,bond_id)
        return

    @staticmethod
    def numb_of_neibs(c,ctot):
        n = [np.sum(np.exp(-(1/0.265)*Distance_Functions.spherical(ctot,c[i]))) for i in range(c.shape[0])]
        return np.array(n)


    def find_random_bond_of_type(self,obj,bb,seed,idr,propmethod=None):
        """Sample a random bond matching the requested bond type.

        Parameters
        ----------
        obj : Topology
            Topology instance to search for candidate bonds.
        bb : tuple
            Pair of atom *types* that identifies the desired bond
            type (e.g. ("C", "H")).
        seed : int or None
            Random seed used to make the selection reproducible.
        idr : int
            Index (0 or 1) used when scoring candidate sites; it
            specifies which atom in the bond is treated as the
            reference.
        propmethod : str or None
            Name of the site–selection strategy ("random",
            "height", "neibs", "height_neibs", "separation_distance",
            "uniform", ...). Each method defines a different
            probability distribution over candidate sites.

        Returns
        -------
        tuple
            A pair of atom indices representing the chosen bond.
        """

        def height(zf):
            za = np.abs(zf)
            zrel = za - za.min()
            m = zrel.max()
            prop = 1-np.exp(-5*m*zrel)
            return prop
        def neibs(cf,c):
            prop = np.exp(-self.numb_of_neibs(cf,c))
            return prop
        def separation(nums):
            sids = self.obj1.at_ids [self.obj1.at_types ==self.separation_type]
            sep = False
            jmax = self.obj1.natoms
            j=0
            set_nums = set(nums)
            box = self.obj1.get_box(0)
            ncites =0
            while (sep==False):
                if len(set_nums) ==0:
                    self.separation_distance*=0.8
                    set_nums = set(nums)
                num = np.random.choice(list(set_nums))

                cre = c[ids[num]]
                c_comp = c[sids]

                dists = self.minimum_image_distance(c_comp,cre,box)

                if (dists< self.separation_distance).any():
                    #print('cite {:d} to close'.format(ids[num]))
                    ncites+=1
                    sep = False
                else:
                    print('# of Failed cites = {:d},  cite {:d} ok'.format(ncites,ids[num]))
                    sep = True
                set_nums.remove(num)
                j+=1
                if j>jmax:
                    raise Exception('infinite while loop')
            return num

        np.random.seed(seed)

        bbids = ass.numpy_keys(obj.connectivity)
        bbts = ass.numpy_values(obj.connectivity)

        f = bbts == np.array(bb)
        f = np.logical_and(f[:,0],f[:,1])
        bids = bbids[f]
        nums = np.arange(0,bids.shape[0],1,dtype=int)
        ids = bids[:,idr]
        c = obj.get_coords(0).copy()
        z = c[:,2]
        if self.shape=='flat' and self.prefix_break_id=='1':
            z = z[ids]
            zm = np.sum(z)/z.shape[0]
            z -= zm
            if self.updown_method =='random':

                updown = np.random.choice([True,False])
            else:
                if not hasattr(self.obj1,'updown'):
                    self.obj1.updown = False
                self.obj1.updown = not self.obj1.updown
                updown = self.obj1.updown
            if updown:
                fz = z>0
            else:
                fz = z<0

            nums = nums[fz]
            zf = z[fz]
        else:
            num = np.random.choice(nums)
            return tuple(bids[num])
        print(propmethod)

        if propmethod=='random':
            num =  np.random.choice(nums)
        elif propmethod =='height':
            prop = height(zf)
        elif propmethod=='neibs':
            prop = neibs(c[ids][fz],z)
        elif propmethod =='height_neibs':
            prop = neibs(c[ids][fz],z)*height(zf)
        elif propmethod =='separation_distance':
            if f.any():
                num = separation(nums)
            else:
                num = np.random.choice(nums)
            return tuple(bids[num])
        elif propmethod =='uniform':
            self.initial_separation_distance = self.separation_distance
            grid_x,grid_y = self.grid
            box = self.obj1.get_box(0)
            Lx = box[0]
            Ly = box[1]
            dLx = Lx/grid_x
            dLy = Ly/grid_y
            areas = [(i,j,u) for i in range(grid_x) for j in range(grid_y) for u in [0,1]]
            if not hasattr(self.obj1,'filled_areas'):
                self.obj1.filled_areas = {k:0 for k in areas}
            if updown:
                kselect=(0,0,1)
                areas = {k:v for k,v in self.obj1.filled_areas.items() if k[2] == 1}
            else:
                kselect=(0,0,0)
                areas = {k:v for k,v in self.obj1.filled_areas.items() if k[2] == 0}
            for k,v in areas.items():
                if v < areas[kselect]:
                    kselect = k
            same = []
            for k,v in areas.items():
                if v == areas[kselect]:
                    same.append(k)
            kselect = same[ np.random.choice(np.arange(0,len(same),1,dtype=int)) ]
            #print(kselect)
            cids = c[ids,:][fz]
            cx = cids[:,0]
            cy = cids[:,1]
            kx = kselect[0]
            ky = kselect[1]
            fx = np.logical_and(kx*dLx < cx, cx <= (kx+1)*dLx)
            fy = np.logical_and(ky*dLy < cy, cy <= (ky+1)*dLy)
            f = np.logical_and(fx,fy)

            self.obj1.filled_areas[kselect]+=1
            #print(self.obj1.filled_areas)
            if f.any():
                num = separation(nums[f])
            else:
                num = np.random.choice(nums[f])
            #print(self.obj1.at_types[bids[num]])
            self.separation_distance = self.initial_separation_distance
            return tuple(bids[num])

        else:
           raise ValueError('There is no method name  as {:s}'.format(propmethod))

        prop/=prop.sum()
        num = np.random.choice(nums,p=prop)
        bond_id = tuple(bids[num])
        return bond_id
    @staticmethod
    @jit(nopython=True,fastmath=True,parallel=True)
    def minimum_image_distance(coords,cref,box):
            """Compute minimum–image distances to a reference point.

            The calculation assumes periodic boundary conditions and
            folds all coordinates back into the central simulation box
            relative to ``cref`` before computing Euclidean distances.

            Parameters
            ----------
            coords : array, shape (N, 3)
                Atom coordinates.
            cref : array, shape (3,)
                Reference coordinate.
            box : array-like, shape (3,)
                Box lengths along ``x``, ``y`` and ``z``.

            Returns
            -------
            ndarray, shape (N,)
                Minimum–image distances of each coordinate to
                ``cref``.
            """
            r = coords - cref

            for j in range(3):
                b = box[j]
                b2 = b/2
                fm = r[:,j] < - b2
                fp = r[:,j] >   b2
                r[:,j][fm] += b
                r[:,j][fp] -= b
            d = np.zeros((r.shape[0],),dtype=np.float64)
            for i in prange(r.shape[0]):
                for j in range(3):
                    x = r[i,j]
                    d[i] += x*x
                d[i] = np.sqrt(d[i])

            return d
    @staticmethod
    @jit(nopython=True,fastmath=True,parallel=True)
    def minimum_image_distance_coords(coords,cref,box):
            """Return minimum–image distances and wrapped coordinates.

            Similar to :meth:`minimum_image_distance` but also returns
            the coordinates after applying the minimum–image
            convention, which is useful for extracting local
            neighbourhoods around ``cref``.
            """
            r = coords - cref
            imag_coords = coords.copy()
            for j in range(3):
                b = box[j]
                b2 = b/2
                fm = r[:,j] < - b2
                fp = r[:,j] >   b2

                r[:,j][fm] += b
                imag_coords[:,j][fm] +=b

                r[:,j][fp] -= b
                imag_coords[:,j][fp] -= b
            d = np.zeros((r.shape[0],),dtype=np.float64)
            for i in prange(r.shape[0]):
                for j in range(3):
                    x = r[i,j]
                    d[i] += x*x
                d[i] = np.sqrt(d[i])

            return d,imag_coords

    @staticmethod
    def find_reaction_neibhour_coords(coords,cref,box,rcut):
        """Select atoms in the neighbourhood of the reaction site.

        Parameters
        ----------
        coords : array, shape (N, 3)
            Coordinates of all atoms in the reference system.
        cref : array, shape (3,)
            Coordinate of the reactive atom in ``obj1``.
        box : array-like, shape (3,)
            Simulation box lengths.
        rcut : float
            Cutoff radius that defines the local reaction
            neighbourhood.

        Returns
        -------
        ndarray, shape (M, 3)
            Coordinates of the atoms within ``rcut`` of the reaction
            site using minimum–image distances.
        """

        d,imag_coords =  React_two_systems.minimum_image_distance_coords(coords,cref,box)
        reaction_neibs = imag_coords[d<rcut]
        return reaction_neibs

    @staticmethod
    def identify_trail(obj,trail_from,trail_to):
        """Identify atoms that belong to the product side of a bond.

        Starting from ``trail_to`` the connectivity graph is followed
        recursively (excluding ``trail_from``) until no new atoms are
        discovered. The resulting set corresponds to the trailing
        fragment that will be removed when breaking the bond.
        """
        trailing_set_old = set()
        trailing_set = {trail_to}
        while len(trailing_set) != len(trailing_set_old):
            trailing_set_old = trailing_set.copy()
            for j in trailing_set_old:
                for neib in obj.neibs[j]:
                    if neib !=trail_from:
                        trailing_set.add(neib)

        return trailing_set

    @staticmethod
    def remove_trails(obj,trail_from,trail_to):
        """Remove trailing fragment and return a summary dataframe.

        The atoms that belong to the product side of the bond are
        identified via :meth:`identify_trail`, their coordinates are
        collected and the atoms are removed from the topology. A
        ``pandas.DataFrame`` with the removed atom ids, types,
        charges, masses and coordinates is returned for bookkeeping.
        """
        trailing_ids = set()
        trailing_ids = React_two_systems.identify_trail(obj,trail_from,trail_to)
        trailing_ids = np.array(list(trailing_ids))

        cf = obj.get_coords(0)[trailing_ids]

        free_radicals = pd.DataFrame({'at_ids':trailing_ids,
                                      'at_tys':obj.at_types[trailing_ids],
                                      'mol_ids':obj.mol_ids[trailing_ids],
                                      'mol_names':obj.mol_names[trailing_ids],
                                      'atom_charge':obj.atom_charge[trailing_ids],
                                      'atom_mass':obj.atom_mass[trailing_ids],
                                      'x':cf[:,0],
                                      'y':cf[:,1],
                                      'z':cf[:,2],
                                      })
        obj.remove_atoms_ids(trailing_ids)

        return free_radicals

    @staticmethod
    @jit(nopython=True,fastmath=True)
    def morse(r,De,re,alpha):
        """Morse potential used for the attractive reaction bond."""
        return De*(np.exp(-2*alpha*(r-re))-2*np.exp(-alpha*(r-re)))

    @staticmethod
    @jit(nopython=True,fastmath=True)
    def morse_rep(r,re,alpha):
        """Purely repulsive Morse–like term for steric overlaps."""
        return np.exp(-alpha*(r-re))
    @staticmethod
    def cost_overlaps(vector_n_angles,creact1,id2,coords_neib_obj1,coords_obj2,
                      morse_bond,morse_overlaps):
        """Pseudo–energy used to place ``obj2`` around ``obj1``.

        The cost consists of an attractive Morse term evaluated at the
        distance between the reactive atoms (the putative new bond)
        and a repulsive contribution that penalizes close contacts
        between ``obj2`` and the neighbourhood of the reaction site in
        ``obj1``. The optimizer searches over three translations and
        three Euler angles provided as ``vector_n_angles``.
        """
        ctr = RotTrans.trans_n_rot(vector_n_angles, coords_obj2)
        # distance of reaction bond
        refdist = RotTrans.distance(creact1,ctr[id2])

        f1 = React_two_systems.morse
        f2 = React_two_systems.morse_rep
        d = [Distance_Functions.spherical(coords_neib_obj1, c)
             for i,c in enumerate(ctr) if i!=id2]
        d =np.array(d)# shape ctr.shape[0]
        return  f1(refdist,*morse_bond) + np.sum(f2(d,*morse_overlaps))

    def place_obj2(self):
        """Optimize the position and orientation of ``obj2``.

        The second system is first translated to the centre of the
        simulation box and then a global optimization (currently
        ``scipy.optimize.differential_evolution``) is performed on the
        six–dimensional space of translations and rotations. The
        objective is the pseudo–energy defined in
        :meth:`cost_overlaps`, which balances forming a reasonable
        reaction bond distance with avoiding steric clashes.
        """
        #t0 = perf_counter()
        frame = self.frame
        coords_obj2 = self.obj2.get_coords(frame)
        coords_obj1 = self.obj1.get_coords(frame)

        creact1 = coords_obj1[self.react_id1]


        box = self.obj1.get_box(frame)
        bm = box/2
        cm2 = np.mean(coords_obj2,axis=0)
        coords_obj2 += bm-cm2

        coords_neibs_obj1 = self.find_reaction_neibhour_coords(coords_obj1,creact1,box,self.rcut)

        bounds= [(-m/2,m/2) for m in box] + [(-np.pi,np.pi)]*3

        arguments = (creact1,self.react_id2,
                     coords_neibs_obj1,coords_obj2,
                     self.morse_bond,self.morse_overlaps)

        if False:
            opt_res = dual_annealing(self.cost_overlaps, bounds,
                    args =  arguments,
                    maxiter = 1000,
                    restart_temp_ratio=1e-5,
                    minimizer_kwargs={'method':'SLSQP',#
                                     'bounds':bounds,

                    'options':{'maxiter':300,
                        'disp':False,
                        'ftol':1e-3},
                                },
                              )

        else:
            opt_res = differential_evolution(self.cost_overlaps, bounds,
                    args = arguments ,
                    maxiter = 60,disp=False,polish=True)
        print('Pseudo Energy = {:4.6f}'.format(opt_res.fun))

        if not hasattr(self.obj1,'se'):
            self.obj1.se =[]
        self.obj1.se.append(opt_res.fun)

        self.opt_res = opt_res
        self.p = opt_res.x

        new_coords = RotTrans.trans_n_rot(self.p,coords_obj2)

        miz = new_coords[:,2].min()
        mxz = new_coords[:,2].max()
        bz = self.bounds_z
        between_bounds = bz[0] <= 0.5*(miz+mxz) <= bz[1]
        if self.use_bounds and  between_bounds:
            # If the object lies comfortably within the requested
            # bounds, tighten / soften the overlap potential and
            # perform a second refinement to improve placement.
            self.morse_overlaps = (self.morse_overlaps[0]*2,self.morse_overlaps[1]/2)
            print('Min z surf = {:4.3f} Max z surf = {:4.3f}\n Min placement {:4.3f} Max placement {:4.3f}'.format(*bz,miz,mxz))
            print('refining placement')
            self.place_obj2()
            return
        else:
            self.obj2.timeframes[frame]['coords'] = new_coords
        #tf = perf_counter()-t0
        #print('time of placing object = {:.3e} sec'.format(tf))
        return

class RotTrans:
    """Rotation and translation utilities for 3D coordinates.

    The static methods in this class provide simple building blocks for
    composing rigid-body transformations of coordinate arrays. They are
    used by :class:`React_two_systems` when optimising the placement of
    the second reacting object relative to the first.
    """

    def __init__(self):
        # This class is used purely as a namespace for static methods.
        return

    @staticmethod
    def Rx(theta):
        """Return rotation matrix for a rotation around the *x* axis."""
        c = np.cos(theta)
        s = np.sin(theta)
        r = np.zeros((3,3))
        r[0,0] = 1
        r[1,1] = c
        r[2,2] = c
        r[1,2] = -s
        r[2,1] = s
        return r
    @staticmethod
    def Ry(theta):
        """Return rotation matrix for a rotation around the *y* axis."""
        c = np.cos(theta)
        s = np.sin(theta)
        r = np.zeros((3,3))
        r[1,1] = 1
        r[0,0] = c
        r[2,2] = c
        r[0,2] = -s
        r[2,0] = s
        return r
    @staticmethod
    def Rz(theta):
        """Return rotation matrix for a rotation around the *z* axis."""
        c = np.cos(theta)
        s = np.sin(theta)
        r = np.zeros((3,3))
        r[2,2] = 1
        r[0,0] = c
        r[1,1] = c
        r[0,1] = -s
        r[1,0] = s
        return r

    @staticmethod
    def rotate(c,yaw,pitch,roll):
        """Apply successive rotations to coordinates.

        Parameters
        ----------
        c : ndarray, shape (N, 3)
            Input coordinates relative to the origin.
        yaw, pitch, roll : float
            Rotation angles (in radians) around the *x*, *y* and *z*
            axes respectively.

        Returns
        -------
        ndarray, shape (N, 3)
            Rotated coordinates.
        """
        Rx = RotTrans.Rx(yaw)
        Ry = RotTrans.Ry(pitch)
        Rz = RotTrans.Rz(roll)
        cn = c.copy()
        for r in [Rx,Ry,Rz]:
            for i in range(cn.shape[0]):
                cn[i] = np.dot(r,cn[i])
        return cn

    @staticmethod
    def distance(r1,r2):
        """Return Euclidean distance between two 3D points."""
        r = r2-r1
        return np.sum(r*r)**0.5

    @staticmethod
    def rhat(r1,r2):
        """Return unit vector from ``r1`` to ``r2``."""
        d = RotTrans.distance(r1,r2)
        return (r2-r1)/d

    @staticmethod
    def translate(r1,rx):
        """Translate coordinate ``r1`` by displacement vector ``rx``."""
        return r1+rx
    @staticmethod
    def trans_n_rot(vector_n_angles,coords):
        """Translate and then rotate a set of coordinates.

        Parameters
        ----------
        vector_n_angles : array-like, shape (6,)
            First three elements are translation components, last three
            are rotation angles (yaw, pitch, roll) in radians.
        coords : ndarray, shape (N, 3)
            Input coordinates.

        Returns
        -------
        ndarray, shape (N, 3)
            Transformed coordinates after translation and rotation
            around their centre of mass.
        """
        vector = vector_n_angles[:3]
        angles = vector_n_angles[3:]
        translated_coords = coords + vector
        rotref = np.mean(translated_coords,axis=0)
        relc = translated_coords - rotref
        cr = RotTrans.rotate(relc, *angles)
        return cr+rotref



class Topology:
    """Minimal container for atomistic topology and force-field data.

    This class stores per-atom arrays (types, charges, masses,
    molecule ids/names) together with coordinate timeframes and
    force-field parameter tables. It acts as the common in-memory
    representation for molecular systems manipulated by the analysis
    and reaction workflow classes.

    Parameters
    ----------
    natoms : int
        Number of atoms in the system.
    at_types : array-like or str, optional
        Atom type identifiers for each atom. If a scalar is given it is
        broadcast to all atoms.
    atom_code : array-like or str, optional
        Alternate per-atom label, typically mirroring ``at_types``.
    mol_ids : array-like or int, optional
        Molecule id for each atom; broadcast if scalar.
    mol_names : array-like or str, optional
        Molecule name for each atom; broadcast if scalar.
    atom_charge : array-like or float, optional
        Partial atomic charges; broadcast if scalar.
    atom_mass : array-like or float, optional
        Atomic masses; broadcast if scalar.
    timeframes : dict, optional
        Mapping from frame index to dictionaries containing e.g.
        coordinates, box vectors, time, step.
    neibs, connectivity, angles, dihedrals, pairs, exclusions : dict, optional
        Topology dictionaries describing bonded and nonbonded
        connectivity.
    atomtypes, bondtypes, angletypes, dihedraltypes : dict, optional
        Force-field parameter tables stored under ``self.ff``.
    """

    def __init__(self,natoms,at_types='ATOP',atom_code='ATOP',mol_ids=1,mol_names='MTOP',
                 atom_charge=0.0,atom_mass=10.0,timeframes= dict(),neibs=dict(),
                 connectivity=dict(),angles=dict(),dihedrals=dict(),pairs=dict(),exclusions=dict(),
                 atomtypes=dict(),bondtypes=dict(),angletypes=dict(),dihedraltypes=dict(),
                 ):
        self.at_ids = np.arange(0,natoms,1,dtype='i')
        self.ff = self.FFparams()
        defaults_str = ['at_types','mol_names','atom_code']
        defaults_int = ['mol_ids']
        defaults_float = ['atom_charge','atom_mass']
        update_defaults = ['timeframes','neibs','connectivity','angles','dihedrals','pairs','exclusions']
        update_ff_defaults = ['atomtypes','bondtypes','angletypes','dihedraltypes']
        self.exclusions_map = dict()
        for a in defaults_str:
            setattr(self,a,np.empty(natoms,dtype=object))
        for a in defaults_int:
            setattr(self,a,np.empty(natoms,dtype=int))
        for a in defaults_float:
            setattr(self,a,np.empty(natoms,dtype=float))

        if mol_ids is not None:
            self.mol_ids[:] = mol_ids
        if atom_code is not None:
            self.atom_code[:] = atom_code
        if mol_names is not None:
            self.mol_names[:] = mol_names
        if at_types is not None:
            self.at_types[:] = at_types

        if atom_charge is not None:
            self.atom_charge[:] = atom_charge
        if atom_mass is not None:
            self.atom_mass[:] = atom_mass


        for a in update_defaults:
            d = locals()[a]
            setattr(self,a,d)
        for a in update_ff_defaults:
            d = locals()[a]
            setattr(self.ff,a,d)
        return

    class FFparams():
        """Lightweight container for force-field parameter tables."""

        def __init__(self):
            self.atomtypes = dict()
            self.bondtypes = dict()
            self.angletypes = dict()
            self.dihedraltypes = dict()
            self.nonbond_params = dict()
            return
        def add_posres(self,data):
            """Attach positional restraint definitions to the force field.

            Parameters
            ----------
            data : dict or list of dict
                Positional restraint entries, each describing the atoms
                and restraint parameters to be applied.
            """
            if type(data) is list:
                if type(data[0]) is not dict:
                    raise Exception('give prober informations see function documentation')
            if type(data) is dict:
                data = [data]
            self.posres = data
            return
        @property
        def nbondtypes(self):
            """Number of bond types"""
            return len(self.bondtypes)

        @property
        def nangletypes(self):
            """Number of angle types"""
            return len(self.angletypes)

        @property
        def natomtypes(self):
            """Number of atom types"""
            return len(self.atomtypes)

        @property
        def ndihedraltypes(self):
            """Number of dihedral types"""
            return len(self.dihedraltypes)

    @property
    def total_charge(self):
        """Total charge of the system (sum of all atomic charges)."""
        return self.atom_charge.sum()
    @property
    def total_mass(self):
        """Total mass of the system (sum of all atomic masses)."""
        return self.atom_mass.sum()
    @property
    def inspect_system(self):
        """Return a :class:`pandas.DataFrame` summarising atom properties."""
        names = ['at_ids','at_types','mol_ids',
                 'mol_names','atom_charge','atom_mass','atom_code']

        return pd.DataFrame({ a:getattr(self,a) for a in names} )


    @property
    def nframes(self):
        """Number of coordinate frames stored in ``timeframes``."""
        return len(self.timeframes)

    def get_key(self):
        """Return a frame key using the current ``key_method`` strategy."""
        key_func = getattr(self,self.key_method)
        key = key_func()
        return key

    def get_timekey(self):
        """Return time difference between current frame and first frame.

        The returned value is rounded according to ``self.round_dec``
        and is typically used as the x-axis key for time-dependent
        properties.
        """
        t0 = self.get_time(self.first_frame)
        tf = self.get_time(self.current_frame)
        return round(tf-t0,self.round_dec)

    def get_exkey(self):
        """Return relative box extension along *x* between first and current frame."""
        e0 = self.get_box(self.first_frame)[0]
        et = self.get_box(self.current_frame)[0]
        return round((et-e0)/e0,self.round_dec)

    def dict_to_sorted_numpy(self,attr_name):
        """Precompute a sorted view of dictionary keys as a 2D array.

        For a topology dictionary whose keys are tuples of atom indices
        (e.g. bonds, angles, dihedrals), this creates an ``(N, 2)``
        array with the first and last index of each key sorted by the
        first index. The array is stored as ``sorted_<attr_name>_keys``
        on the instance and is useful for efficient range searches.
        """
        #t0 = perf_counter()

        attr = getattr(self,attr_name)
        if type(attr) is not dict:
            raise TypeError('This function is for working with dictionaries')

        keys = attr.keys()
        x = np.empty((len(keys),2),dtype=int)
        for i,k in enumerate(keys):
            x[i,0]=k[0] ; x[i,1]=k[-1]

        x = x[x[:,0].argsort()]
        setattr(self,'sorted_'+attr_name+'_keys',x)

        #tf = perf_counter() - t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return

    def keysTotype(self,attr_name):
        """Group topology keys by their type (dictionary value).

        Parameters
        ----------
        attr_name : str
            Name of the topology dictionary attribute (e.g.
            ``"connectivity"``, ``"angles"``).

        Returns
        -------
        dict
            Mapping from unique type values to an array of index
            tuples that carry that type.
        """
        dictionary = getattr(self,attr_name)
        if type(dictionary) is not dict:
            raise TypeError('This function is for working with dictionaries')
        types = self.unique_values(dictionary.values())
        temp_ids = {i:[] for i in types}
        for k,v in dictionary.items():
            temp_ids[v].append(np.array(k))
        ids = {v : np.array(temp_ids[v]) for v in types}

        return ids
    @property
    def connectivity_pertype(self):
        """Connectivity keys grouped by bond type."""
        return self.keysTotype('connectivity')
    @property
    def angles_pertype(self):
        """Angle keys grouped by angle type."""
        return self.keysTotype('angles')
    @property
    def dihedrals_pertype(self):
        """Dihedral keys grouped by dihedral type."""
        return self.keysTotype('dihedrals')

    def define_connectivity(self,bond_dists):
        """Infer bonded connectivity from distances and type pairs.

        Parameters
        ----------
        bond_dists : dict
            Mapping from sorted type pairs ``(type_i, type_j)`` to a
            tuple ``(r_min, r_max)`` describing the allowable bond
            distance range for that type pair.

        Notes
        -----
        The method reads coordinates for frame 0, loops over all atom
        pairs, and, whenever the distance lies below the upper cutoff
        ``r_max``, adds a bond to ``self.connectivity`` and updates the
        associated type mapping.
        """
        bond_dists = {tuple(np.sort(k)):v for k,v in bond_dists.items()}
        self.read_file(self.topol_file)
        c = self.get_coords(0)
        at_ids = self.at_ids
        at_types = self.at_types
        connectivity = dict()
        con_ty = dict()
        for i1,at1 in zip(at_ids,at_types):
            for i2,at2 in zip(at_ids,at_types):
                if i1==i2:
                    continue
                tyc = tuple(np.sort([at1,at2]))
                if tyc in bond_dists:
                    c1 = c[i1]
                    c2 = c[i2]
                    rd = c2 -c1
                    d = np.dot(rd,rd)**0.5
                    r = bond_dists[tyc]
                    if d<r[1]:
                        ids = (i1,i2)
                        conn_id,c_type = self.sorted_id_and_type(ids)
                        connectivity[conn_id] = c_type
                        if d>r[0]:
                            con_ty[conn_id] = 1
                        else:
                            con_ty[conn_id] = 2

        self.con_ty = con_ty
        self.connectivity = connectivity
        return

    def map_the_topology(self,mapper):
        self.mass_map = mapper['mass']
        self.charge_map = mapper['charge']
        self.define_connectivity(mapper['bonds'])
        return

    def read_topfile(self,file):
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
        lookfor = ['atomtypes','bondtypes','angletypes','dihedraltypes', 'nonbond_params'
                   'atoms','bonds','angles','pairs','exclusions']
        nline = dict()
        un_at_types = np.unique(self.at_types)
        nt = un_at_types.shape[0]
        for i,line in enumerate(lines):
            l = line.strip().split('[')[-1].split(']')[0].strip()
            if l in lookfor:
                nline[l] = i

        #finding masses and charges

        mass_map = dict()
        charge_map = dict()
        n = nline['atomtypes']+1
        jt = 0
        for line in lines[n:]:
            if ';' in line:
                continue
            l = line.strip().split()
            k = l[0]
            mass_map[k] = float(l[1])
            charge_map[k] = float(l[2])
            jt+=1
            if jt==nt:
                break

        self.read_itp_file(file,find_mass=mass_map,find_charge=charge_map)
        self.local_to_global_topology()


        self.atom_charge = self.find_with_typemap(charge_map)
        self.atom_mass = self.find_with_typemap(mass_map)
        #connectiviy

        self.ff = self.FFparams()
        for name,jd in zip(['atomtypes','bondtypes','angletypes','dihedraltypes', 'nonbond_params'],[1,2,3,4,2]):
            try:
                n = nline[name]+1
            except KeyError:
                setattr(self.ff,name,dict())
            else:
                attr = dict()
                for line in lines[n:]:
                    if '[' in line or ']' in line:
                        break

                    if ';' in line or 'include' in line:
                        continue
                    l = line.strip().split()
                    if len(l)==0:
                        continue
                    if jd>1:
                        ty = self.sorted_type(l[:jd])
                        value = (' '.join(ty),*l[jd:])
                    else:
                        if len(l) ==7:
                            ty = l[1]
                            value = (l[0],*l[jd:])
                        else:
                            ty=l[0]
                            value = (ty,*l[jd:])
                    if name =='dihedraltypes':
                        if ty not in attr:
                            attr[ty] = []
                        attr[ty].append(value)
                    else:
                        attr[ty] = value
                setattr(self.ff,name,attr)


        return
    def read_topfile_ff(self,file):

        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
        lookfor = ['atomtypes','bondtypes','angletypes','dihedraltypes','nonbond_params']
        nline = dict()

        for i,line in enumerate(lines):
            l = line.strip().split('[')[-1].split(']')[0].strip()
            if l in lookfor:
                nline[l] = i

        self.ff = self.FFparams()
        for name,jd in zip(lookfor,[1,2,3,4,2]):
            try:
                n = nline[name]+1
            except KeyError:
                setattr(self.ff,name,dict())
            else:
                attr = dict()
                for line in lines[n:]:
                    if '[' in line or ']' in line:
                        break
                    if ';' in line or 'include' in line or '#' in line:
                        continue
                    l = line.strip().split()
                    if len(l)==0:
                        continue
                    if jd>1:
                        ty = self.sorted_type(l[:jd])
                        value = (' '.join(ty),*l[jd:])
                    else:
                        if len(l) ==7:
                            ty = l[1]
                            value = (l[0],*l[jd:])
                        else:
                            ty=l[0]
                            value = (ty,*l[jd:])
                    if name =='dihedraltypes':
                        if ty not in attr:
                            attr[ty] = []
                        attr[ty].append(value)
                    else:
                        attr[ty] = value
                setattr(self.ff,name,attr)
        return


    def find_with_typemap(self,usedmap):
        first_element_type =  type(list(usedmap.values())[0])
        arr = np.empty(self.natoms,dtype = first_element_type)
        for i,ty in enumerate(self.at_types):
            arr[i] = usedmap[ty]
        return arr
    def read_total_conangdih(self,lines,c,sub=1):
        cad = dict()
        if c not in ['connectivity','angles','dihedrals']:
            raise ValueError('c should be one of these {"connectivity","angles","dihedrals"}')
        if c =='connectivity':
            ld = 2
        elif c =='angles':
            ld = 3
        elif c=='dihedrals':
            ld = 4
        for line in lines:
            if ';' in line:
                continue
            l = line.strip().split()[:ld]
            if len(l)==0 or '[' in line or ']' in line:
                break
            ids = tuple(np.array(l,dtype=int)-sub)
            conn_id,c_type = self.sorted_id_and_type(ids)
            cad[conn_id] = c_type
        setattr(self,c,cad)
        return

    def read_gromacs_topology(self):
        #t0 = perf_counter()

        if ass.iterable(self.connectivity_file):
            for cf in self.connectivity_file:
                if '.itp' in cf:
                    self.read_itp_file(cf)
                else:
                    raise NotImplementedError('Non itp files are not implemented for lists. You can give a top file')
            self.local_to_global_topology()
            self.make_ff_from_itp(cf)
        elif '.top' == self.connectivity_file[-4:]:
            self.read_topfile(self.connectivity_file)
        else:
            if '.itp' == self.connectivity_file[-4:]:
                cf = self.connectivity_file
                self.read_itp_file(cf)
                self.local_to_global_topology()
                self.make_ff_from_itp(cf)
            else:
                raise NotImplementedError('Non itp files are not yet implemented')


        return

    def local_to_global_topology(self,):
        self.connectivity = dict()
        self.angles = dict()
        self.dihedrals = dict()
        self.pairs = dict()
        self.exclusions = dict()
        self.refine_angles = False
        self.refine_dihedrals = False
        if not hasattr(self,'atom_charge'):
            self.atom_charge = np.empty(self.natoms,dtype=float)
        if not hasattr(self,'atom_mass'):
            self.atom_mass = np.empty(self.natoms,dtype=float)
        if not hasattr(self,'atom_code'):
            self.atom_code = np.empty(self.natoms,dtype=object)

        for j in np.unique(self.mol_ids):
            #global_mol_at_ids = self.at_ids[self.mol_ids==j]
            res_nm = np.unique(self.mol_names[self.mol_ids==j])

            assert res_nm.shape ==(1,),'many names for a residue. Check code or topology file'

            res_nm = res_nm[0]
            mol = self.molecule_map[res_nm]

            for i,idm in enumerate(mol['at_ids']):
                id0 = self.loc_id_to_glob[j][i]
                self.atom_charge[id0] = mol['charge'][i]
                self.atom_mass[id0] = mol['mass'][i]
                self.atom_code[id0] = mol['code'][i]
            local_connectivity = self.connectivity_per_resname[res_nm]
            local_angles = self.angles_per_resname[res_nm]
            local_dihedrals = self.dihedrals_per_resname[res_nm]
            local_pairs =  self.pairs_per_resname[res_nm]
            local_exclusions = self.exclusions_per_resname[res_nm]
            for b in local_connectivity:
                id0 = self.loc_id_to_glob[j][b[0]]
                id1 = self.loc_id_to_glob[j][b[1]]
                conn_id,c_type = self.sorted_id_and_type((id0,id1))
                self.connectivity[conn_id] = c_type
            for a in  local_angles:
                id0 = self.loc_id_to_glob[j][a[0]]
                id1 = self.loc_id_to_glob[j][a[1]]
                id2 = self.loc_id_to_glob[j][a[2]]
                a_id,a_t = self.sorted_id_and_type((id0,id1,id2))
                self.angles[a_id] = a_t
            for d in local_dihedrals:
                id0 = self.loc_id_to_glob[j][d[0]]
                id1 = self.loc_id_to_glob[j][d[1]]
                id2 = self.loc_id_to_glob[j][d[2]]
                id3 = self.loc_id_to_glob[j][d[3]]
                d_id,d_t = self.sorted_id_and_type((id0,id1,id2,id3))
                self.dihedrals[d_id] = d_t
            for p in local_pairs:
                id0 = self.loc_id_to_glob[j][p[0]]
                id1 = self.loc_id_to_glob[j][p[1]]
                conn_id,c_type = self.sorted_id_and_type((id0,id1))
                self.pairs[conn_id] = c_type
            for e in local_exclusions:
                id0 = self.loc_id_to_glob[j][e[0]]
                id1 = self.loc_id_to_glob[j][e[1]]
                conn_id,c_type = self.sorted_id_and_type((id0,id1))
                self.exclusions[conn_id] = c_type
        return


    def find_neibs(self):
        '''
        Computes first (bonded) neihbours of a system in dictionary format
        key: atom_id
        value: set of neihbours
        '''
        neibs = dict()
        for k in self.connectivity.keys():
            for i in k: neibs[i] = set() # initializing set of neibs
        for j in self.connectivity.keys():
            neibs[j[0]].add(j[1])
            neibs[j[1]].add(j[0])
        for m in self.at_ids:
            if m not in neibs:
                neibs[m] = set()
        self.neibs = neibs
        return

    def correct_types_from_itp(self):

        at_map = dict()
        done_mole_names = []
        for umi in np.unique(self.mol_ids):
            ids  = np.where(self.mol_ids== umi)[0]
            mol_name = self.mol_names[ids][0]
            if not (mol_name == self.mol_names[ids]).all():
                raise Exception(f'For mol_id {umi} mol_names are different!')

            if mol_name not in done_mole_names:
                done_mole_names.append(mol_name)
                at_map.update( { ty1:ty2 for ty1,ty2 in
                            zip( self.at_types[ids], self.molecule_map[mol_name]['at_types'] ) } )

            self.at_types[ids] = [at_map[ty] for ty in self.at_types[ids]]
        tt = self.at_types
        self.connectivity = { k: tuple(tt[i] for i in k) for k in self.connectivity }
        self.angles = { k: tuple(tt[i] for i in k) for k in self.angles }
        self.dihedrals = { k: tuple(tt[i] for i in k) for k in self.dihedrals }
        self.pairs = { k: tuple(tt[i] for i in k) for k in self.pairs }
        self.exclusions = { k: tuple(tt[i] for i in k) for k in self.exclusions }

        return
    def identify_element(self,ty):
        if ty[1:2].islower():
            return ty[:2]
        else:
            return ty[:1]
    @property
    def attribute_names(self):
        return list(self.__dict__.keys())

    def read_itp_file(self,file,find_mass=dict(),find_charge=dict()):
        t0 = perf_counter()

        with open(file,'r') as f:
            lines = f.readlines()
            f.closed

        # Reading atoms
        jlines = {'atoms':set(),'bonds':set(),'angles':set(),
                  'dihedrals':set(),'moleculetype':set(),'pairs':set(),
                  'exclusions':set(),'atomtypes':set()}
        for j,line in enumerate(lines):
            for k in jlines:
                if k in line and '[' in line and ']' in line:
                    jlines[k].add(j)

        ma = jlines['atomtypes']
        if len(ma) ==1:
            charge_defaults = dict()
            for line in lines[list(ma)[0]+1:]:

                if ';' in line:continue
                if len(line.split()) ==0: continue
                if '[' in line or ']' in line:
                    break

                l = line.strip().split()
                charge_defaults[l[-6]] = float(l[-4])

        mt = jlines['moleculetype']

        exclusions_map = dict()
        if len(mt) ==1:
            for line in lines[list(mt)[0]+1:]:
                if ';' in line:continue
                if len(line.split()) ==0: continue
                if '[' in line or ']' in line:
                    break

                l = line.strip().split()
                #print(l[0],l[1])
                exclusions_map[l[0]] = int(l[1])
        at_ids = [] ; res_num=[];at_types = [] ;res_name = [] ; cngr =[]
        code = [] ; charge = [] ; mass = []

        atomscode_map = dict()
        jli = list(jlines['atoms'])[0]
        i=0

        for line in lines[jli+1:]:
            l = line.split()
            try:
                int(l[0])
            except:
                pass
            else:
                nl  = len(l)
                #print(l)
                at_ids.append(i) #already int
                res_name.append(l[3])
                t = l[4]
                cngr.append(l[5])
                atomscode_map[l[1]] = i
                code.append(l[1])
                at_types.append(t)
                if nl >6:
                    charge.append( float(l[6]))
                elif len(find_charge)==0:
                    try:
                        chd = charge_defaults
                    except:
                        charge.append( 0.0 )
                    else:
                        charge.append(chd[t])
                else:
                    charge.append(find_charge[l[1]])
                if nl>7:
                    ms= float(l[7])
                else:
                    if len(find_mass)==0:
                        try:
                            ms = maps.elements_mass[self.identify_element(t)]
                        except KeyError:
                            ms = maps.elements_mass[self.identify_element(t[0])]

                    else:
                        ms = find_mass[l[1]]
                mass.append(ms)
                res_num.append(int(l[2]))
                i+=1
            if '[' in line or ']' in line:
                break

        at_ids = np.array(at_ids)
        at_types = np.array(at_types)
        res_num = np.array(res_num)
        res_name = np.array(res_name)
        code = np.array(code)
        charge = np.array(charge)
        mass = np.array(mass)

        cngr = np.array(cngr)
        resnames = np.unique(res_name)
        f = {rn: res_name == rn for rn in res_name}
        molecule_map = {rn : {'at_ids':at_ids[f[rn]],
                              'at_types':at_types[f[rn]],
                              'res_num':res_num[f[rn]],
                              'res_name':res_name[f[rn]],
                              'code':code[f[rn]],
                              'charge':charge[f[rn]],
                              'cngr':cngr[f[rn]],
                              'mass':mass[f[rn]]}
                            for rn in resnames
                            }

        if not hasattr(self,'molecule_map'):
            self.molecule_map = molecule_map
        else:
            self.molecule_map.update(molecule_map)
        if not hasattr(self,'exclusions_map'):
            self.exclusions_map = exclusions_map
        else:
            self.exclusions_map.update(exclusions_map)
        if not hasattr(self,'atomscode_map'):
            self.atomscode_map = atomscode_map
        else:
            self.atomscode_map.update( atomscode_map)


        jd = {'bonds':2,'angles':3,'dihedrals':4,'pairs':2,'exclusions':2}
        topol =  {'bonds':[],'angles':[],'dihedrals':[],'pairs':[],'exclusions':[]}


        for key in ['bonds','angles','dihedrals','pairs','exclusions']:
            for jli in jlines[key]:
                ffd = dict()
                for line in lines[jli+1:]:
                    l = line.split()
                    if len(l)<2:
                        continue
                    try:
                        b = np.array(l[:jd[key]],dtype=int)
                    except:
                        pass
                    else:
                        topol[key].append(b)
                    if '[' in line or ']' in line:
                        break


        bonds = np.array(topol['bonds'])

        try:
            bonds[0]
        except IndexError as e:
            logger.warning('Warning: File {:s} probably contains no bonds\n Excepted ValueError : {:}'.format(file,e))

        try:
            sub = bonds.min()
        except:
            sub =1
        self.sub =sub

        bonds -= sub
        angles = np.array(topol['angles']) - sub
        dihedrals = np.array(topol['dihedrals']) - sub
        pairs = np.array(topol['pairs']) - sub
        exclusions = np.array(topol['exclusions']) - sub





        pairs_per_resname = {t:[] for t in resnames }
        exclusions_per_resname = {t:[] for t in resnames }
        connectivity_per_resname = {t:[] for t in resnames }
        angles_per_resname = {t:[] for t in resnames }
        dihedrals_per_resname = {t:[] for t in resnames }

        for b in bonds:
            i0 = np.where(at_ids == b[0])[0][0]
            i1 = np.where(at_ids == b[1])[0][0]
            assert res_name[i0] == res_name[i1], 'Bond {:d} - {:d} is between two different residues'.format(i0,i1)
            res_nm = res_name[i0]
            connectivity_per_resname[res_nm].append(b)

        for a in angles:
            i0 = np.where(at_ids == a[0])[0][0]
            i1 = np.where(at_ids == a[1])[0][0]
            i2 = np.where(at_ids == a[2])[0][0]
            assert res_name[i0] == res_name[i1], 'Angle ids {:d} - {:d} is between two different residues'.format(i0,i1)
            assert res_name[i0] == res_name[i2], 'Angle ids {:d} - {:d} is between two different residues'.format(i0,i2)
            res_nm = res_name[i0]
            angles_per_resname[res_nm].append(a)

        for d in dihedrals:
            i0 = np.where(at_ids == d[0])[0][0]
            i1 = np.where(at_ids == d[1])[0][0]
            i2 = np.where(at_ids == d[2])[0][0]
            i3 = np.where(at_ids == d[2])[0][0]
            assert res_name[i0] == res_name[i1], 'Dihedral ids {:d} - {:d} is between two different residues'.format(i0,i1)
            assert res_name[i0] == res_name[i2], 'Dihedral ids {:d} - {:d} is between two different residues'.format(i0,i2)
            assert res_name[i0] == res_name[i3], 'Dihedral ids {:d} - {:d} is between two different residues'.format(i0,i3)
            res_nm = res_name[i0]
            dihedrals_per_resname[res_nm].append(d)

        for p in pairs:
            i0 = np.where(at_ids == p[0])[0][0]
            i1 = np.where(at_ids == p[1])[0][0]
            assert res_name[i0] == res_name[i1], 'Pair {:d} - {:d} is between two different residues'.format(i0,i1)
            res_nm = res_name[i0]
            pairs_per_resname[res_nm].append(p)
        for e in exclusions:
            i0 = np.where(at_ids == e[0])[0][0]
            i1 = np.where(at_ids == e[1])[0][0]
            assert res_name[i0] == res_name[i1], 'Exclusion {:d} - {:d} is between two different residues'.format(i0,i1)
            res_nm = res_name[i0]
            exclusions_per_resname[res_nm].append(e)

        for c in ['connectivity','angles','dihedrals','pairs','exclusions']:
            name = c+'_per_resname'
            var  = locals()[name]
            if not hasattr(self,name):
                setattr(self,name,var)
            else:
                attr = getattr(self,name)
                #updating
                for t,bad in var.items():

                    attr[t] = bad
                setattr(self,name,attr)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)  # report total setup cost
        return

    def make_ff_from_itp(self,file):
        if not hasattr(self,'ff'):
            self.ff = self.FFparams()
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
        nline = dict()
        lookfor = ['atomtypes','atoms','bonds','angles','dihedrals','bondtypes','angletypes','dihedraltypes']
        nline = {k:set() for k in lookfor}
        for j,line in enumerate(lines):
            for k in lookfor:
                if k in line and '[' in line and ']' in line:
                    nline[k].add(j)
        residues = []
        for line in lines[list(nline['atoms'])[0]+1:]:
            if ';' in line :
                continue
            l = line.strip().split()
            if len(l) == 0 :
                continue

            if '[' in line or ']' in line :
                break
            residues.append(l[3])
        ures = np.unique(residues)
        if len(ures) != 1:
            raise ValueError('Residues in this itp file are not unique')

        jd = {'atomtypes':1,'bondtypes':2,'angletypes':3,'dihedraltypes':4,'bonds':2,'angles':3,'dihedrals':4}
        names = {'atomtypes':'atomtypes','bonds':'bondtypes',
                 'angles':'angletypes','dihedrals':'dihedraltypes',
                 'bondtypes':'bondtypes','angletypes':'angletypes','dihedraltypes':'dihedraltypes'}
        mol_ids = self.mol_ids[self.mol_names==ures[0]]

        for k in ['atomtypes','bondtypes','angletypes','dihedraltypes']:
            attr = dict()
            attr_name = names[k]
            for nlin in list(nline[k]):
                for j,line in enumerate(lines[nlin+1:]):

                    if ';' in line:
                        continue
                    if '[' in line or ']' in line : break
                    l = line.strip().split()

                    if len(l) ==0: continue

                    if k =='atomtypes':
                        if not hasattr(self.ff,'atomtypes'):
                            self.ff.atomtypes = dict()
                        aid = self.atomscode_map[l[0]]
                        for res_id in np.unique(mol_ids):
                            a = self.loc_id_to_glob[res_id][aid]
                            ty = self.at_types[a]

                            mass1 = self.atom_mass[a]
                            mass2 = float(l[2]) if len(l) == 7 else float(l[1])
                            assert round(mass1,3) == round(mass2,3),'mass1 = {:4.3f}  while mass2 = {:4.3f}'.format(mass1,mass2)
                            ch = self.atom_charge[a]
                            code = self.atom_code[a]
                            atyc = ty if self.bytype else code
                            val = (atyc,str(mass1),str(ch),*l[-3:])
                            attr[ty] = val
                    elif k in ['bondtypes','angletypes','dihedraltypes']:
                        tys = getattr(self,k.split('types')[0]+'_'+'types')
                        for ty in tys:

                            atyc = ty #if self.bytype else code
                            val = ('  '.join(atyc),*l[jd[k]:])
                            attr[ty] = val

                    else:
                        raise Exception('something wrong with naming. Check your code')
                    ass.update_dict_in_object(self.ff, attr_name, attr)
        for k in ['bonds','angles','dihedrals']:
            attr = dict()
            attr_name = names[k]
            for nlin in list(nline[k]):
                for j,line in enumerate(lines[nlin+1:]):

                    if ';' in line:
                        continue
                    if '[' in line or ']' in line : break
                    l = line.strip().split()

                    if len(l) ==0: continue
                    aid =  tuple(int(i)-self.sub for i in l[:jd[k]])
                    for res_id in np.unique(mol_ids):
                        a = tuple(self.loc_id_to_glob[res_id][i] for i in aid)
                        cid,ty = self.sorted_id_and_type(a)
                        code = tuple(self.atom_code[i] for i in cid)
                        atyc = ty if self.bytype else code
                        try:
                            default_val = getattr(self.ff,attr_name)[atyc]
                        except KeyError:
                            default_val = ( 'no default value check your code',)
                        except AttributeError:
                            default_val = ( 'not in {:s} check your code'.format(attr_name),)

                        proposed_val = ('  '.join(atyc),*l[jd[k]:])
                        if len(default_val)>len(proposed_val):
                            val = default_val
                        else:
                            val = proposed_val
                        attr[ty] = val
                    ass.update_dict_in_object(self.ff, attr_name, attr)
        return

    def read_topology(self):
        if '.gro' == self.topol_file[-4:]:
            self.read_gro_topol()  # reads from gro file
            try:
                self.connectivity_file
            except:
                raise NotImplementedError('connectivity_info = {} is not implemented yet'.format(self.connectivity_file))
            else:
                if type(self.connectivity_info) is dict:
                    self.map_the_topology(self.connectivity_info)
                else:
                    self.read_gromacs_topology() # reads your itp files to get the connectivity
            if hasattr(self,'fftop'):
                self.read_topfile_ff(self.fftop)
        elif '.dat' == self.topol_file[-4:]:
            self.read_lammps_topol()
            if hasattr(self ,'fftop'):
                self.read_incfile_ff(self.fftop)

        else:
            raise Exception('file {:s} not implemented'.format(self.topol_file.split('.')[-1]))
        #elif '.mol2' == self.topol_file[-5:]:
            #self.read_mol2_topol()
        return

    def read_lammps_topol(self):
        t0 = perf_counter()
        with open(self.topol_file,'r') as f:
            lines = f.readlines()
            f.closed

        def get_value(lines,valuename,dtype=int):
            for line in lines:
                if valuename in line:
                    return dtype(line.split()[0])

        def get_line_of_header(lines,header):
            for i,line in enumerate(lines):
                if header in line:
                    return i
        values = ['atoms','bonds','angles','dihedrals','impropers',
                  'atom types', 'bond types', 'angle types',
                  'dihedral types','improper types']
        headers = ['Masses','Atoms','Bonds','Angles','Dihedrals']

        numbers = {v:get_value(lines,v) for v in values}

        header_lines = {hl:get_line_of_header(lines,hl) for hl in headers}

        natoms = numbers['atoms']
        mol_ids = np.empty(natoms,dtype=int)
        mol_nms = np.empty(natoms,dtype=object)
        at_tys  = np.empty(natoms,dtype=object)
        at_ids  = np.empty(natoms,dtype=int)
        atom_charge = np.empty(natoms,dtype=float)
        #atom_mass = np.empty(natoms,dtype=float)
        hla = header_lines['Atoms']
        atom_lines = lines[hla+2 : hla+2+natoms]
        ncols = len(atom_lines[0].split())

        self.charge_map = dict()
        for i,line in enumerate(atom_lines):
            l = line.strip().split()
            mol_ids[i] = int(l[1])
            mol_nms[i] = l[1]
            at_tys[i] = l[2]
            at_ids[i] = l[0]
            if ncols==4 or ncols ==7 or ncols==10:
                atom_charge[i] = float(l[3])
                self.charge_map[l[2]] = float(l[3])
        starts_from = at_ids.min()
        at_ids -= starts_from

        sort_ids = at_ids.argsort()
        mol_ids = mol_ids[sort_ids]
        mol_nms = mol_nms[sort_ids]
        at_tys = at_tys[sort_ids]
        at_ids = at_ids[sort_ids]
        atom_charge = atom_charge[sort_ids]

        starts_from = 1
        for i in range(1,natoms):
            if at_ids[i-1]+1 != at_ids[i]:
                raise Exception('There are missing atoms')


        self.mol_ids = mol_ids
        self.mol_names = mol_nms
        self.at_types = at_tys
        self.atom_code = at_tys
        self.at_ids = at_ids
        self.atom_charge = atom_charge

        self.find_locGlob()
        # Measure bonds
        self.connectivity= dict()

        hlb = header_lines['Bonds']
        print(header_lines)
        if hlb is None:
            pass
        else:
            nbonds = numbers['bonds']
            bond_lines = lines[hlb+2:hlb + nbonds+2]
            for  i,line in enumerate(bond_lines):
                b = line.strip().split()
                id0 = int(b[-2]) - starts_from
                id1 = int(b[-1]) - starts_from
                conn_id,c_type =  self.sorted_id_and_type((id0,id1))
                if conn_id in self.connectivity:
                    logger.warning('{} is already in connectivity '.format(conn_id))
                self.connectivity[conn_id] = c_type

        # Measure angles
        self.angles = dict()

        hla = header_lines['Angles']
        if hla is None:
            pass
        else:
            nangles = numbers['angles']
            angle_lines = lines[hla+2:hla + nangles+2]
            for  i,line in enumerate(angle_lines):
                a = line.strip().split()
                id0 = int(a[-3]) - starts_from
                id1 = int(a[-2]) - starts_from
                id2 = int(a[-1]) - starts_from
                a_id,a_type =  self.sorted_id_and_type((id0,id1, id2))
                if a_id in self.angles:
                    logger.warning('{} is already in angles '.format(conn_id))
                self.angles[a_id] = a_type


        # Measure dihedrals
        self.dihedrals = dict()

        hld = header_lines['Dihedrals']
        if hld is None:
            pass
        else:
            ndihedrals = numbers['dihedrals']
            dihedral_lines = lines[hld+2:hld + ndihedrals+2]
            for  i,line in enumerate(dihedral_lines):
                d = line.strip().split()
                id0 = int(d[-4]) - starts_from
                id1 = int(d[-3]) - starts_from
                id2 = int(d[-2]) - starts_from
                id3 = int(d[-1]) - starts_from
                d_id,d_type =  self.sorted_id_and_type((id0, id1, id2, id3))
                if d_id in self.dihedrals:
                    logger.warning('{} is already in dihedrals '.format(conn_id))
                self.dihedrals[d_id] = d_type

        # Measure mass
        self.mass_map = dict()
        hlm = header_lines['Masses']

        for line in lines[hlm+2:+hlm+2+numbers['atom types']]:
            l = line.strip().split()
            self.mass_map[l[0]] = float(l[1])

        self.atom_mass = np.array([self.mass_map[t] for t in self.at_types])
        self.find_locGlob()
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def read_incfile_ff(self, incfile):
        with open(incfile,'r') as f:
            lines = f.readlines()
            f.closed

        import re

        kcal_to_kj = 4.184
        ang_to_nm  = 0.1

        pair = dict()
        bonds = dict()
        angles = dict()
        dihedrals = dict()

        fudgeLJ = 1.0
        fudgeQQ = 1.0

        for lin in lines:
            line = lin.strip('\n')

            print(line)
            # --- special_bonds ---
            if line.startswith("special_bonds"):
                # example: special_bonds lj/coul 0.0 0.0 1.0
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(nums) == 3:
                    # in GROMACS:
                    # fudgeLJ = 1-2 LJ scaling  (LAMMPS second number)
                    # fudgeQQ = 1-2 Coul scaling (LAMMPS second number)
                    fudgeLJ = float(nums[1])
                    fudgeQQ = float(nums[1])

            # --- pair coefficients ---
            elif line.startswith("pair_coeff"):
                # pair_coeff i j epsilon sigma
                parts = line.split()
                i, j = str(parts[1]), str(parts[2])
                eps_lmp = float(parts[3])       # kcal/mol
                sigma_lmp = float(parts[4])     # Å

                eps_gmx = eps_lmp * kcal_to_kj
                sig_gmx = sigma_lmp * ang_to_nm

                pair[(i,j)] = (sig_gmx, eps_gmx)

            # --- harmonic bonds ---
            elif line.startswith("bond_coeff"):
                # bond_coeff id k r0
                parts = line.split()
                btype = str(parts[1])
                k = float(parts[2]) * kcal_to_kj * 2.0/(ang_to_nm)**2   # LAMMPS uses K*(r-r0)^2 ; GMX uses 1/2*K*(r-r0)^2
                r0 = float(parts[3]) * ang_to_nm
                bonds[btype] = (r0, k)

            # --- angles ---
            elif line.startswith("angle_coeff"):
                # angle_coeff id k theta0
                parts = line.split()
                atype = str(parts[1])
                k = float(parts[2]) * kcal_to_kj * 2.0
                theta = float(parts[3])
                angles[atype] = ( theta, k)

            # --- fourier dihedrals ---
            elif line.startswith("dihedral_coeff"):
                # dihedral_coeff id n c0 phi0 c1 ... (LAMMPS format)
                parts = line.split()
                dtype = str(parts[1])
                # Fourier terms come in triples: k, multiplicity, phase
                # Convert to GROMACS "fourier" periodic form:
                coeffs = []
                rest = parts[3:]  # skip: id style nterms
                # Parse triples
                for k in range(0, 3*int(parts[2]), 3):
                    amp = float(rest[k]) * kcal_to_kj  # convert units
                    mult = int(float(rest[k+1]))
                    phase = int(float(rest[k+2]))
                    if phase not in [ 180, 0]: raise ValueError('phase is not 180 or 0')
                    coeffs.append(( phase, amp, mult))

                dihedrals[dtype] =  coeffs
        self.bond_coeffs = bonds
        self.angle_coeffs = angles
        self.dihedral_coeffs = dihedrals
        self.pair_coeffs = pair
        self.ff = self.FFparams()
        for p, v in  self.pair_coeffs.items():
            if p[0] == p[1]:
                s = f'{p[0]} {p[1]} {self.mass_map[str(p[0])]:6.5f} {self.charge_map[str(p[0])]:6.5f} A {v[0]:6.5f} {v[1]:6.5f}'
                self.ff.atomtypes[p[0]] = tuple(s.split())
            else:
                 s = f'{p[0]} {p[1]} 1 {v[0]:6.5f} {v[1]:6.5f}'
                 self.ff.nonbond_params[p] =  (f'{p[0]} {p[1]}', '1', f'{v[0]:6.5f}', f'{v[1]:6.5f}')
        for b,v in self.bond_coeffs.items():
            s = f'{b} 1 {v[0]:6.5f} {v[1]:6.5f}'
            self.ff.bondtypes[b] =  tuple(s.split())

        for a,v in self.angle_coeffs.items():
            s = f'{a} 1 {v[0]:6.5f} {v[1]:6.5f}'
            self.ff.angletypes[a] =  tuple(s.split())
        for d,v in self.dihedral_coeffs.items():
            if type(v) is list:
                x = [tuple(f'{d} 9 {vv[0]:1d} {vv[1]:6.5f} {vv[2]:1d}'.split()) for vv in v]
            self.ff.dihedraltypes[d] =  x
        return

    def read_gro_topol(self):
        with open(self.topol_file,'r') as f:
            f.readline()
            natoms = int(f.readline().strip())
            #allocate memory
            mol_ids = np.empty(natoms,dtype=int)
            mol_nms = np.empty(natoms,dtype=object)
            at_tys  = np.empty(natoms,dtype=object)
            at_ids  = np.empty(natoms,dtype=int)
            for i in range(natoms):
                line = f.readline()
                mol_ids[i] = int(line[0:5].strip())
                mol_nms[i] = line[5:10].strip()
                at_tys[i] = line[10:15].strip()
                at_ids[i] = i
            f.close()


        self.mol_ids = mol_ids
        self.mol_names = mol_nms
        self.at_types = at_tys
        self.at_ids = at_ids

        self.find_locGlob()

        return

    def find_locGlob(self):
        loc_id_to_glob = dict() ; glob_id_to_loc = dict()
        for j in np.unique(self.mol_ids):
            loc_id_to_glob[j] = dict()
            glob_id_to_loc[j] = dict()
            filt = self.mol_ids== j
            res_nm = np.unique(self.mol_names[filt])
            if res_nm.shape !=(1,):
                raise ValueError('many names for a residue, res_id = {:d}'.format(j))
            else:
                res_nm = res_nm[0]
            g_at_id = self.at_ids[filt]

            for i,g in enumerate(g_at_id):
                loc_id = i
                loc_id_to_glob[j][loc_id] = g
                glob_id_to_loc[j][g] = loc_id

        self.loc_id_to_glob = loc_id_to_glob
        self.glob_id_to_loc = glob_id_to_loc
        return
    def find_same_type_atoms(self):


        same_at , same_b, same_a, same_d = set(), set(), set(), set()

        for a1, t1 in self.ff.atomtypes.items():
            for a2, t2 in  self.ff.atomtypes.items():
                if ( t1[1] == t2[1] ) and ( t1[3:] == t2[3:] ):
                    same_at.add( (a1, a2) )
        for a1, t1 in self.ff.bondtypes.items():
            for a2, t2 in  self.ff.bondtypes.items():
                if t1[1:] == t2 [1:] :
                    same_b.add( (a1, a2) )
        for a1, t1 in self.ff.angletypes.items():
            for a2, t2 in  self.ff.angletypes.items():
                if t1[1:] == t2 [1:] :
                    same_a.add( (a1, a2) )
        for a1, t1 in self.ff.dihedraltypes.items():
            for a2, t2 in  self.ff.dihedraltypes.items():
                if t1[1:] == t2 [1:] :
                    same_d.add( (a1, a2) )
        same_types = []

        for i, ti in enumerate(self.ff.atomtypes):
            for j, tj in enumerate(self.ff.atomtypes):
                Tr1 = (ti,tj) in same_at or (tj,ti) in same_at

                if not Tr1: continue

                Tr2 = False
                for bond_pairs in same_b:
                    p1, p2 = bond_pairs
                    if (p1[0], p2[1]) in same_at:

                        Tr2 = True

                Tr3 = False
                for angle_pairs in same_a:
                    p1, p2 = angle_pairs
                    tle =  (p1[0], p2[1]) in same_at
                    tri =  (p1[1], p2[2]) in same_at
                    if tle and tri:
                        Tr3 = True

                Tr4 = False
                for dih_pairs in same_d:
                    p1, p2 = dih_pairs
                    tle =  (p1[0], p2[1]) in same_at
                    trm =  (p1[1], p2[2]) in same_at
                    tri =  (p1[2], p2[3]) in same_at
                    if tle and trm and tri:
                        Tr4 = True

                if Tr1 and Tr2 and Tr3 and Tr4:

                    same_types.append( (ti, tj) )
        groups = []      # list of sets
        lookup = dict()      # maps atom -> group index

        pairs = same_types
        for a, b in pairs:
            if a in lookup and b in lookup:
                # merge groups if different
                if lookup[a] != lookup[b]:
                    ga, gb = lookup[a], lookup[b]
                    groups[ga].update(groups[gb])
                    for x in groups[gb]:
                        lookup[x] = ga
                    groups[gb] = set()  # empty merged bucket
            elif a in lookup:
                groups[lookup[a]].add(b)
                lookup[b] = lookup[a]
            elif b in lookup:
                groups[lookup[b]].add(a)
                lookup[a] = lookup[b]
            else:
                # create new group
                idx = len(groups)
                groups.append({a, b})
                lookup[a] = idx
                lookup[b] = idx

        # filter out empty merged sets
        groups = [sorted(list(g)) for g in groups if len(g) > 0]
        groups.sort()

        return groups

    def change_at_types(self, am,lammps_style=None):

        self.at_types = np.array([am[x] for x in self.at_types ])

        self.atom_code = self.at_types
        self.connectivity = {k:(am[v[0]] , am[v[1]])                   for k,v in self.connectivity.items() }
        self.angles = {k:(am[v[0]] , am[v[1]], am[v[2]])               for k,v in self.angles.items() }
        self.dihedrals = {k:(am[v[0]] , am[v[1]], am[v[2]], am[v[3]])  for k,v in self.dihedrals.items() }

        self.ff.atomtypes  = {am[k] : tuple([f'{am[k]}', *v[1:] ])
                    for k, v in self.ff.atomtypes.items() }

        def find_type( k , kind ):

            if kind == 'bonds':
                if lammps_style is None:
                    ty = (am[k[0]], am[k[1]] )
                else:
                    ty = lammps_style['bonds'][k]


            if kind == 'angles':

                if lammps_style is None:
                    ty = (am[k[0]], am[k[1]], am[k[2]] )
                else:
                    ty = lammps_style['angles'][k]

            if kind == 'dihedrals':

                if lammps_style is None:
                    ty = (am[k[0]], am[k[1]], am[k[2]], am[k[3]] )
                else:
                    ty = lammps_style['dihedrals'][k]
            return ty
        bt = dict()
        for k, v in self.ff.bondtypes.items():
            ty = find_type(k, 'bonds')
            bt[ty] = tuple( [ '  '.join(ty), *v[1:] ] )
        self.ff.bondtypes =  bt

        at = dict()
        for k, v in self.ff.angletypes.items():
            ty = find_type(k, 'angles')
            at[ty] = tuple( [ '  '.join(ty), *v[1:] ] )
        self.ff.angletypes = at

        dt = dict()
        for k,v in self.ff.dihedraltypes.items():
            ty = find_type(k, 'dihedrals')
            if type(v) is list:
                dt[ty] = [tuple([ '  '.join(ty), *vv[1:] ]) for vv in v]
            else:
                dt[ty] = tuple([ '  '.join(ty), *v[1:] ])
        self.ff.dihedraltypes = dt

        self.ff.nonbond_params  = {(am[k[0]], am[k[1]]) : tuple([f'{am[k[0]]}  {am[k[1]]}', *v[1:] ])
                    for k, v in self.ff.nonbond_params.items() }
        return

    @property
    def natoms(self):
        return self.at_ids.shape[0]
    @property
    def ndihedrals(self):
        return len(self.dihedrals)
    @property
    def nbonds(self):
        return len(self.connectivity)
    @property
    def nangles(self):
        return len(self.angles)
    @staticmethod
    def unique_values(iterable):
        try:
            iter(iterable)
        except:
            raise Exception('Give an ass.iterable variable')
        else:
            un = []
            for x in iterable:
                if x not in un:
                    un.append(x)
            return un
    @property
    def atom_types(self):
        return self.unique_values(self.at_types)
    @property
    def bond_types(self):
        return self.unique_values(self.connectivity.values())
    @property
    def angle_types(self):
        return self.unique_values(self.angles.values())
    @property
    def dihedral_types(self):
        return self.unique_values(self.dihedrals.values())


    def sorted_type(self,t):
        if t[0]<=t[-1]:
            t = tuple(t)
        else:
            t = tuple(t[::-1])
        return t
    def sorted_id_and_type(self,a_id):
        t = [self.at_types[i] for i in a_id]
        if t[0]<=t[-1]:
            t = tuple(t)
            a_id = tuple(a_id)
        else:
            t = tuple(t[::-1])
            a_id = tuple(a_id[::-1])
        #if a_id[0]<=a_id[-1]:
       #     a_id = tuple(a_id)
        #else:
        #    a_id = tuple(a_id[::-1])
        return a_id,t

    @property
    def refining_angles_condition(self):
        try:
            a  = self.refine_angles
        except:
            return False
        else:
            return  a
    @property
    def refining_dihedrals_condition(self):
        try:
            a  = self.refine_dihedrals
        except:
            return True
        else:
            return  a
    def find_new_angdihs(self,new):
        angdihs = dict()
        for neib in self.neibs[new[0]]:
            if neib in new:
                continue
            idn = (neib,*new)
            idns,t = self.sorted_id_and_type(idn)
            angdihs[idns] = t
        for neib in self.neibs[new[-1]]:
            if neib in new:
                continue
            idn = (*new,neib)
            idns,t = self.sorted_id_and_type(idn)
            angdihs[idns] = t
        return angdihs

    def find_angles(self):
        '''
        Computes the angles of a system in dictionary format
        key: (atom_id1,atom_id2,atom_id3)
        value: object Angle
        Method:
            We search the neihbours of bonded atoms.
            If another atom is bonded to one of them an angle is formed
        We add in the angle the atoms that participate
        '''
        if not self.refining_angles_condition:
            return

        #t0 = perf_counter()
        self.angles = dict()
        for k in self.connectivity.keys():
            #"left" side angles k[0]
            for neib in self.neibs[k[0]]:
                if neib in k: continue
                ang_id ,ang_type = self.sorted_id_and_type((neib,k[0],k[1]))
                if ang_id[::-1] not in self.angles.keys():
                    self.angles[ang_id] = ang_type
            #"right" side angles k[1]
            for neib in self.neibs[k[1]]:
                if neib in k: continue
                ang_id ,ang_type = self.sorted_id_and_type((k[0],k[1],neib))
                if ang_id[::-1] not in self.angles.keys():
                    self.angles[ang_id] = ang_type
        #tf = perf_counter()-t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def find_dihedrals(self):
        '''
        Computes dihedrals of a system based on angles in dictionary
        key: (atom_id1,atom_id2,atom_id3,atom_id4)
        value: object Dihedral
        Method:
            We search the neihbours of atoms at the edjes of Angles.
            If another atom is bonded to one of them a Dihedral is formed is formed
        We add in the angle the atoms that participate
        '''
        if not self.refining_dihedrals_condition:
            return
        #t0 = perf_counter()
        self.dihedrals=dict()
        for k in self.angles.keys():
            #"left" side dihedrals k[0]
            for neib in self.neibs[k[0]]:
                if neib in k: continue
                dih_id,dih_type = self.sorted_id_and_type((neib,k[0],k[1],k[2]))
                if dih_id[::-1] not in self.dihedrals:
                    self.dihedrals[dih_id] = dih_type
            #"right" side dihedrals k[2]
            for neib in self.neibs[k[2]]:
                if neib in k: continue
                dih_id,dih_type = self.sorted_id_and_type((k[0],k[1],k[2],neib))
                if dih_id[::-1] not in self.dihedrals:
                    self.dihedrals[dih_id] = dih_type
        #tf = perf_counter()
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def find_masses(self):

        mass = np.empty(self.natoms,dtype=float)
        for i in range(self.natoms):
            mass[i] = self.mass_map[self.at_types[i]]
        self.atom_mass = mass

        return

    @staticmethod
    def get_equal_from_string(s,v,make=float):
        try:
            x = s.split(v+'=')[-1].split()[0]
        except:
            x = s.split(v+' =')[-1].split()[0]
        return make(x)

    def read_gro_by_frame(self,ofile,frame):
        line = ofile.readline()
        l = line.strip().split()
        if len(l)==0:
            return False
        #first line
        try:
            time = self.get_equal_from_string(line.strip(),'t')
        except:
            logger.warning('Warning: in gro file. There is no time info')
            time = 0
        try:
            step = self.get_equal_from_string(line.strip(),'step',int)
        except:
            step = 0
            logger.warning('Warning: in gro file. There is no step info')
        self.timeframes[frame] = {'time':time,'step':step}
        # second line
        natoms = int(ofile.readline().strip())
        if natoms != self.natoms:
            raise ValueError('This frame has {} atoms instead of {}'.format(natoms,self.natoms))

        #file
        coords = np.empty((natoms,3),dtype=float)
        for i in range(natoms):
            line = ofile.readline()
            l=line[20:44].split()
            coords[i,0] = float(l[0])
            coords[i,1] = float(l[1])
            coords[i,2] = float(l[2])

        box = np.array(ofile.readline().strip().split(),dtype=float)

        self.timeframes[frame]['coords'] = coords
        self.timeframes[frame]['boxsize'] = box
        return True

    def read_trr_by_frame(self,ofile,frame):
        try:
            header,data = ofile.read_frame()
        except EOFError:
            raise EOFError
        except Exception:
            return True
        self.timeframes[frame] = header
        self.timeframes[frame]['boxsize'] = np.diag(data['box']).copy()
        self.timeframes[frame]['coords'] = data['x']
        try:
            self.timeframes[frame]['velocities'] = data['v']
        except:
            pass
        try:
            self.timeframes[frame]['forces'] = data['f']
        except:
            pass
        return True

    def read_lammpstrj_by_frame(self,reader,frame):
        conf = reader.readNextStep()
        if conf is None:
            return False
        if not reader.isSorted(): reader.sort()
        try:
            uxs = conf['x']
        except KeyError:
            uxs = conf['xu']
        try:
            uys = conf['y']
        except KeyError:
            uys = conf['yu']
        try:
            uzs = conf['z']
        except KeyError:
            uzs = conf['zu']
        # allocate
        natoms = uxs.shape[0]
        coords = np.empty((natoms,3))
        coords[:,0] = uxs ; coords[:,1] = uys ; coords[:,2] = uzs
        cbox = conf['box_bounds']
        tricl = cbox[:,2] != 0.0
        if tricl.any():
            raise NotImplementedError('Triclinic boxes are not implemented')
        offset = cbox[:,0]
        boxsize = cbox[:,1]-cbox[:,0]
        coords -= offset
        #make nm
        boxsize/=10
        coords/=10
        try:
            dt = self.kwargs['dt']
        except KeyError:
            dt =1e-6
        self.timeframes[frame] = {'conf':conf,'time':conf['step_no']*dt,'step':conf['step_no'],
                                  'boxsize':boxsize,'coords':coords}
        return True

    def read_from_disk_or_mem(self,ofile,frame):
        def exceptEOF():
            try:
                ret = self.read_by_frame(ofile, frame)
            except EOFError:
                return False
            else:
                return ret

        if self.memory_demanding:
            return exceptEOF()
        elif frame in self.timeframes.keys():
            self.is_the_frame_read =True
            return True
        else:
            try:
                if self.is_the_frame_read:
                    return False
            except AttributeError:
               return exceptEOF()

    def read_lammpstrj_file(self,num_end=int(1e16)):
        t0 = perf_counter()
        with lammpsreader.LammpsTrajReader(self.trajectory_file) as ofile:
            nframes = 0
            while( self.read_lammpstrj_by_frame(ofile, nframes) and nframes<=num_end):
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes += 1
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)  # log with frame count
        return

    def read_trr_file(self,num_end=int(1e16)):
        t0 = perf_counter()
        with GroTrrReader(self.trajectory_file) as ofile:
            end = False
            nframes = 0
            while( end == False and nframes<=num_end):
                try:
                    self.read_trr_by_frame(ofile, nframes)
                except EOFError:
                    end = True
                else:
                    if self.memory_demanding:
                        del self.timeframes[nframes]
                    nframes += 1
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return

    def read_gro_file(self,num_end=int(1e16)):
        t0 = perf_counter()
        with open(self.trajectory_file,'r') as ofile:
            nframes =0
            while(self.read_gro_by_frame(ofile,nframes) and nframes<=num_end):

                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes+=1
            ofile.close()
            tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return

    def read_file(self,trajectory_file=None,num_end=int(1e16)):

        if trajectory_file is None:
            try:
                # just checking if these attributes excist
                self.trajectory_file
                self.trajectory_file_type
                self.read_by_frame
                self.traj_opener
                self.traj_opener_args
            except AttributeError:
                raise Exception('You need to provide a trajectory file to read. The reading is not set')
        else:
            self.setup_reading(trajectory_file)

        if   self.traj_file_type == 'gro':
            self.read_gro_file(num_end)
        elif self.traj_file_type == 'trr':
            self.read_trr_file(num_end)
        elif self.traj_file_type =='lammpstrj':
            self.read_lammpstrj_file(num_end)
        return

    def setup_reading(self,trajectory_file):
        self.trajectory_file = trajectory_file
        if '.gro' == self.trajectory_file[-4:]:
            self.traj_file_type = 'gro'
            self.read_by_frame = self.read_gro_by_frame # function
            self.traj_opener = open
            self.traj_opener_args = (self.trajectory_file,)
        elif '.trr' == self.trajectory_file[-4:]:
            self.traj_file_type ='trr'
            self.read_by_frame =  self.read_trr_by_frame # function
            self.traj_opener = GroTrrReader
            self.traj_opener_args = (self.trajectory_file,)
        elif '.lammpstrj' == self.trajectory_file[-10:]:
            self.traj_file_type ='lammpstrj'
            self.read_by_frame = self.read_lammpstrj_by_frame
            self.traj_opener = lammpsreader.LammpsTrajReader
            self.traj_opener_args = (self.trajectory_file,)
        else:
            raise NotImplementedError('Trajectory file format ".{:s}" is not yet Implemented'.format(trajectory_file.split('.')[-1]))

    def write_gro_file(self,fname=None,whole=False,
                       option='',frames=None,step=None,**kwargs):
        t0 = perf_counter()
        options = ['','transmiddle','translate']
        if option not in options:
            raise ValueError('Available options are : {:s}'.format(', '.join(options)))

        if fname is None:
            fname = 'Analyisis_written.gro'
        with open(fname,'w') as ofile:
            for frame,d in self.timeframes.items():
                if frames is not None:
                    if  frame <frames[0] or frame>frames[1]:
                        continue
                if step is not None:
                    if frame%step  !=0:
                        continue

                if option =='transmiddle':
                    coords = self.translate_particle_in_box_middle(self.get_coords(frame),
                                                          self.get_box(frame))
                elif option=='':
                    coords = self.get_coords(frame)

                elif option=='translate':
                    v = np.array(kwargs['trans'])
                    coords = self.get_coords(frame) + v
                    coords = implement_pbc(coords,d['boxsize'])
                else:
                    raise NotImplementedError('option "{:}" Not implemented'.format(option))
                if whole:
                        coords = self.unwrap_coords(coords, d['boxsize'])

                self.write_gro_by_frame(ofile,
                                        coords, d['boxsize'],
                                        time = d['time'],
                                        step =d['step'])
            ofile.close()
        tf = perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def write_gro_by_frame(self,ofile,coords,box,name='gro_by_frame',time=0,step=0):
        ofile.write('{:s},  t=   {:4.3f}  step=   {:8.0f} \n'.format(name,time,step))
        ofile.write('{:6d}\n'.format(coords.shape[0]))
        for i in range(coords.shape[0]):
            c = coords[i]
            ofile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'\
            % (self.mol_ids[i],self.mol_names[i], self.at_types[i],self.at_ids[i%99999]+1 ,c[0],c[1] ,c[2] ))
        ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))
        return

    def renumber_residues(self,start_from=1):
        mol_ids = np.empty(self.natoms,dtype=int)
        counter =start_from
        mol_ids[0] = start_from

        for i in range(1,self.natoms):
            mid0 = self.mol_ids[i-1]
            mid1 = self.mol_ids[i]
            if mid1!=mid0:
                counter+=1
            mol_ids[i] = counter
        self.mol_ids = mol_ids

        return

    def merge_ff(self,obj,add=''):
        a = self.add_tuple
        names = ['atomtypes','bondtypes','angletypes','dihedraltypes','nonbond_params']
        def mod0(t):
            if type(t) is not list:
                t = list(t)
                t[0] = ' '.join([b +add for b in t[0].split(' ')  ])
                return tuple(t)
            else:
                li = []
                for xx in t:
                    x = list(xx)
                    x[0] = ' '.join([b +add for b in x[0].split(' ')  ])
                    li.append(tuple(x))
                return li
        def mod01(t):
            t = list(t)
            t[0] = t[0] + add
            t[1] = t[1] + add
            try:
                float(t[1])
            except:
                t[1] =t[1][:-1]
            return tuple(t)
        for name in names:
            ffdata = getattr(self.ff,name)
            medata = getattr(obj.ff,name)
            if name =='atomtypes':
                ffdata.update({k+add:mod01(v) for k,v in medata.items() })
            else:
                ffdata.update({a(k,add):mod0(v) for k,v in medata.items() })
        if hasattr(obj.ff,'posres') and not hasattr(self.ff,'posres'):
            self.ff.posres = obj.ff.posres
        elif hasattr(obj.ff,'posres') and  hasattr(self.ff,'posres'):
            self.ff.posres.update( obj.ff.posres )
        return
    @staticmethod
    def add_tuple(t,add=''):
        return tuple(y+add for y in list(t))
    def update_topology(self,n,obj,add=''):
        a = self.add_tuple
        self.connectivity.update( {(n+c[0],n+c[1]) : a(t,add) for c,t in obj.connectivity.items()} )
        self.pairs.update( {(n+c[0],n+c[1]):a(t,add) for c,t in obj.pairs.items()} )
        self.exclusions.update( {(n+c[0],n+c[1]):a(t,add) for c,t in obj.exclusions.items()} )
        #self.find_neibs()
        self.neibs.update({n+aid: {n+a for a in neibs} for aid,neibs in obj.neibs.items()} )
        self.angles.update( {(n+c[0],n+c[1],n+c[2]):a(t,add) for c,t in obj.angles.items()} )
        self.dihedrals.update( {(n+c[0],n+c[1],n+c[2],n+c[3]):a(t,add) for c,t in obj.dihedrals.items()} )
        return
    def merge_system(self,obj,add=''):
        n = self.natoms

        self.update_topology(n,obj,add)

        self.at_ids = np.concatenate((self.at_ids,obj.at_ids+self.at_ids.max()))
        self.at_types = np.concatenate((self.at_types,obj.at_types +add))
        self.mol_ids = np.concatenate((self.mol_ids,obj.mol_ids + self.mol_ids.max()))
        self.mol_names = np.concatenate((self.mol_names,obj.mol_names))
        self.atom_mass = np.concatenate((self.atom_mass,obj.atom_mass))
        self.atom_charge = np.concatenate((self.atom_charge,obj.atom_charge))
        self.atom_code = np.concatenate((self.atom_code,obj.atom_code+add))
        for frame in self.timeframes:
            c1 = self.get_coords(frame)
            c2 = obj.get_coords(frame)
            self.timeframes[frame]['coords'] = np.concatenate((c1,c2))

        self.renumber_ids()

        self.renumber_residues()
        self.find_locGlob()

        self.merge_ff(obj,add)

        self.exclusions_map.update(obj.exclusions_map)

        return

    def renumber_ids(self,start_from=0):
        at_ids = np.arange(0,self.natoms,1,dtype=int)
        at_ids+=start_from
        self.at_ids = at_ids
        return


    def remove_atoms_ids(self,ids,reinit=False,filter_topology=True):
        filt = np.logical_not(np.isin(self.at_ids,ids))
        self.filter_system(filt,reinit,filter_topology)
        return

    def remove_atoms(self,crit,frame=0,reinit=True):
        coords = self.get_coords(frame)
        filt = crit(coords)
        filt = np.logical_not(filt)
        self.filter_system(filt,reinit)
        return

    def remove_molecules(self,criterium_function,fargs=tuple(), fkwargs=dict(), frame=0,reinit=False):
        coords = self.get_coords(frame)
        filt = criterium_function(coords, *fargs, **fkwargs)
        res_cut = self.mol_ids[filt]
        filt = ~np.isin(self.mol_ids,res_cut)
        self.filter_system(filt,reinit)
        return

    def filter_topology(self,removed_ids):

        m = self.old_to_new_ids

        for i in removed_ids:
            try:
                del self.neibs[i]
            except KeyError:
                pass

        for c in ass.numpy_keys(self.connectivity):
            if c[0] in removed_ids or c[1] in removed_ids:
                del self.connectivity[tuple(c)]
        for c in ass.numpy_keys(self.pairs):
            if c[0] in removed_ids or c[1] in removed_ids:
                del self.pairs[tuple(c)]
        for c in ass.numpy_keys(self.exclusions):
            if c[0] in removed_ids or c[1] in removed_ids:
                del self.exclusions[tuple(c)]
        for a in ass.numpy_keys(self.angles):
            if a[0] in removed_ids or a[1] in removed_ids or a[2] in removed_ids:
                del self.angles[tuple(a)]
        for d in ass.numpy_keys(self.dihedrals):
            if d[0] in removed_ids or d[1] in removed_ids or d[2] in removed_ids or d[3] in removed_ids:
                del self.dihedrals[tuple(d)]


        self.connectivity = {(m[i[0]],m[i[1]]):t for i,t in self.connectivity.items() }
        self.pairs = {(m[i[0]],m[i[1]]):t for i,t in self.pairs.items() }
        self.exclusions = {(m[i[0]],m[i[1]]):t for i,t in self.exclusions.items() }
        self.find_neibs()
        self.angles = {(m[i[0]],m[i[1]],m[i[2]]):t for i,t in self.angles.items() }
        self.dihedrals = {(m[i[0]],m[i[1]],m[i[2]],m[i[3]]):t for i,t in self.dihedrals.items()}

        return

    def filter_ff(self):
        at = self.atom_types
        bt  = self.bond_types
        angt = self.angle_types
        diht = self.dihedral_types

        tys = [at,bt,angt,diht]
        names = ['atomtypes','bondtypes','angletypes','dihedraltypes']
        for ty,name in zip(tys,names):
            ffdata = getattr(self.ff,name)
            for t in list(ffdata.keys()):
                if t not in ty:
                    del ffdata[t]

        for t in list(self.ff.nonbond_params.keys()):
            if t[0] not in self.at_types or t[1] not in self.at_types:
                del self.ff.nonbond_params[t]
        return
    def filter_system(self,filt,reinit=False,filter_topology=True):

        otn = dict()
        sub = 0

        for j,f in enumerate(filt):
            if not f:
                sub+=1
                otn[j] = None
            else:
                otn[j] = j-sub

        self.old_to_new_ids = otn


        if filter_topology:
            removed_ids = self.at_ids[~filt]
            self.filter_topology(removed_ids)



        self.at_ids = self.at_ids[filt]
        self.at_types = self.at_types[filt]
        self.mol_ids = self.mol_ids[filt]
        self.mol_names = self.mol_names[filt]
        self.atom_mass = self.atom_mass[filt]
        self.atom_charge = self.atom_charge[filt]
        self.atom_code = self.atom_code[filt]
        for frame in self.timeframes:
            self.timeframes[frame]['coords'] = self.get_coords(frame)[filt]

        self.renumber_ids()
        self.renumber_residues()
        self.find_locGlob()
        self.filter_ff()
        if reinit:
            self.topology_initialization(reinit)
        return

    def write_residues(self,res,fname='selected_residues.gro',
                       frames=(0,0),box=None,boxoff=0.4):
        with open(fname,'w') as ofile:
            fres = np.isin(self.mol_ids, res)

            for frame in self.timeframes:
                if frames[0] <= frame <= frames[1]:

                    coords = self.get_coords(frame) [fres]
                    coords -= coords.min(axis=0)
                    at_ids = self.at_ids[fres]
                    ofile.write('Made by write_residues\n')
                    ofile.write('{:6d}\n'.format(coords.shape[0]))
                    for j in range(coords.shape[0]):
                        i = at_ids[j]
                        c = coords[j]
                        ofile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'\
                        % (self.mol_ids[i],self.mol_names[i], self.at_types[i]
                           ,self.at_ids[i%100000] ,c[0],c[1] ,c[2] ))
                    if box is None:
                        box = coords.max(axis=0) - coords.min(axis=0) + boxoff
                    ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))

            ofile.closed
        return

    def apply_pbc(self):
        for frame in self.timeframes:
            box = self.get_box(frame)
            coords = self.get_coords(frame)
            self.timeframes[frame]['coords'] = implement_pbc(coords,box)
        return

    @property
    def nmolecules(self):
        return np.unique(self.mol_ids).shape[0]

    def find_args_per_residue(self,filt,attr_name):
        args = dict()
        for j in np.unique(self.mol_ids[filt]):
            x = np.array(np.where(self.mol_ids==j)[0],dtype=int)
            args[j] = x
        setattr(self,attr_name,args)
        setattr(self,'N'+attr_name, len(args))
        return



    def multiply_periodic(self,multiplicity,one_molecule=False):
        if len(multiplicity) !=3:
            raise Exception('give the multyplicity in the form (times x, times y, times z)')
        for d,mult in enumerate(multiplicity):
            if mult ==0: continue

            natoms = self.natoms
            nmols = self.nmolecules
            totm = mult+1

            if totm<2:
                raise Exception('multiplicity {} is not valid'.format(multiplicity))
            #new topology
            shape = natoms*totm
            # allocate array
            at_ids = np.empty(shape,dtype=int)
            mol_ids = np.empty(shape,dtype=int)
            at_types = np.empty(shape,dtype=object)
            mol_names = np.empty(shape,dtype=object)
            atom_code = np.empty(shape,dtype=object)
            atom_mass = np.empty(shape,dtype=float)
            atom_charge = np.empty(shape,dtype=float)
            for m in range(0,mult+1):
                na = m*natoms
                self.update_topology(na,self)
                mm = m*nmols if not one_molecule else 0
                for i in range(natoms):
                    idx = i+na
                    at_ids[idx] = idx
                    at_types[idx] =self.at_types[idx%natoms]
                    mol_ids[idx] = self.mol_ids[idx%natoms]+mm
                    mol_names[idx] = self.mol_names[idx%natoms]
                    atom_code[idx] = self.atom_code[idx%natoms]
                    atom_mass[idx] = self.atom_mass[idx%natoms]
                    atom_charge[idx] = self.atom_charge[idx%natoms]
            self.at_ids = at_ids
            self.at_types = at_types
            self.mol_ids = mol_ids
            self.mol_names = mol_names
            self.atom_code = atom_code
            self.atom_mass = atom_mass
            self.atom_charge = atom_charge


            #allocate coords

            for frame in self.timeframes:
                coords = np.empty((shape,3))
                c = self.get_coords(frame)
                box = self.get_box(frame)

                idx = 0
                coords[idx:idx+natoms] = c.copy()
                for j in range(1,mult+1):
                    L = box[d]*j
                    idx=j*natoms
                    coords[idx:idx+natoms] = c.copy()
                    coords[idx:idx+natoms,d]+=L

                self.timeframes[frame]['coords'] = coords
                self.timeframes[frame]['boxsize'][d]*=mult+1

        self.resort_by_molname()
        self.find_locGlob()
        return

    def resort_by_molname(self):
        #first mol first
        names = []
        for name in self.mol_names:
            if name not in names:
                names.append(name)

        #map_ids old to new
        mapids = dict()
        n = 0
        for name in names:
            argsn = np.where(self.mol_names==name)[0]
            for i,j in enumerate(argsn):
                mapids[j] = i  + n
            n = argsn.shape[0]
        for attrname in ['at_types','mol_names','atom_code','atom_mass','atom_charge','mol_ids']:
            attrold = getattr(self,attrname)
            attrnew = np.empty_like(attrold)
            for j,i in mapids.items():
                attrnew[i] = attrold[j]
            setattr(self,attrname,attrnew)

        self.resorting_mapids = mapids
        self.renumber_residues()

        self.connectivity = {(mapids[c[0]],mapids[c[1]]): val for c,val in self.connectivity.items()}
        self.pairs = {(mapids[c[0]],mapids[c[1]]): val for c,val in self.pairs.items()}
        self.exclusions = {(mapids[c[0]],mapids[c[1]]): val for c,val in self.exclusions.items()}
        self.angles = {(mapids[c[0]],mapids[c[1]],mapids[c[2]]): val for c,val in self.angles.items()}
        self.dihedrals = {(mapids[c[0]],mapids[c[1]],mapids[c[2]],mapids[c[3]]): val for c,val in self.dihedrals.items()}
        self.neibs = {mapids[j]:{mapids[neib] for neib in s } for j,s in self.neibs.items()}

        for frame in self.timeframes:

            cold = self.get_coords(frame)
            cnew = np.empty_like(cold)

            for j,i in mapids.items():
                cnew[i] = cold[j]
            self.timeframes[frame]['coords'] = cnew

        return

    def get_coords(self,frame):
        return self.timeframes[frame]['coords']

    def get_reference_point(self, frame, method=None):
        if method is None:
            method = self.reference_point_method

        if method == 'origin':
            return np.array([0.0, 0.0, 0.0])
        if method == 'center':
            return self.get_box(frame)/2.0
        elif (type(method) is list or type(method) is tuple) and len(method) ==3  :
            return np.array (method)

        coords = self.get_coords(frame)
        mass = self.atom_mass

        if method in list(self.mol_ids):
            f = method == self.mol_ids
        elif method in list(self.mol_names):
            f = method == self.mol_names
        elif method in list(self.at_ids):
            return coords[method]
        else:
            raise Exception(f'Cannot find reference point via method "{method}"')

        refp = CM(coords[f], mass[f])
        return refp


    def get_velocities(self,frame):
        return self.timeframes[frame]['velocities']

    def get_forces(self,frame):
        return self.timeframes[frame]['forces']

    def get_box(self,frame):
        return self.timeframes[frame]['boxsize']

    def get_time(self,frame):
        return self.timeframes[frame]['time']


    def get_whole_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.unwrap_coords(coords, box)
        return coords
    @property
    def dihedrals_per_type(self):

        tys = self.dihedral_types
        d = {t:[] for t in tys}
        for k,t in self.dihedrals.items():
            d[t].append(np.array(k))
        for t in tys:
            d[t] = np.array(d[t])
        return d
    @property
    def connectivity_per_type(self):

        tys = self.bond_types
        d = {t:[] for t in tys}
        for k,t in self.connectivity.items():
            d[t].append(np.array(k))
        for t in tys:
            d[t] = np.array(d[t])
        return d
    @property
    def angles_per_type(self):

        tys = self.angle_types
        d = {t:[] for t in tys}
        for k,t in self.angles.items():
            d[t].append(np.array(k))
        for t in tys:
            d[t] = np.array(d[t])
        return d

    def ids_from_topology(self,topol_vector):
        inter = len(topol_vector)
        if inter == 2:
            ids = self.connectivity_per_type
        elif inter == 3:
            ids = self.angles_per_type
        elif inter == 4:
            ids = self.dihedrals_per_type
        else:
            raise Exception('Large topology vectors with size >= {} are not Implemented'.format(inter))
        topol_vector = tuple(topol_vector)

        if topol_vector in ids: tp = topol_vector
        elif topol_vector[::-1] in ids: tp = topol_vector[::-1]
        else:
            raise Exception('{} not in {}'.format(topol_vector,list(ids)))

        arr0 = ids[tp][:,0] ; arr1 = ids[tp][:,-1]

        return arr0,arr1

    def ids_from_keyword(self,keyword,exclude=[]):
        if keyword == 'dihs':
            ids = self.dihedrals_per_type
        if keyword == 'angles':
            ids = self.angles_per_type
        if keyword =='bonds':
            ids = self.connectivity_per_type
        ids1 = np.empty(0,dtype=int)
        ids2 = np.empty(0,dtype=int)
        for k,i in ids.items():
            if k in exclude:
                continue
            ids1 = np.concatenate( (ids1,i[:,0]) )
            ids2 = np.concatenate( (ids2,i[:,-1]) )
        return ids1,ids2


    def ids_from_backbone(self,bonddist, filt=None):
        logger.info('getting vector ids from backbone')
        # Get the ids
        if filt is None:
            ids = self.at_ids
        else:
            ids = self.at_ids[filt]

        # identify if the SAME bond_dist_matrix was calculated before
        same_before = True
        same_before = hasattr(self,'backbone_dist_matrix')
        same_before = hasattr(self,'prev_ids_bdm')
        if same_before:
            same_before = self.prev_ids_bdm.shape == ids.shape
        if same_before:
            same_before = (self.prev_ids_bdm == ids).all()

        # get the matrix
        if same_before:
            bd = self.backbone_dist_matrix
        else:
            bd = self.find_bond_distance_matrix(ids)
            self.backbone_dist_matrix = bd
        # get the bond dist based on input
        if type(bonddist) is not int:
            if bonddist == 'max': # useful for end to end
                bonddist = int( bd.max() ) # int for safety
            elif bonddist =='mean':
                vals = bd[bd > 0]
                if vals.size == 0:
                    raise Exception('Cannot compute bonddist="mean": no entries > 1 in bond-distance matrix')
                bonddist = int( round( float( vals.mean() ) ) )
            else:
                raise Exception('Unkown option {bonddist} for bonddist variable')

        # get the ones with bonddist number of bonds between them
        b1,b2 = np.nonzero(bd == bonddist)
        ids1 = ids[b1]
        ids2 = ids[b2]

        # Remove duplicated pair entries
        u_ids1, u_ids2 = [], []
        seen = set()

        for i1, i2 in zip(ids1, ids2):
            pair = frozenset((i1, i2))
            if pair in seen:
                continue

            seen.add(pair)
            u_ids1.append(i1)
            u_ids2.append(i2)

        ids1 = np.array(u_ids1)
        ids2 = np.array(u_ids2)


        self.prev_ids_bdm = ids.copy()

        return ids1,ids2

    def ids_from_molname(self, mol_name, bonddist='max' ):
        logger.info(f'getting vector ids via mol_name and backbone bonddist option = {bonddist}')
        filt = self.mol_names == mol_name
        ids1, ids2 = self.ids_from_backbone(bonddist , filt)
        return ids1, ids2

    def ids_between_atom_types(self, t1, t2):
        i1 = self.at_ids[self.at_types == t1]
        i2 = self.at_ids[self.at_types == t2]
        ids1 = list(i1)*len(i2)
        n = len(i1)
        ids2 = []
        for j in i2:
            ids2.extend([j]*n)
        ids1, ids2 = np.array(ids1), np.array(ids2)
        filt_not_same = ids1 != ids2
        ids1 = ids1[filt_not_same]
        ids2 = ids2[filt_not_same]
        return ids1, ids2

    def find_vector_ids(self,topol_vector,exclude=[]):
        '''
        Parameters
        ----------
        topol_vector : list of atom types, or int for e.g. 1-2,1-3,1-4 vectors
            Used to find e.g. the segmental vector ids
        exclude : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        ids1 : int array of atom ids
        ids2 : int array of atom ids

        '''

        logger.info(f'Finding vector ids: topol_vector passed {topol_vector}')
        t0 = perf_counter()
        ty = type(topol_vector)
        if ty is list or ty is tuple:
            ids1,ids2 = self.ids_from_topology(topol_vector)
        if ty is int:
            ids1,ids2 = self.ids_from_backbone(int(topol_vector) - 1)
        if ty is str:
            if topol_vector in ['dihs','angles','bonds']:
                ids1,ids2 = self.ids_from_keyword(topol_vector,exclude)
            elif topol_vector in self.mol_names:
                ids1,ids2 = self.ids_from_molname(topol_vector)
            else:
                if '-' in topol_vector or ' ' in topol_vector or '_' in topol_vector:

                    if '-' in topol_vector: k ='-'
                    elif '_' in topol_vector: k='_'
                    elif ' ' in topol_vector: k = ' '

                    t1,t2 = tuple(topol_vector.split(k))

                    if t1 in self.at_types and t2 in self.at_types:
                        ids1, ids2 = self.ids_between_atom_types(t1, t2)
                    elif t1 in self.mol_names:
                        if t2 !='max' and t2 !='mean':
                            ids1, ids2 = self.ids_from_molname(t1, int(t2) - 1)
                        else:
                            ids1, ids2 = self.ids_from_molname(t1, t2)
                        #logger.info(f'ids1 = {ids1[0:5]} , ids2 = {ids2[0:5]}')
                    else:
                        raise Exception(f'topol_vec passed {topol_vec} does not correspond to molecule or atom')
                else:
                    raise Exception('could not understand string identifier find topology ids')



        logger.info(f'time to find vector list (number of vectors = {ids1.shape[0]}) --> {perf_counter()-t0:.3e}')
        return ids1,ids2


    def find_bond_distance_matrix(self,ids):
        '''
        takes an array of atom ids and finds how many bonds
        are between each atom id with the rest on the array

        Parameters
        ----------
        ids : numpy array of int

        Returns
        -------
        distmatrix : numpy array shape = (ids.shape[0],ids.shape[0])

        '''

        t0 = perf_counter()
        size = ids.shape[0]
        distmatrix = np.zeros((size,size),dtype=int)
        for j1,i1 in enumerate(ids):
            nbonds = self.bond_distance_id_to_ids(i1,ids)
            distmatrix[j1,:] = nbonds
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return distmatrix

    def bond_distance_id_to_ids(self,i,ids):
        '''
        takes an atom id and find the number of bonds between it
        and the rest of ids.
        If it is not connected the returns a very
        large number

        Parameters
        ----------
        i : int
        ids : array of int

        Returns
        -------
        nbonds : int array of same shape as ids
            number of bonds of atom i and atoms ids

        '''
        chunk = {i}
        n = ids.shape[0]
        nbonds = np.ones(n)*(-1) # array of bond distance of i  to ids (-1) for not bonded
        incr_bonds = 0
        new_neibs = np.array(list(chunk))
        while new_neibs.shape[0]!=0:

            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            nbonds[f] = incr_bonds

            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:
                    if neib not in chunk:
                        new_set.add(neib)
                        chunk.add(neib)

            new_neibs = np.array(list(new_set))
            incr_bonds+=1
        return nbonds


    def nbonds_of_ids_from_other_ids(self,ids,args):
        '''
        This function finds the minimum number of bonds
        that each atom id in ids has from args.
        If an id is not connected in any way with any of the args then
        a very large number is return within the nbonds array

        Parameters
        ----------
        ids : int array
        args : int array

        Returns
        -------
        nbonds : int array of shape as ids

        '''
        n = ids.shape[0]
        nbonds = np.ones(n)*10**10

        #old_chunk = set() ;
        new_neibs = args.copy()
        chunk = set(args)
        incr_bonds=0
        #same_set = False
        fnotin_args = np.logical_not(np.isin(ids,args))

        while  new_neibs.shape[0] !=0:

            #f = np.logical_and(np.isin(ids,new_neibs),fnotin_args)
            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            f = np.logical_and(f,fnotin_args)


            nbonds [ f ] = incr_bonds
            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:
                    if neib in chunk: continue
                    new_set.add(neib)
                    chunk.add(neib)

            new_neibs = np.array(list(new_set))
            incr_bonds += 1

        return nbonds

    def append_timeframes(self,object2):
        '''
        Used to append one trajectory to another
        !It assumes that time of the second starts at the end of the first
        Parameters
        ----------
        object2 : A second object of the class Analysis

        Returns
        -------
        None.

        '''
        tlast = self.get_time(self.nframes-1)
        nfr = self.nframes
        for i,frame in enumerate(object2.timeframes):
            self.timeframes[nfr+i] = object2.timeframes[frame]
            self.timeframes[nfr+i]['time']+=tlast
        return
    def unwrap_coords(self,coords,box):
        '''
        Do not trust this function. Works only for linear polymers
        Parameters
        ----------
        coords :
        box :

        Returns
        -------
        unc : Unrwap coordinates.

        '''
        unc = coords.copy()
        b2 =  box/2
        k0 = self.sorted_connectivity_keys[:,0]
        k1 = self.sorted_connectivity_keys[:,1]
        n = np.arange(0,k0.shape[0],1,dtype=int)
        dim = np.array([0,1,2],dtype=int)

        unc = unwrap_coords_kernel(unc, k0, k1, b2, n, dim, box)

        return unc

    def unwrap_all(self):
        for k in self.timeframes:
            coords = self.timeframes[k]['coords']
            box =self.timeframes[k]['boxsize']
            self.timeframes[k]['coords'] = self.unwrap_coords(coords, box)
        return


    def box_mean(self):
        t0 = perf_counter()
        box = np.zeros(3)
        args = (box,)
        nframes = self.loop_trajectory('box_mean', args)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return box/nframes

    def find_EndGroup_args(self,ty = None,serial=True):
        eargs = []
        if serial:
            for j,args in self.chain_args.items():
                 eargs.append(args[0]) ; eargs.append(args[-1])
            eargs = np.array(eargs)
        elif ty is not None:
            eargs = self.at_ids[self.at_types == ty]
        else:
            raise NotImplementedError('give either the type or serial=True if your atom chains are seriarly stored')
        self.EndGroup_args = eargs
        return

    def get_EndGroup_args(self):
        try:
            args = self.EndGroup_args
        except AttributeError as err:
            raise AttributeError('{}\nCall function "find_EndGroup_args" to set this attribute'.format(err))
        else:
            return args
    @staticmethod
    def element_from_type(ty):
        elements = list(maps.elements_mass.keys())
        n = len(ty)
        while ty[:n] not in elements:
            n-=1
            if n==0:
                raise ValueError('cannot find a corresponding element for type {:s}'.format(ty))
        return ty[:n]

    def match_types(self,types1,types2):

        for ctype1,ctype2 in zip(types1,types2):

            if len(ctype1) == 2:
                dictionary = self.ff.bondtypes
            elif len(ctype1) ==3:
                dictionary = self.ff.angletypes
            elif len(ctype1) == 4:
                dictionary = self.ff.dihedraltypes

            val = list(dictionary[ctype2])
            val[0] = '  '.join(ctype1)
            dictionary[ctype1] = tuple(val)
        return


    def element_based_matching(self,types):

        for ctype1 in types:
            eletype1 = tuple(self.element_from_type(c) for c in ctype1)
            if len(ctype1) == 2:
                dictionary = self.ff.bondtypes
            elif len(ctype1) ==3:
                dictionary = self.ff.angletypes
            elif len(ctype1) == 4:
                dictionary = self.ff.dihedraltypes

            for ctype2 in list(dictionary.keys()):
                eletype2 = tuple(self.element_from_type(c) for c in ctype2)
                if eletype1 == eletype2:
                    val = list(dictionary[ctype2])
                    val[0] = '  '.join(ctype1)
                    print('matching type {} to {}'.format(ctype1,ctype2))
                    dictionary[ctype1] = tuple(val)
                    break
        return



    def get_chem_per_molecule(self,mol_id):
        cad = ['connectivity','angles','dihedrals','pairs','exclusions']
        b = {c:[] for c in cad}
        at_ids = self.at_ids[self.mol_ids==mol_id]
        for c in cad:
            data = getattr(self,c)
            if len(data) ==0:
                b[c] = []
                continue
            c_ids = np.array(list(data.keys()))

            f2 = np.isin(c_ids, at_ids)
            f = np.ones(f2.shape[0],dtype=bool)
            for j in range(f2.shape[1]):
                f = np.logical_and(f,f2[:,j])
            cfi = c_ids [ f ]
            b[c] =  [tuple(j for j in cfi[i])  for i in range(cfi.shape[0])]
        return b
    @property
    def molecules(self):
        mols = dict()
        for nm in np.unique(self.mol_names):
            mols[nm] = np.unique(self.mol_ids[nm == self.mol_names]).shape[0]
        return mols

    def write_itp(self,path,Defaults=True,
                  include_pairs=[],include_exclusions=[],
                  extra_itp_lines=dict()):

        mols = self.molecules
        for k in mols:
            try:
                nexcl = self.exclusions_map[k]
            except KeyError:
                print('Exclusions not found for {:s}'.format(k))
                nexcl = 3
            #num = mols[k]
            if path !='':
                fname = '{:s}/{:s}.itp'.format(path,k)
            else:
                fname ='{:s}.itp'.format(k)

            lines = ['; generated by md_pipeline library', '','']
            lines.extend(['','[ moleculetype ]', '; molname      nrexcl'])
            lines.append('{:5s}    {:1d}'.format(k,nexcl))
            lines.extend(['','[ atoms ]',';id atype resnr  resname atname cgnr'])
            s = 1
            mol_id = self.mol_ids[k==self.mol_names][0]
            chem = self.get_chem_per_molecule(mol_id)
            globloc = self.glob_id_to_loc[mol_id]
            #ids_mol = self.at_ids[mol_id==self.mol_ids]
            for j,a in enumerate(self.at_ids[mol_id == self.mol_ids]):
                code = self.atom_code[a]
                ty = self.at_types[a]
                mid = self.mol_ids[a]
                mn = self.mol_names[a]
                ch = self.atom_charge[a]
                ms = self.atom_mass[a]
                line = '{:5d}  {:5s}  {:5d}  {:5s}  {:5s}  {:5d}  {:8.6f}  {:8.6f}'.format(j+s,code,mid,mn,ty,j+s,ch,ms)
                lines.append(line)


            lines.extend(['','[ bonds ]',';ai   aj  func'])
            for c in chem['connectivity']:
                bc =self.connectivity[c]
                try:
                    if bc not in self.ff.bondtypes:
                        bcc = tuple(list(bc)[::-1])
                        func = self.ff.bondtypes[bcc][1]
                    else:
                        func = self.ff.bondtypes[bc][1]
                except IndexError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find func for bondtype {:}'.format( bc))
                except KeyError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find bondtype {:}'.format( bc))
                i1,i2 = globloc[c[0]], globloc[c[1]]
                lines.append('{:5d}  {:5d}   {:}'.format(i1+s,i2+s,func))


            lines.extend(['','[ angles ]',';ai   aj   ak func'])
            for c in chem['angles']:
                ac = self.angles[c]
                try:
                    if ac not in self.ff.angletypes:
                        acc = tuple(list(ac)[::-1])
                        func = self.ff.angletypes[acc][1]
                    else:
                        func = self.ff.angletypes[ac][1]
                except IndexError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find func for angletype {:}'.format( ac))
                except KeyError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find angletype {:}'.format( ac))

                i1,i2,i3 = globloc[c[0]], globloc[c[1]], globloc[c[2]]
                lines.append('{:5d}  {:5d}  {:5d}   {:}'.format(i1+s,i2+s,i3+s,func))


            lines.extend(['','[ dihedrals ]','; improper ai   aj   ak   al  func'])
            lines.extend(['','[ dihedrals ]','; proper ai   aj   ak   al  func'])
            for c in chem['dihedrals']:

                dc = self.dihedrals[c]
                try:
                    if dc not in self.ff.dihedraltypes:
                        key = tuple(list(dc)[::-1])
                    else:
                        key = dc
                    if type(self.ff.dihedraltypes[key]) is list:
                        func = self.ff.dihedraltypes[key][0][1]
                    else:
                        func = self.ff.dihedraltypes[key][1]
                except IndexError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find func for dihedraltype {:}'.format( dc))
                except KeyError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find dihedraltype {:}'.format( dc))

                i1,i2,i3,i4 = globloc[c[0]], globloc[c[1]], globloc[c[2]], globloc[c[3]]
                lines.append('{:5d}  {:5d}  {:5d}  {:5d}    {:}'.format(i1+s,i2+s,i3+s,i4+s,func))
            lines.append('')

            lines.extend(['','[ pairs ]',';ai   aj  func'])
            for c in chem['pairs']:
                func = ''
                i1,i2 = globloc[c[0]], globloc[c[1]]
                lines.append('{:5d}  {:5d}   {:}'.format(i1+s,i2+s,func))



            lines.extend(['','[ exclusions ]',';ai   aj  '])
            for c in chem['exclusions']:
                func = ''
                i1,i2 = globloc[c[0]], globloc[c[1]]
                lines.append('{:5d}  {:5d}   {:}'.format(i1+s,i2+s,func))

            lines.append('')
            for p in include_pairs:
                pairs1,pairs2 = self.find_vector_ids(p)
                lines.extend(['','[ pairs ]','; i  j '])
                for p1,p2 in zip(pairs1,pairs2):
                    lines.append('{:5d}   {:5d}    1'.format(p1+s,p2+s))
            for e in include_exclusions:
                pairs1,pairs2 = self.find_vector_ids(e)
                lines.extend(['','[ exclusions ]','; i  j '])
                for p1,p2 in zip(pairs1,pairs2):
                    lines.append('{:5d}   {:5d}  '.format(p1+s,p2+s))

            if hasattr(self.ff,'posres'):
                data = getattr(self.ff,'posres')

                for d in data:
                    filt = getattr(self,d['by']) == d['val']
                    fmol = self.mol_names == k
                    f = np.logical_and(filt,fmol)
                    if f.any():
                        lines.extend(['','[ position_restraints ]','; i  j '])
                        for atid in self.at_ids[f]:
                            i = self.glob_id_to_loc[mol_id][atid]
                            #r = d['r']
                            k = d['k']
                            lines.append('{:d}  {:d}   {:8.5f}  {:8.5f}  {:8.5f}'.format(i+s,1,k,k,k))
            if k in extra_itp_lines:
                lines.extend(extra_itp_lines[k])
            with open(fname,'w') as f:
                for line in lines:
                    f.write('{:s}\n'.format(line))
                f.close()
        return

    @property
    def system_name(self):
        name = []
        for m in self.sorted_mol_names:
            j = self.molecules[m]
            name.append(m+'_'+str(j))

        return '-'.join(name)
    @property
    def sorted_mol_names(self):
        sorted_names = []
        for m in self.mol_names:
            if m not in sorted_names:
                sorted_names.append(m)
        return sorted_names
    def write_topfile(self,fname,nbfunc='1',combrule='2', extra_itp_lines=dict(),
                      genpairs='no',defaults=True,includes = [],
                      opls_convection=True,convection='opls',include_pairs=[],include_exclusions=[]):
        if opls_convection or convection =='opls':
            fudgeLJ='0.5'
            fudgeQQ='0.5'
            combrule='3'
            genpairs='yes'
        elif convection =='AMBER':
            fudgeLJ='0.5'
            fudgeQQ='0.83333'
        elif convection =='full':
            fudgeLJ, fudgeQQ = '1.0', '1.0'

        if fname[-4:]!='.top': fname += '.top'
        lines = ['; generated by md_pipeline library','','']


        lines.extend(['','','[ defaults ]',
                      '; nbfunc       comb-rule      gen-pairs      fudgeLJ   fudgeQQ'])
        lines.append('            '.join([str(nbfunc),str(combrule),str(genpairs),str(fudgeLJ),str(fudgeQQ)]))


        lines.extend(['','[ atomtypes ]'])
        lines.append('; type   name    mass    charge  ptype   sig     eps')
        def jt(k):
            return '  '.join([str(i) for i in k])

        for k,v in self.ff.atomtypes.items():
            vv = v[-5:]
            lines.append('{:10s} {:5s}  {:9.6f}  {:9.6f}  {:5s}  {:9.6f} {:9.6f}'.format(
                v[0],k,float(vv[0]),float(vv[1]),vv[2],float(vv[3]),float(vv[4])))

        lines.extend(['','[ nonbond_params ]',';  i     j   func    sigma(c12)    epsilon(c6)'])
        for k,v in self.ff.nonbond_params.items():
            lines.append(f'{v[0]:10s} {int(v[1]):3d} {float(v[2]):9.6f} {float(v[3]):9.6f}')

        lines.extend(['','[ bondtypes ]',';  i     j   func    b0    kb'])
        for k,v in self.ff.bondtypes.items():
            try:
                v[1]
                v[2]
                v[3]
            except IndexError:
                if not defaults:
                    raise Exception('I dont know the force field parameters of bondtype {}'.format(k))
            else:

                lines.append('{:5s}  {:5s}  {:1d}  {:9.6f}  {:9.6f}'.format(*k,int(v[1]),float(v[2]),float(v[3])))
        lines.extend(['','[ angletypes ]','; i    j    k func       th0         cth'])
        for k,v in self.ff.angletypes.items():
            try:
                v[1]
                v[2]
                v[3]
            except IndexError:
                if not defaults:
                    raise Exception('I dont know the force field parameters of angletype {}'.format(k))
            else:
                lines.append('{:5s}  {:5s} {:5s} {:1d}  {:9.6f}  {:9.6f}'.format(*k,int(v[1]),float(v[2]),float(v[3])))

        lines.extend(['','[ dihedraltypes ]','; i   j  k   l'])
        for k,v in self.ff.dihedraltypes.items():
            if type(v) is not list:
                l = '   '.join(v[1:])
                lines.append('{:5s}  {:5s}  {:5s}  {:5s}  {:s}'.format(*k,l))
            else:
                for vv in v:
                    l = '   '.join(vv[1:])
                    lines.append('{:5s}  {:5s}  {:5s}  {:5s}  {:s}'.format(*k,l))
        lines.extend(['','',''])

        for k in includes:
            k1 = k + '.itp' if k[-4:] !='.itp' else k
            lines.append('#include "{:s}"'.format(k1))

        for k in self.molecules:
            lines.append('#include "{:s}.itp"'.format(k))
        if '/' in fname:
            path = '/'.join(fname.split('/')[:-1])
        else:
            path = ''

        self.write_itp(path,Defaults=defaults,
                       include_pairs=include_pairs,
                       include_exclusions=include_exclusions,
                       extra_itp_lines=extra_itp_lines)

        lines.extend(['','',''])


        lines.extend(['','[ system ]',self.system_name])

        lines.extend(['','[ molecules ]',';molecule name number'])
        for k in self.sorted_mol_names:
            v = self.molecules[k]
            lines.append('   '.join([k,str(v)]) )

        with open(fname,'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.close()
        return

    def clean_dihedrals_from_topol_based_on_ff(self):
        nonex = self.nonexisting_types('dihedral')['inff']
        new_dihs = dict()
        for k,v in self.dihedrals.items():
            if v not in nonex:
                new_dihs[k] = v
        self.dihedrals=new_dihs
        return

    def nonexisting_types(self,which):
        listcheck = ['atom','bond','angle' ,'dihedral']
        if which not in listcheck:
            raise ValueError('give one of the following {}'.format(listcheck))

        attr = getattr(self.ff,which+'types')
        data = getattr(self, which+'_types')
        nonex_inff = [k for k in data if k not in attr]
        nonex = [k for k in attr if k not in data]
        re = {'inff': nonex_inff,'intopol':nonex}
        return re

    def write_potential_inc (self,fname, to_kcal =True, to_Angstrom=True,
                             pair_style='lj/cut/coul/long', rcut=10, sp=0.5,
                             bond_style='harmonic', angle_style='harmonic'):
        types_map, btypes_map, atypes_map, dtypes_map = self.get_lammps_types()

        ce = 1/4.178 if to_kcal else 1
        ca = 10.0 if to_Angstrom else 1


        with open(fname, 'w') as f:
            f.write(f'pair_style {pair_style} {rcut:3.2f}\n')
            for a, x in self.ff.atomtypes.items():
                p1 = str( round( float(x[-1]) *ce, 6) )
                p2 = str( round( float(x[-2]) *ca, 6) )
                f.write(f'pair_coeff {types_map[x[0]]} {types_map[x[1]]} {p1} {p2} # {x[0]} {x[1]} \n')
            f.write('pair_modify mix geometric\n')
            f.write('pair_modify tail yes\n')

            f.write('\n')

            f.write(f'special_bonds lj/coul 0.0 0.0 {sp :3.2f}\n')
            f.write('kspace_style pppm 1e-4\n')

            f.write('\n')

            f.write(f'bond_style {bond_style}\n' )
            for a, x in self.ff.bondtypes.items():
                p1 = str( round( float(x[-1]) *ce/ca**2, 6) )
                p2 = str( round( float(x[-2]) *ca, 6) )
                f.write(f'bond_coeff {btypes_map[a]:2d}  {p1:12s} {p2:12s} # {x[0]} \n')

            f.write('\n')

            f.write(f'angle_style {angle_style}\n' )
            for a, x in self.ff.angletypes.items():
                p1 = str( round( float(x[-1]) *ce, 6) )
                p2 = str( round( float(x[-2]), 6) )
                f.write(f'angle_coeff {atypes_map[a]:2d}  {p1:12s} {p2:12s} # {x[0]} \n')

            dih_funcs = np.unique([v[1] for v in self.ff.dihedraltypes.values() ] )
            dih_func_map = {'5': 'opls', '4':'opls', '3':'multi/harmonic'}
            styles_found = ' '.join( np.unique([dih_func_map[x]  for x in dih_funcs ]) )

            f.write('\n')

            f.write(f'dihedral_style hybrid {styles_found}\n' )

            for a, x in self.ff.dihedraltypes.items():

                func = x[1]
                style = dih_func_map[func]
                coe = np.array([float(c) for c in x[2:]])

                if style =='opls' and func =='5':
                    coe *= ce
                    coeffs = ' '.join( [f'{c:4.5f}' for c in coe ])
                elif style =='multi/harmonic':
                    coe *= ce
                    coeffs = ' '.join( [f'{c:4.5f}' for c in coe[ : 5] ])
                    if coe.shape[0] > 5:
                        for c in coe[5: ]:
                            if c != 0.0:
                                raise Exception(f'Dihedral {a} cannot be translated via multi/harmonic style')
                elif style =='opls' and func=='4':

                    p1 = round(float(x[3])*ce*2, 5)
                    mult = int(x[4])
                    ang =  int(float (x[2]) )
                    sn = 1
                    if mult in [2, 4]:
                        sn*=-1
                    if ang == 180:
                        sn*=-1
                    elif ang == 0:
                        pass
                    else:
                        raise Exception(f'Dihedral {a} cannot be translated to oplsa type')
                    coe = np.zeros(4)
                    coe[mult-1] = sn*p1
                    coeffs = ' '.join( [f'{c:4.5f}' for c in coe ])

                f.write(f'dihedral_coeff  {dtypes_map[a]:4d}  {style:15s}  {coeffs:70s} # {x[0]} \n')


            f.closed
            return
    def get_lammps_types(self):
        unty = np.unique(self.at_types)

        types_map = {t: n+1 for n,t in enumerate(unty)}
        btypes_map = {bt:j+1 for j,bt in enumerate(self.ff.bondtypes)}
        atypes_map = {at:j+1 for j,at in enumerate(self.ff.angletypes)}
        dtypes_map = {at:j+1 for j,at in enumerate(self.ff.dihedraltypes)}

        return types_map, btypes_map, atypes_map, dtypes_map

    def write_Lammps_dat(self, fname, frame=0, to_Angstrom=True):
        if to_Angstrom:
            length_scaling = 10
        box = self.get_box(0) * length_scaling # to Angstrom
        coords = self.get_coords(0) * length_scaling # to Angstrom
        ty = self.at_types
        unty = np.unique(self.at_types)
        natoms = len(ty)
        types_map, btypes_map, atypes_map, dtypes_map = self.get_lammps_types()

        mass_map = {t: self.atom_mass[ np.where(self.at_types == t)[0][0] ] for t in unty}

        if natoms != coords.shape[0]:
            raise ValueError('number of atoms and coordinates do not much')

        with open(fname,'w') as f:
            f.write('# generated by md_pipeline library by Nikolaos Patsalidis, units = real \n\n')

            f.write(f'{natoms} atoms\n')
            f.write(f'{self.ff.natomtypes} atom types\n')
            f.write(f'{self.nbonds} bonds\n')
            f.write(f'{self.ff.nbondtypes} bond types\n')
            f.write(f'{self.nangles} angles\n')
            f.write(f'{self.ff.nangletypes} angle types\n')
            f.write(f'{self.ndihedrals} dihedrals\n')
            f.write(f'{self.ff.ndihedraltypes} dihedral types\n')

            #f.write('\n')
            for b,s in zip(box,['x','y','z']):
                f.write('\n{:4.2f} {:4.2f} {:s}lo {:s}hi'.format(0.0,b,s,s) )

            f.write('\n\n\n\nMasses\n\n')
            for t in unty:
                mt=types_map[t]
                mass = mass_map[t]
                f.write('{} {:4.6f}\n'.format(mt,mass) )

            f.write('\n\nAtoms\n\n')
            ch = self.atom_charge
            mol_ids = self.mol_ids

            for iat,(t,c) in enumerate(zip(ty,coords)):
                tmd =types_map[t]
                charge =ch[iat]
                mol_id = mol_ids[iat]
                f.write('{:d}  {:d}  {:d}  {:4.6f}   {:4.6f}   {:4.6f}   {:4.6f}   \n'.format(iat+1,mol_id,tmd,charge,*c)  )

            f.write('\nBonds\n\n')

            for jb, (bid, ty) in enumerate(self.connectivity.items()):
                t = btypes_map[ty]
                f.write(f'{jb+1:d} {t:d} {bid[0] + 1: d} {bid[1]+1: d}\n')

            f.write('\nAngles\n\n')

            for ja, (aid, ty) in enumerate(self.angles.items()):
                t = atypes_map[ty]
                f.write(f'{ja+1:d} {t:d} {aid[0] + 1: d} {aid[1]+1: d} {aid[2]+1: d}\n')

            f.write('\nDihedrals\n\n')

            for jd, (did, ty) in enumerate(self.dihedrals.items()):
                t = dtypes_map[ty]
                f.write(f'{jd+1:d} {t:d} {did[0] + 1: d} {did[1]+1: d} {did[2]+1: d} {did[3]+1: d}\n')

            f.closed
        return

class Analysis(Topology):
    """Base analysis class for unconfined polymer / molecular systems.

    This class extends :class:`Topology` with routines for reading
    trajectories, initialising connectivity/angle/dihedral information
    from topology files, and looping over frames to compute structural
    and dynamical properties.

    Parameters
    ----------
    topol_file : str
        Path to the topology / coordinate file (e.g. GROMACS .gro
        or custom .dat) used to initialise atom-level information.
    connectivity_info : str or list
        Connectivity description, typically one or more GROMACS
        .itp files or an equivalent .dat file (for LAMMPS). Bonds from
        these files are used to infer angles and dihedrals.
    memory_demanding : bool, optional
        If ``True``, frames are not stored in memory during
        analysis but read from disk on demand.
    **kwargs
        Additional options controlling how the topology and
        trajectories are interpreted. Common keys include:

        * ``types_from_itp`` (bool): if ``True`` and atom types
          differ between ``topol_file`` and ``connectivity_info``,
          the types from the ``.itp`` files are kept.
        * ``fftop`` (str): optional path to a force-field topology
          file used to supplement parameters.
        * ``key_method`` (str): name of the method (without the
          ``"get_"`` / ``"key"`` suffix) used to generate frame
          keys, e.g. ``"time"`` for :meth:`get_timekey`.
        * ``round_dec`` (int): number of decimal digits used when
          rounding time/extension keys.

    Notes
    -----
    The analysis object is initialized with a topology file and
    connectivity information. The topology file defines the initial
    configuration, while the connectivity information is used to
    infer angles and dihedrals. Additional options can be passed to
    control the interpretation of the topology and trajectories.
    """

    def __init__(self,
                 topol_file, # gro/trr/dat for now
                 connectivity_info, #itp #dat,
                 reference_point_method= 'origin',
                 memory_demanding=False,
                 **kwargs):
        """Construct an analysis object from topology and connectivity.

        Parameters
        ----------
        topol_file : str
            Coordinate / topology file that defines the initial
            configuration (e.g. .gro (GROMACS), or .dat (LAMMPS)).
        connectivity_info : str or list
            Connectivity description, typically one or more GROMACS
            .itp files or an equivalent .dat file. Bonds from
            these files are used to infer angles and dihedrals.
        reference_point_method,
            Sets the reference_point for relative distance calculations used to identify different populations.
                Options:
                - 'origin' (str): gets the origin 0, 0 , 0
                - 'center' (str): sets the box center as reference
                -  'molecule name' (str): sets the molecule(s)' center of mass as the reference point
                - molecule id (int): sets the molecule's with that id center of mass as the reference
                - atom id (int): sets the atom with that id as the reference
        memory_demanding : bool, optional
            If ``True``, frames are not stored in memory during
            analysis but read from disk on demand.
        **kwargs
            Additional options controlling how the topology and
            trajectories are interpreted. Common keys include:

            * ``types_from_itp`` (bool): if ``True`` and atom types
              differ between ``topol_file`` and ``connectivity_info``,
              the types from the ``.itp`` files are kept.
            * ``fftop`` (str): optional path to a force-field topology
              file used to supplement parameters.
            * ``key_method`` (str): name of the method (without the
              ``"get_"`` / ``"key"`` suffix) used to generate frame
              keys, e.g. ``"time"`` for :meth:`get_timekey`.
            * ``round_dec`` (int): number of decimal digits used when
              rounding time/extension keys.
        """
        #t0 = perf_counter()
        self.topol_file = topol_file
        self.connectivity_info = connectivity_info
        self.connectivity_file = connectivity_info
        self.kwargs = kwargs
        self.memory_demanding = memory_demanding
        self.reference_point_method = reference_point_method

        self.timeframes = dict()

        if 'types_from_itp' in kwargs:
            self.types_from_itp = kwargs['types_from_itp']
        else:
            self.types_from_itp = True

        if 'fftop' in kwargs:
            self.fftop = kwargs['fftop']

        if 'key_method' not in kwargs:
            self.key_method = 'get_timekey'
        else:
            self.key_method = 'get_' + kwargs['key_method'] + 'key'

        if 'round_dec' not in kwargs:
            self.round_dec = 7
        else:
            self.round_dec = kwargs['round_dec']
        False_defaults = ['refine_dihedrals','refine_angles','bytype']
        for nm in False_defaults:
            if nm not in kwargs:
                setattr(self,nm,False)
            else:
                if type(kwargs[nm]) is not bool:
                    raise ValueError('{:s} must be either True or False (Default)'.format(nm))
                setattr(self,nm,kwargs[nm])

        self.read_topology()
        self.topology_initialization()

        if self.types_from_itp and hasattr(self,'molecule_map') and self.topol_file[-4:] =='.gro':
            self.correct_types_from_itp()

        self.timeframes = dict() # we will store the coordinates,box,step and time here

        #tf = perf_counter()-t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)

        return



    def topology_initialization(self,reinit=False):
        """Derive connectivity, angles and dihedrals and cache helpers.

        When called for the first time this method:

        * builds neighbour lists,
        * finds all angles and dihedrals implied by the connectivity,
        * constructs sorted connectivity arrays for efficient lookups,
        * identifies per-residue and per-chain atom index ranges.

        If ``reinit`` is ``True`` the connectivity-related attributes
        are reset first before recomputing.
        """
        t0 = perf_counter()
        ## We want masses into numpy array
        if reinit:
            self.connectivity = dict()
            self.neibs = dict()
            self.angles = dict()
            self.dihedrals = dict()
            self.find_locGlob()

            # Build connectivity graph and derived angle/dihedral lists
        if reinit:
            self.find_neibs()
            self.find_angles()
            self.find_dihedrals()
        #Find the ids (numpy compatible) of each type and store them
        else:
            self.find_neibs()


        self.dict_to_sorted_numpy('connectivity') # sort connectivity for vectorized coreFunctions

        self.find_args_per_residue(np.ones(self.mol_ids.shape[0],dtype=bool),'molecule_args')
        self.find_args_per_residue(np.ones(self.mol_ids.shape[0],dtype=bool),'chain_args')
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)

        return


    def box_variance(self):
        """Estimate variance of the simulation box dimensions.

        The mean box lengths are first obtained from :meth:`box_mean`
        and then the per-frame squared deviations are accumulated via
        the ``"box_var"`` core function.

        Returns
        -------
        ndarray, shape (3,)
            Variances of the box lengths along ``x``, ``y`` and ``z``.
        """
        t0 = perf_counter()
        box_var = np.zeros(3)
        box_mean = self.box_mean()
        args = (box_var,box_mean**2)
        nframes = self.loop_trajectory('box_var', args)  # accumulate squared deviations over all frames
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return box_var/nframes

    def frame_closer_tobox(self,target_box):
        """Find the frame whose box is closest to ``target_box``.

        Parameters
        ----------
        target_box : array_like, shape (3,)
            Target box lengths used as reference.

        Returns
        -------
        int
            Index of the frame whose box has the smallest Euclidean
            distance to ``target_box``.
        """
        mind = 10**9  # running minimum distance to target_box
        with self.traj_opener(*self.traj_opener_args) as ofile:
            t0 = perf_counter()
            nframes=0
            while(self.read_from_disk_or_mem(ofile, nframes)):  # sequentially stream frames
                box=self.timeframes[nframes]['boxsize']
                d = np.dot(target_box-box,target_box -box)**0.5
                if d < mind:
                    mind = d
                    frame_min = nframes
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes+=1
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return frame_min
    def loop_trajectory(self,fun,args):
        """Loop over all frames and call a core function on each.

        This is a generic wrapper around the time-integration logic.
        A function from :class:`coreFunctions` is looked up by name and
        called for each frame with ``self`` and ``args`` as arguments.

        Depending on ``memory_demanding`` the frames are either read
        from disk on the fly or taken from ``self.timeframes``.

        Parameters
        ----------
        fun : str
            Name of the function in :class:`coreFunctions` to call.
        args : tuple
            Positional arguments forwarded to the core function.

        Returns
        -------
        int
            Number of frames processed.
        """

        funtocall = getattr(coreFunctions,fun)  # resolve core kernel once by name

        if len(self.timeframes) == 0 or self.memory_demanding:
            with self.traj_opener(*self.traj_opener_args) as ofile:
                nframes=0
                while(self.read_from_disk_or_mem(ofile,nframes)):  # stream from disk frame by frame
                    self.current_frame = nframes
                    funtocall(self,*args)
                    if self.memory_demanding:
                        del self.timeframes[nframes]
                    nframes+=1
        else:
            nframes = self.loop_timeframes(funtocall,args)  # operate on in-memory cache
        del self.current_frame
        return nframes

    def loop_timeframes(self,funtocall,args):
        """Apply a function to all frames stored in memory.

        Parameters
        ----------
        funtocall : callable
            Function taking ``(self, *args)``.
        args : tuple
            Arguments forwarded to ``funtocall``.
        """
        for frame in self.timeframes:
            self.current_frame = frame
            funtocall(self,*args)
        nframes = len(self.timeframes)
        return nframes

    @property
    def first_frame(self):
        """Index of the first frame available for analysis."""
        if not self.memory_demanding:
            return list(self.timeframes.keys())[0]
        else:
            return 0

    def cut_timeframes(self,num_start=None,num_end=None):
        """Restrict the in-memory trajectory to a frame window.

        Parameters
        ----------
        num_start : int or None, optional
            First frame index to keep. If ``None`` the beginning of the
            trajectory is used.
        num_end : int or None, optional
            Last frame index to keep (exclusive). If ``None`` all
            remaining frames are kept.
        """
        if num_start is None and num_end is None:
            raise Exception('Give either a number to cut from the start or from the end for the timeframes dictionary')
        if num_start is not None:
            i1 = num_start
        else:
            i1 =0
        if num_end is not None:
            i2 = num_end
        else:
            i2 = len(self.timeframes)
        new_dict = ass.dict_slice(self.timeframes,i1,i2)
        if len(new_dict) ==0:
            raise Exception('Oh dear you have cut all your timeframes from memory')

        self.timeframes = new_dict
        return

    def get_filters_info(self,filters,relative_coords=None,
        ids1= None, ids2=None, segmental_ids=None, mol_name=None):

        """Build the ``additional_info`` dictionary consumed by :class:`Filters`.

        This helper centralizes all per-frame quantities needed by the filter
        functions (e.g. relative coordinates, segment definitions).

        Parameters
        ----------
        filters : dict
            Filter specification dict passed to :func:`Filter_Operations.calc_filters`.
            Only the keys present in this dict are used to decide what information
            to compute and include.
        relative_coords : ndarray, optional
            Precomputed relative coordinates to use for spatial filters.
            Shape must be ``(N, 3)``.
        ids1, ids2 : array_like of int, optional
            Vector end-point atom indices.
            If provided and spatial filters are requested, the mid-points are used
            to build ``relative_coords``.
        segmental_ids : array_like, optional
            Segment definitions, typically shape ``(n_segments, seg_size)``.
            If needed and not provided, they are resolved via:

            - cached ``self.stored_segmental_ids`` (if available), else
            - ``self.find_segmental_ids(ids1, ids2)`` when ``ids1``/``ids2`` are given, else
            - ``self.find_general_ids(mol_name)`` when ``mol_name`` is given.
        mol_name : str, optional
            Molecule name used as a fallback to define segments when needed.

        Returns
        -------
        dict
            Dictionary passed as ``additional_info`` into filter functions.

            Possible keys:

            - ``"relative_coords"``: (N, 3) float array
            - ``"ids1"``, ``"ids2"``: (N,) int arrays
            - ``"segmental_ids"``: segment definitions
            - ``"obj"``: ``self`` (analysis object)
        """

        def get_segmental_ids():
            nonlocal segmental_ids
            if segmental_ids is None:
                if hasattr(self, 'stored_segmental_ids'):
                    segmental_ids = self.stored_segmental_ids
                else:
                    if ids1 is not None and ids2 is not None:
                        segmental_ids = self.find_segmental_ids(ids1, ids2)
                    elif mol_name is not None:
                        segmental_ids = self.find_general_ids(mol_name)
                    else:
                        raise Exception('filters cannot be computed:\n Segments are not provided and cannot be defined, neither found stored\n--> Check Code Implementation')
                    self.stored_segmental_ids = segmental_ids
            return segmental_ids

        filters_info = dict()

        if 'x' in filters or 'y' in filters or 'z' in filters or 'space' in filters:

            coords = self.get_coords(self.current_frame)
            rf = self.get_reference_point(self.current_frame)

            if relative_coords is not None:
                relc = relative_coords
            elif ids1 is not None and ids2 is not None:
                relc = 0.5*(coords[ids1] + coords[ids2]) -rf # mid-point treatment for vectors
            else:
                segmental_ids = get_segmental_ids()
                relc = self.segs_CM(coords, segmental_ids) - rf
            filters_info['relative_coords'] = relc

        if ('conformations' in filters) or ('bonds_to_train' in filters) or ('bonds_to_non_train' in filters):
            filters_info['ids1'] = ids1
            filters_info['ids2'] = ids2
            filters_info['obj'] = self

        if 'adsorption' in filters:

            segmental_ids = get_segmental_ids()

            filters_info['segmental_ids'] = segmental_ids
            filters_info['obj'] = self

        return filters_info

    def calc_atomic_coordination(self,maxdist,type1,type2):
        """Compute atomic coordination numbers between two atom types.

        For each atom of ``type1`` the number of neighbours of
        ``type2`` within ``maxdist`` is counted for every frame and
        then averaged over the trajectory.

        Parameters
        ----------
        maxdist : float
            Cutoff distance defining a neighbour.
        type1, type2 : str or iterable of str
            Atom types that define the central atoms (``type1``) and
            neighbours (``type2``).

        Returns
        -------
        dict
            Single-key dictionary mapping ``"type1-type2"`` to a
            1D NumPy array of average coordination numbers for each
            ``type1`` atom.
        """
        t0 = perf_counter()
        def find_args_of_type(types):  # resolve atom indices for one or more types
            if not ass.iterable(types):
                types = [types]
            ids = np.array([],dtype=int)
            for ty in types:
                a1 = np.where([self.at_types==ty])[1]
                ids = np.concatenate((ids,a1))
            ids = np.unique(ids)
            return ids

        args1 = find_args_of_type(type1)
        args2 = find_args_of_type(type2)


        coordination = np.zeros(args1.shape[0])
        args = (maxdist,args1,args2,coordination)

        nframes = self.loop_trajectory('atomic_coordination',args)  # accumulate neighbours per frame

        coordination/=nframes  # time-average coordination per central atom
        def tkey(ty):
            if ass.iterable(ty):
                k = '_'.join(ty)
            else:
                k = ty
            return k

        key ='-'.join([tkey(type1),tkey(type2)])

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return {key:coordination}

    def find_q(self,dmin,dq,dmax,direction):
        """Generate a set of :math:`q` vectors for structure-factor calculations.

        Parameters
        ----------
        dmin, dmax : float
            Minimum and maximum magnitude of :math:`q`.
        dq : float
            Increment in :math:`|q|`.
        direction : array_like or None
            If ``None`` an isotropic distribution of directions is
            approximated; otherwise a 3D vector that defines a single
            preferred direction for :math:`q`.

        Returns
        -------
        ndarray
            Array of shape ``(n_q, 3)`` containing the :math:`q`
            vectors used in :meth:`calc_Sq`.
        """
        qmag = np.arange(dmin,dmax+dq,dq)  # magnitudes of q

        if direction is None:
            # Approximate isotropic sampling by distributing magnitude equally on all axes
            q = np.array([np.ones(3)*qm**(1/3) for qm in qmag] )
        else:
            if len(direction)!=3:
                raise Exception('Wrong direction vector')
            d = np.array([di for di in direction],dtype=float)
            dm = np.sum(d*d)**0.5
            d/=dm  # normalise direction to unit vector
            q = np.array([d*qm for qm in qmag])  # scale unit direction by each |q|

        return q

    def calc_Sq(self,qmin,dq,qmax,direction=None,ids=None):
        """Compute the static structure factor :math:`S(q)` from coordinates.

        Parameters
        ----------
        qmin, qmax : float
            Minimum and maximum magnitude of the scattering vector.
        dq : float
            Increment in :math:`|q|`.
        direction : array-like or None, optional
            If ``None`` an isotropic distribution of :math:`q`
            vectors is used; otherwise a 3D direction vector defining
            the orientation of :math:`q`.
        ids : ndarray or None, optional
            Optional subset of atom indices to include. For
            :class:`Analysis_Confined` the default is the polymer atoms.

        Returns
        -------
        dict
            Dictionary with keys ``"q"`` (array of q-vectors) and
            ``"Sq"`` (ctsponding structure factor values).
        """
        t0 = perf_counter()
        if isinstance(self,Analysis_Confined):
            ids = self.polymer_ids
        q = self.find_q(qmin,dq,qmax,direction)  # construct q-vectors grid
        Sq = np.zeros(q.shape[0],dtype=float)  # accumulator over frames
        args =(q,Sq,ids)
        nframes = self.loop_trajectory('Sq',args)  # call core kernel for each frame
        Sq/=nframes  # normalize by number of frames
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return {'q':q,'Sq':1+Sq}

    def calc_Sq_byInverseGr(self,dr,dmax,dq,qmax,
                            qmin=0,
                            ids=None,direction=None,intra=False,inter=False):
        """Compute :math:`S(q)` via Fourier transform of :math:`g(r)`.

        The pair distribution :math:`g(r)` is first obtained from
        :meth:`calc_pair_distribution` and then transformed to
        :math:`S(q)` using the appropriate 1D/2D/3D Fourier kernel
        depending on ``direction``.

        Parameters
        ----------
        dr : float
            Bin width in real space.
        dmax : float
            Maximum real-space distance.
        dq : float
            Increment in :math:`|q|`.
        qmax : float
            Maximum :math:`|q|` value.
        qmin : float, optional
            Minimum :math:`|q|` value, default is 0.
        ids : unused
            Present for API consistency; selection is handled via
            ``type1``, ``type2`` and ``intra``/``inter`` in
            :meth:`calc_pair_distribution`.
        direction : {None, 'xy', 'z'}, optional
            When ``None`` a full 3D transformation is used. ``'xy'``
            uses a 2D kernel appropriate for planar averaging and
            ``'z'`` performs a 1D transform.
        intra, inter : bool, optional
            Passed through to :meth:`calc_pair_distribution` to select
            intra- or inter-molecular pairs.

        Returns
        -------
        dict
            The dictionary returned by :meth:`calc_pair_distribution`
            extended with keys ``"Sq"`` and ``"q"``.
        """
        def Fourier3D(q,r,gr,rho):  # isotropic 3D transform of g(r)

            Sq = np.zeros(q.shape[0])
            for i in range(q.shape[0]):
                Sq[i] = simpson((gr-1)*r*np.sin(q[i]*r)/q[i],r)
            return 1+4*np.pi*rho*Sq
        def Fourier2D(q,r,gr,rho):

            Sq = np.zeros(q.shape[0])
            for i in range(q.shape[0]):
                Sq[i] = simpson((gr-1)*np.sin(q[i]*r)/q[i],r)
            return 1+np.pi*rho*Sq
        def Fourier1D(q,r,gr,rho):

            Sq = np.zeros(q.shape[0])
            for i in range(q.shape[0]):
                Sq[i] = simpson((gr-1)*np.sin(q[i]*r)/(q[i]*r),r)
            return 1+rho*Sq

        # Step 1: obtain g(r) and density from the pair-distribution routine
        res = self.calc_pair_distribution(dr,dmax,None,None,intra,inter)

        q =np.arange(qmin+dq,qmax+dq,dq)
        d = res['d']
        g = res['gr']
        rho = res['rho']
        box = self.box_mean()
        # Step 2: choose appropriate Fourier kernel based on requested projection
        if direction is None or direction =='':
            Sq = Fourier3D(q,d,g,rho)
        elif direction =='xy':
            Sq = Fourier2D(q,d,g,rho*box[2])
        elif direction =='z':
            Sq = Fourier1D(q,d,g,rho*box[1]*box[0])

        res['Sq'] = Sq
        res['q'] = q

        return res

    def calc_internal_distance(self,n,filters=dict()):
        """Distance between atoms separated by ``n`` bonds along a chain.

        Parameters
        ----------
        n : int
            Chemical separation along the chain (1, 2, 3, ...).
        filters : dict, optional
            Optional filter definitions used in
            :meth:`calc_vectors_t` to restrict the population.

        Returns
        -------
        dists_t : dict
            Mapping time -> array of internal distances.
        filt_t : dict
            Mapping time -> boolean mask describing which segments
            were included.
        """
        vect,filt_t = self.calc_vectors_t(n,filters=filters)
        # Convert per-frame bond vectors into Euclidean distances
        dists_t =  {t:np.sum(v*v,axis=1)**0.5 for t,v in vect.items()}
        return dists_t,filt_t

    def list_molecule_ids(self,mol):
        """Group atom indices by molecule for a given species.

        Parameters
        ----------
        mol : str
            Molecule name (as stored in ``self.mol_names``).

        Returns
        -------
        list of ndarray
            List where each element contains the atom indices belonging
            to one molecule of type ``mol``.
        """
        mf = self.mol_names == mol
        ids = self.at_ids[mf]
        this_mol_ids = self.mol_ids[mf]
        molecule_ids = []

        for mid in np.unique(this_mol_ids):
            molecule_ids.append( ids[ this_mol_ids == mid ] )
        return molecule_ids

    def calc_cluster_size_t(self,mol,dcut,method='min',mol2=None):
        """Cluster-size statistics as a function of time.

        Parameters
        ----------
        mol : str
            Name of the molecule species used to define clusters.
        dcut : float
            Distance cutoff used when deciding whether two segments
            belong to the same cluster.
        method : {"com", "min"}, optional
            Distance metric used for clustering (centre-of-mass vs
            minimum distance).
        mol2 : str or None, optional
            Reserved for future use (second species).

        Returns
        -------
        dict
            Rich dictionary containing time series of cluster-size
            distributions, summary statistics (mean/std/max), and
            auxiliary debugging information.
        """
        t0 = perf_counter()
        available_methods = ['com','min']
        if method not in available_methods:
            raise ValueError('Uknown method {:s} --> choose from {:}'.format(method,available_methods))
        
        topol_vector = mol

        ids1,ids2 = self.find_vector_ids(topol_vector)  # segment endpoints for this molecule type

        segmental_ids = self.find_segmental_ids(ids1, ids2)  # map segments back to chain topology

        maptomols = dict()
        for i,seg in enumerate(segmental_ids):  # map each segment index to its unique parent molecule id
            un = np.unique(self.mol_ids[seg])
            if un.shape[0] != 1:
                raise Exception('Segmental ids do not give a unique molecule value --> {}'.format(un))
            maptomols[i] = un[0]

        distribution = dict()  # per-frame lists of cluster sizes
        clusters = dict()      # optional detailed cluster membership per frame
        args = (segmental_ids,dcut,distribution,clusters)

        nframes = self.loop_trajectory('cluster_size'+'_'+method,args)  # build per-frame cluster lists and distributions

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        data = dict()

        data['debugging'] = {'maptomols':maptomols}

        data['clusters'] = clusters
        data['time'] = np.array(list(distribution.keys()))/1000
        data['cluster-mean'] = np.array([np.mean(v) for v in distribution.values()])  # mean size vs time
        data['cluster-std'] = np.array([np.std(v) for v in distribution.values()])   # size fluctuations vs time
        data['cluster-maxsize'] = np.array([np.max(v) for v in distribution.values()])  # largest cluster per frame



        sizes = []  # flatten all per-frame cluster sizes into a single array
        for v in distribution.values():
            sizes.extend(v)
        sizes = np.array(sizes)
        data['sizes'] = sizes

        maxsize = sizes.max()

        data['nc'] = np.arange(1,maxsize+1,1,dtype=int)

        counts = np.array([np.count_nonzero(sizes == j) for j in range(1,maxsize+1) ])  # histogram over global cluster sizes

        data['counts'] = counts
        data['counts/n'] = counts/len(segmental_ids)
        data['clustersize-distribution'] = counts/np.sum(counts)
        data['molecule-distribution'] = data['nc']*data['clustersize-distribution']/np.sum(data['nc']*data['clustersize-distribution'])

        data['probclust-time'] = dict()
        data['probmol-time'] = dict()
        nc = data['nc']
        for k,v in distribution.items():  # build time-resolved probability distributions
            v = np.array([np.count_nonzero(np.array(v) == j) for j in nc ])
            data['probclust-time'][k] = v
            data['probmol-time'][k] = nc*v/np.sum(v*nc)

        data.update( {'probmol-{:d}-time'.format(j) : ass.numpy_values(data['probmol-time'])[:,i] for i,j in enumerate(nc)} )

        return data

    def calc_segmental_pair_distribution(self,binl,dmax,topol_vector,far_region=0.8):
        """Segmental pair distribution function between bonded segments.

        Parameters
        ----------
        binl : float
            Bin width for the distance histogram.
        dmax : float
            Maximum distance considered.
        topol_vector : str
            Topology vector string describing which atom types form the
            segmental bonds (e.g. ``"C C"``).
        far_region : float, optional
            Fraction of the tail region used to normalise ``g(r)``.

        Returns
        -------
        dict
            Pair-distribution information as returned by
            :meth:`normalize_gofr`.
        """
        t0 = perf_counter()

        ids1,ids2 = self.find_vector_ids(topol_vector)

        segmental_ids = self.find_segmental_ids(ids1, ids2)  # group vector ends into segments

        bins = np.arange(0,dmax+binl,binl)
        gofr = np.zeros(bins.shape[0]-1,dtype=float)
        args = (bins,segmental_ids,gofr)

        nframes = self.loop_trajectory('gofr_segments', args)  # fill gofr histogram from all frames

        gofr/=nframes



        n1 = len(segmental_ids)/2 # number of unique segments (ids appear twice)
        n = int((len(segmental_ids)-1)*n1)

        pair_distribution = self.normalize_gofr(bins,gofr,n,n1,far_region,dmax)  # normalise histogram into g(r)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return pair_distribution

    def get_ids_based_on_atomtype(self,ty):
        if ty is None:
            ids = self.at_ids.copy()
        elif ass.iterable(ty):
            f =  self.at_types == ty[0]
            for i in range(1,len(ty)):
                f = np.logical_or(f, self.at_types == ty[i])
                ids = self.at_ids[f].copy()
        elif type(ty) is str:
            ids = self.at_ids[self.at_types==ty].copy()
        else:
            raise ValueError('uknown way to find ids')
        return ids

    def find_gr_pairs(self,ty1,ty2,intra=False,inter=False):
        """Select atom pairs to be used in :math:`g(r)` calculations.

        Parameters
        ----------
        ty1, ty2 : str or None
            Atom types defining the two members of each pair. ``None``
            selects all atom types on that side.
        intra : bool, optional
            If ``True``, keep only intra-molecular pairs (same
            ``mol_id``).
        inter : bool, optional
            If ``True``, keep only inter-molecular pairs (different
            ``mol_id``).

        Returns
        -------
        ids1, ids2 : ndarray of int
            Arrays of atom indices representing the first and second
            partner of each pair.
        n1 : int
            Number of reference atoms (``len(ids1)`` before pairing).
        n : int
            Total number of pairs after filtering.
        """
        if inter and intra:
            raise ValueError('Cannot demand both intra and inter pairs at the same time')

        ids1 = self.get_ids_based_on_atomtype(ty1)
        ids2 = self.get_ids_based_on_atomtype(ty2)

        # Build all distinct (i,j) pairs for the requested atom sets
        pairs = [[i,j] for i in ids1 for j in ids2 if i!=j ]
        npairs = len(pairs)
        pairs = np.array(pairs)

        if intra:
            filt = np.ones(npairs,dtype=bool)
            for i,j in enumerate(pairs):

                if self.mol_ids[j[0]] == self.mol_ids[j[1]]:
                    filt[i] = False

            pairs = pairs[filt]

        if inter:
            filt = np.ones(npairs,dtype=bool)
            for i,j in enumerate(pairs):

                if self.mol_ids[j[0]] != self.mol_ids[j[1]]:
                    filt[i] = False

            pairs = pairs[filt]

        return pairs[:,0], pairs[:,1], len(ids1), pairs.shape[0]

    def calc_pair_distribution(self,binl,dmax,type1=None,type2=None,intra=False,inter=False,
                               far_region=0.8):
        """Compute the radial pair-distribution function :math:`g(r)`.

        This routine accumulates a histogram of pair distances between
        atoms of type ``type1`` and ``type2`` over all frames and
        normalises it to obtain :math:`g(r)`, coordination numbers and
        number densities.

        Parameters
        ----------
        binl : float
            Bin width for the distance histogram.
        dmax : float
            Maximum distance up to which the distribution is computed.
        type1, type2 : str or None, optional
            Atom types defining the pair distribution. If one of them is
            ``None`` the other type is correlated with *all* atoms. If both
            are ``None`` the distribution is computed between all atoms.
        intra : bool, optional
            If ``True``, include only *intra*-molecular pairs.
        inter : bool, optional
            If ``True``, include only *inter*-molecular pairs.
        far_region : float, optional
            Fraction of ``dmax`` that defines the "bulk-like" tail
            (``(far_region * dmax, dmax)``) used to estimate the asymptotic
            number density for normalisation.

        Returns
        -------
        dict
            Dictionary with keys such as ``"d"`` (bin centres),
            ``"gr"`` (pair-distribution function), ``"coordination"``,
            and several auxiliary densities as produced by
            :meth:`normalize_gofr`.

        Notes
        -----
        Use :meth:`find_gr_pairs` to pre‑select pairs and
        :meth:`normalize_gofr` to post‑process the raw histogram.
        """
        t0 = perf_counter()

        bins = np.arange(0,dmax+binl,binl)
        gofr = np.zeros(bins.shape[0]-1,dtype=float)

        ids1,ids2, n1 ,n = self.find_gr_pairs(type1,type2,intra,inter)

        args = (ids1,ids2,bins,gofr)

        nframes = self.loop_trajectory('gofr_pairs', args)


        gofr/=nframes  # time-average raw pair counts over all frames

        pair_distribution = self.normalize_gofr(bins,gofr,n,n1,far_region,dmax)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return pair_distribution

    @staticmethod
    def normalize_gofr(bins,gofr,n,n1,far_region,dmax):
        """Normalise a raw pair-distance histogram.

        Parameters
        ----------
        bins : ndarray
            Array of histogram bin edges.
        gofr : ndarray
            Raw pair counts per bin accumulated over all frames.
        n : int
            Total number of contributing pairs.
        n1 : int
            Number of reference atoms (those counted as ``type1``).
        far_region : float
            Fraction of ``dmax`` used to define the region where the
            density is assumed to be bulk‑like.
        dmax : float
            Maximum distance considered in the histogram.

        Returns
        -------
        dict
            Dictionary containing bin centres (``"d"``), raw counts,
            shell volumes, number densities, coordination numbers and the
            normalised :math:`g(r)` under the key ``"gr"``.
        """
        pair_distribution = dict()
        cb = center_of_bins(bins)
        pair_distribution['d'] = cb
        npairs = gofr.copy()
        pair_distribution['npairs'] = npairs

        vshell = (4*np.pi/3)*(bins[1:]**3-bins[:-1]**3)

        pairdens = npairs/vshell
        pair_distribution['v'] = (4*np.pi/3)*cb**3
        pair_distribution['pair density'] = pairdens

        numdens = pairdens/n1

        pair_distribution['number density'] = numdens
        pair_distribution['probablilty number density'] = numdens/numdens.sum()

        pair_distribution['coordination'] = npairs/n

        b =(far_region*dmax,dmax)
        f = np.logical_and(b[0]<cb,cb<=b[1])
        far_rho = numdens[f].mean()
        far_rho_std = numdens[f].std()

        pair_distribution['far_rho'] = far_rho
        pair_distribution['far_rho_std'] = far_rho_std

        norm_numdens= numdens/far_rho  # g(r) = rho(r) / rho_bulk

        pair_distribution['gr'] = norm_numdens
        return pair_distribution

    def calc_size(self):
        size = np.zeros(3)
        args = (size,)
        nframes = self.loop_trajectory('minmax_size',args)
        return size/nframes




    def init_xt(self,xt,dtype=float):
        """Convert a time-indexed dictionary to a stacked NumPy array.

        Parameters
        ----------
        xt : dict
            Mapping from time keys (typically floats) to NumPy arrays of
            identical shape.
        dtype : data-type, optional
            Target dtype for the resulting array.

        Returns
        -------
        ndarray
            Array of shape ``(n_frames, *xt[t].shape)`` containing the
            values stacked in the order of ``xt.keys()``.
        """
        x0 = xt[list(xt.keys())[0]]
        nfr = len(xt)
        shape = (nfr,*x0.shape)
        x_nump = np.empty(shape,dtype=dtype)

        for i,t in enumerate(xt.keys()):
           x_nump[i] = xt[t]

        return  x_nump

    def init_prop(self,xt):
        """Allocate work arrays for time-dependent properties.

        Parameters
        ----------
        xt : dict
            Time-indexed input series whose length defines the number of
            time points for the property.

        Returns
        -------
        Prop_nump : ndarray
            Zero-initialised array that will hold the accumulated
            property values.
        nv : ndarray
            Zero-initialised counter array used to track how many
            contributions have been added at each time index.
        """
        nfr = len(xt)
        Prop_nump = np.zeros(nfr,dtype=float)
        nv = np.zeros(nfr,dtype=float)
        return Prop_nump,nv


    def calc_vectors_t(self,topol_vector,filters={}):
        """Compute segmental bond vectors as a function of time.

        Parameters
        ----------
        topol_vector : int or sequence of str
            Either an integer (2, 3 or 4) specifying 1‑2 / 1‑3 / 1‑4
            connectivity, or an explicit list of atom types used by
            :meth:`find_vector_ids` to locate segment vectors.
        filters : dict, optional
            Analysis filters applied on top of the raw vectors.

        Returns
        -------
        vec_t : dict
            Time-indexed dictionary of segmental vectors with shape
            ``(n_segments, 3)`` per frame.
        filt_per_t : dict
            Per-filter, per-time boolean masks indicating which
            segments contribute.
        """
        ids1,ids2 = self.find_vector_ids(topol_vector)
        args = (ids1,ids2,filters)

        t0 = perf_counter()

        vec_t = dict()
        filt_per_t = dict()

        args = (ids1, ids2, filters, vec_t, filt_per_t)

        nframes = self.loop_trajectory('vects_t', args)

        filt_per_t = ass.rearrange_dict_keys(filt_per_t)

        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return vec_t,filt_per_t


    def get_Dynamics_inner_kernel_functions(self,prop,filt_option,weights_t,**kwargs):
        """Select kernel functions for :meth:`Dynamics`.

        Parameters
        ----------
        prop : str
            Property identifier (``"P1"``, ``"P2"``, ``"MSD"``,
            ``"scalar"``, ``"Fs"`` or a custom kernel name).
        filt_option : {"simple", "strict", "change", "const"} or None
            Filtering strategy used inside the kernel. ``None`` disables
            filtering.
        weights_t : dict or None
            Optional time-dependent weights. If not ``None`` a weighted
            kernel variant is selected.

        Returns
        -------
        tuple
            ``(kernel_func, args_builder, inner_func)`` where each
            element is a callable used by :func:`DynamicProperty_kernel`.
        """
        mapper = {'p1':'costh_kernel',
                  'p2':'cos2th_kernel',
                  'msd':'norm_square_kernel',
                  'scalar':'mult_kernel',
                  'fs':'Fs_kernel',
                  }  # map high-level prop names to inner kernels
        prop = prop.lower()
        if prop in mapper:
            inner_func_name = mapper[prop.lower()]
        else:
            inner_func_name = prop

        name = 'dynprop'
        af_name = 'get'

        if filt_option is not None:
            ps = '_{:s}'.format(filt_option)
            name += ps
            af_name+= ps
        if weights_t is not None:
            ps = '_weighted'
            name += ps
            af_name += ps

        func_name = '{:s}__kernel'.format(name)
        args_func_name = '{:s}__args'.format(af_name)

        logger.info(' func name : "{:s}" \n argsFunc name : "{:s}" \n innerFunc name : "{:s}" '.format(func_name,args_func_name,inner_func_name))

        # Resolve main kernel, argument-builder, and inner property kernel by constructed names
        funcs = (globals()[func_name],
                 globals()[args_func_name],
                 globals()[inner_func_name])

        return funcs

    def set_partial_charge(self):
        if not hasattr(self,'partial_charge'):
            charge = np.empty((self.at_types.shape[0],1),dtype=float)
            for i,ty in enumerate(self.at_types):
                charge[i] = maps.charge_map[ty]
            self.partial_charge = charge
        return

    def segment_ids_per_chain(self,segmental_ids):
        seg0 = segmental_ids[:,0]
        segch = {j : np.isin(seg0,chargs)
                 for j,chargs in self.chain_args.items()}
        return segch

    @staticmethod
    def id_neibs(id0,neibs):
        """Return all atoms connected to ``id0`` via the neighbour graph.

        Parameters
        ----------
        id0 : int
            Starting atom index.
        neibs : dict
            Dictionary mapping atom indices to sets of bonded
            neighbours.

        Returns
        -------
        ndarray
            Sorted array of atom indices that are connected to
            ``id0`` through one or more bonds.
        """
        setids = neibs[id0]
        setids_old = set()
        while( len(setids) != len(setids_old) ):
            setids_old = setids
            for i in setids.copy():
                setids = setids | neibs[i]
        return np.array(list(setids))

    def find_general_ids(self, topol_vector):

        if topol_vector in self.mol_names:
            segmental_ids = self.list_molecule_ids(topol_vector)
        else:
            ids1,ids2 = self.find_vector_ids(topol_vector)

            segmental_ids = self.find_segmental_ids(ids1, ids2)
        self.stored_segmental_ids = segmental_ids
        return segmental_ids

    def find_segmental_ids(self,ids1,ids2):
        """
        Build segment atom-id lists by traversing the bond connectivity graph.
        For each start atom in `ids1`, this function performs a BFS on [self.neibs](cci:1://file:///c:/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/unified_md_pipeline/package/md_pipeline/__init__.py:5933:8-5935:23)
        until it reaches any atom in `ids2`. The returned segment is the (shortest)
        bond path connecting the start atom to the reached target atom.
        Parameters
        ----------
        ids1, ids2 : array-like of int
            Start and target atom-id sets.
        Returns
        -------
        list[np.ndarray]
            List of unique segments. Each segment is an array of atom ids along the
            bond path from a start in `ids1` to a reached atom in `ids2`.
        """

        t0 = perf_counter()
        seg_ids = []
        neibs = self.neibs
        for i,j in zip(ids1,ids2):

            st_i, st_j = {i}, {j}
            both_closest_neibs = set()
            while len(both_closest_neibs) ==0:
                for ii in st_i.copy():
                    st_i.update( neibs[ii] )
                for jj in st_j.copy():
                    st_j.update( neibs[jj] )
                both_closest_neibs = st_i & st_j

            while i not in both_closest_neibs and j not in both_closest_neibs:
                for kk in both_closest_neibs.copy():
                    both_closest_neibs.update(neibs[kk])
            args_per_vec = list(both_closest_neibs)

            seg_ids.append (np.array(args_per_vec))
        
        # enfornce uniqness/ Temporary fix. Should be checked later
        
        seg_ids_copy = []
        seen = set()
        
        for sg in seg_ids:
            key = tuple(sorted(sg))
            if key not in seen:
                seen.add(key)
                seg_ids_copy.append(sg)

        seg_ids = seg_ids_copy

        try:
            seg_ids = np.array(seg_ids)
        except:
            seg_ids = seg_ids
            print('Warning: segmental ids are not uniform in number of atoms')

        self.stored_segmental_ids = seg_ids

        tf = perf_counter() - t0
        try:
            logger.info(f'Number of segments/Size of segments = {seg_ids.shape}, type = {type(seg_ids)}')
        except:
            logger.info(f'Number of segments = {len(seg_ids)}, type = {type(seg_ids)}')
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return  seg_ids

    def calc_chain_dipole_moment_t(self,filters=dict(),**kwargs,):
        """Compute chain dipole moments as a function of time.

        Parameters
        ----------
        filters : dict, optional
            Filters applied to chains (keys are propagated with a
            ``"chain_"`` prefix internally).
        **kwargs
            Additional options controlling how the dipoles are
            constructed. Recognised values include ``option="contour"``,
            ``"endproj"`` and ``"proj"``.

        Returns
        -------
        dipoles_t : dict
            Time-indexed dictionary of chain dipole vectors with shape
            ``(n_chains, 3)`` per frame.
        filters_t : dict
            Per-filter, per-time boolean masks used during the
            accumulation.
        """
        t0 = perf_counter()

        filters = {'chain_'+k : v for k,v in filters.items()}



        if 'option' in kwargs:
            option = kwargs['option']
        else:
            option =''
        dipoles_t = dict()
        filters_t = dict()
        if option =='contour':
            ids1,ids2 = self.find_vector_ids(kwargs['monomer'])
            segmental_ids = self.find_segmental_ids(ids1, ids2)
            segch = self.segment_ids_per_chain(segmental_ids)
            args = (filters,ids1,ids2,segmental_ids,
                    segch,dipoles_t,filters_t)
            ext ='__contour'
        elif option=='':
            ext =''
            args = (filters,dipoles_t,filters_t)
        elif option=='endproj':
            ext ='__endproj'
            args = (filters,dipoles_t,filters_t)
        elif option=='proj':
            ext ='__proj'
            projvec = np.array([x for x in kwargs['projvec']])
            args = (filters,projvec,dipoles_t,filters_t)
        nframes = self.loop_trajectory('chain_dipole_moment'+ext,args)

        filters_t = ass.rearrange_dict_keys(filters_t)

        tf = perf_counter() - t0

        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t



    def calc_total_dipole_moment_t(self,q=None):
        """Compute the total system dipole moment as a function of time.

        Parameters
        ----------
        q : array_like, shape (3,), optional
            Optional direction vector. If given, each dipole is
            projected onto ``q / |q|`` before being stored.

        Returns
        -------
        dict
            Dictionary mapping time keys to total dipole vectors of
            shape ``(3,)`` (or projected scalars when ``q`` is given).
        """
        t0 = perf_counter()


        self.set_partial_charge()

        dipoles_t = dict()
        if q is not None:
            q = np.array(q) ; q = q/np.sum(q*q)**0.5
            q = q.reshape(1,3)
        args = (dipoles_t,q)

        nframes = self.loop_trajectory('total_dipole_moment',args)  # per-frame kernel sums q*r over all atoms

        tf = perf_counter() - t0

        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t

    def calc_segmental_dipole_moment_t(self,topol_vector,filters=dict()):
        """Compute segmental dipole moments versus time.

        Parameters
        ----------
        topol_vector : sequence of str
            Atom-type pattern used to identify segment edges
            (passed to :meth:`find_vector_ids`).
        filters : dict, optional
            Analysis filters applied when building segment dipoles.

        Returns
        -------
        dipoles_t : dict
            Time-indexed dictionary of segment dipole vectors.
        filters_t : dict
            Per-filter, per-time boolean masks.
        """
        t0 = perf_counter()

        #self.set_partial_charge()

        ids1,ids2 = self.find_vector_ids(topol_vector)

        segmental_ids = self.find_segmental_ids(ids1, ids2)

        dipoles_t = dict()
        filters_t = dict()

        args = (filters,ids1,ids2,segmental_ids,dipoles_t,filters_t)

        nframes = self.loop_trajectory('segmental_dipole_moment',args)

        filters_t = ass.rearrange_dict_keys(filters_t)

        tf = perf_counter() - t0

        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t

    def calc_segmental_dipole_moment_correlation(self,topol_vector,
                                       filters=dict()):
        """Compute static correlations between segment dipole moments.

        Parameters
        ----------
        topol_vector : sequence of str
            Pattern used to define segment edges.
        filters : dict, optional
            Additional filters used to select subsets of segments.

        Returns
        -------
        dict
            Nested dictionary ``corrs[filt]`` with fields ``"nc"``
            (number of bonds separating segment pairs), ``"corr"`` and
            ``"corr(std)"`` (mean correlation and standard deviation).
        """
        t0 = perf_counter()
        filters.update({'system':None})
        dipoles_t,filters_t = self.calc_segmental_dipole_moment_t(topol_vector,
                                       filters)

        ids1,ids2 = self.find_vector_ids(topol_vector)
        bond_distmatrix = self.find_bond_distance_matrix(ids1)

        unb = np.unique(bond_distmatrix)
        b0 = dict()
        b1 = dict()
        for k in unb:
            if k>0:
                b =  np.nonzero(bond_distmatrix ==k)
                b0[k] = b[0]
                b1[k] = b[1]

        correlations ={filt:{k:[] for k in unb if k>0} for filt in filters_t}

        args = (dipoles_t,filters_t,b0,b1,correlations)

        nframes = self.loop_trajectory('vector_correlations', args)

        corrs = {kf:{'nc':[],'corr':[],'corr(std)':[]} for kf in correlations}
        for kf in filters_t:
            for k in correlations[kf]:
                c = correlations[kf][k]
                corrs[kf]['nc'].append(k)
                corrs[kf]['corr'].append(np.mean(c))
                corrs[kf]['corr(std)'].append(np.std(c))
            corrs[kf] = {key:np.array(corrs[kf][key]) for key in corrs[kf]}

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return corrs

    def stress_per_t(self,filters=dict()):
        """Compute per-atom stress tensors as a function of time.

        Parameters
        ----------
        filters : dict, optional
            Filters applied when accumulating per-atom stress.

        Returns
        -------
        atomstress_t : dict
            Time-indexed dictionary of per-atom stress arrays.
        filt_per_t : dict
            Per-filter, per-time boolean masks.
        """
        t0 = perf_counter()
        atomstress_t = dict()
        filt_per_t = dict()

        args = (filters,atomstress_t,filt_per_t)

        nframes = self.loop_trajectory('stress_per_atom_t', args)  # delegate per-frame work to coreFunctions

        filt_per_t = ass.rearrange_dict_keys(filt_per_t)

        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return atomstress_t,filt_per_t

    def chains_CM(self,coords):
        """Compute centres of mass for all chains.

        Parameters
        ----------
        coords : ndarray, shape (N, 3)
            Atomic coordinates for the current frame.

        Returns
        -------
        ndarray
            Array of shape ``(n_chains, 3)`` with the centre-of-mass
            position of each chain.
        """
        chain_arg_keys = self.chain_args.keys()

        chain_cm = np.empty((len(chain_arg_keys),3),dtype=float)

        for i,args in enumerate(self.chain_args.values()):  # one CM per chain
            chain_cm[i] = CM(coords[args],self.atom_mass[args])

        return chain_cm

    def segs_CM(self,coords,segids):
        """Compute centres of mass for arbitrary segments.

        Parameters
        ----------
        coords : ndarray, shape (N, 3)
            Atomic coordinates for the current frame.
        segids : array_like of ndarray
            Iterable of integer index arrays or a 2D integer array of
            shape ``(n_segments, n_ids_per_segment)`` specifying which
            atoms belong to each segment.

        Returns
        -------
        ndarray
            Array of shape ``(n_segments, 3)`` with segment centres of
            mass.
        """
        mass = self.atom_mass
        if isinstance(segids, np.ndarray) and segids.ndim == 2:
            c = coords[segids]
            w = mass[segids]
            return np.sum(c * w[..., None], axis=1) / np.sum(w, axis=1)[:, None]

        n = len(segids)
        segcm = np.empty((n,3),dtype=float)
        for i,si in enumerate(segids):
            segcm[i] = CM(coords[si], mass[si])
        return segcm


    def calc_dihedrals_t(self,dih_type,filters=dict()):
        """Compute dihedral angles as a function of time.

        Parameters
        ----------
        dih_type : sequence of str
            Dihedral type key used to select entries from
            :attr:`dihedrals_per_type`.
        filters : dict, optional
            Filters applied when computing dihedrals.

        Returns
        -------
        dihedrals_t : dict
            Dictionary mapping time keys to arrays of shape
            ``(n_dihedrals,)`` (angles in radians).
        filt_per_t : dict
            Per-filter, per-time boolean masks.
        """
        t0 = perf_counter()

        dihedrals_t = dict()
        filt_per_t = dict()

        dih_type = tuple(dih_type)

        dih_ids = self.dihedrals_per_type[dih_type] #array (ndihs,4)
        ids1 = dih_ids[:,0] ; ids2 = dih_ids[:,3]

        args = (dih_ids,ids1,ids2,filters,dihedrals_t,filt_per_t)

        nframes = self.loop_trajectory('dihedrals_t', args)

        filt_per_t = ass.rearrange_dict_keys(filt_per_t)

        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return dihedrals_t, filt_per_t

    def calc_Rg(self,option='__permol'):
        """Compute the radius of gyration.

        Parameters
        ----------
        option : {"", "__permol"}, optional
            ``""`` computes a single system-wide :math:`R_g` and its
            standard deviation. ``"__permol"`` collects :math:`R_g`
            statistics per molecule type.

        Returns
        -------
        dict
            Mapping from labels (e.g. ``"Rg"``, ``"<mol>_Rg"``) to
            averaged values over all frames.

        Notes
        -----
        This method is a thin wrapper around the trajectory kernel
        ``"Rg"`` / ``"Rg__permol"`` executed via
        :meth:`loop_trajectory`.
        """
        t0 = perf_counter()

        if option=='':
            Rg = {'Rg':0.0,'Rgstd':0.0}
        elif 'permol' in option:
            option = '__permol'
            Rg = dict()
            for m in self.molecules:
                Rg.update({m+'_Rg':0.0,m+'_Rgstd':0.0})
        else:
            raise ValueError('option "{:s}" is not specified'.format(option))

        args = (Rg,)

        nframes = self.loop_trajectory('Rg'+option,args)

        Rg = {k1:v/nframes for k1,v in Rg.items()}

        tf= perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return Rg

    def calc_segmolCM_t(self,topol_vector,filters=dict(),option=''):
        """Centres of mass for segments or molecules versus time.

        Parameters
        ----------
        topol_vector : str or sequence
            Segment identifier passed to :meth:`find_general_ids`.
        filters : dict, optional
            Optional filters applied to the selected segments.
        option : str, optional
            Suffix selecting specialised trajectory kernels.

        Returns
        -------
        segcm_t : dict
            Time-indexed dictionary of centres of mass.
        filt_per_t : dict
            Per-filter, per-time boolean masks.
        """
        t0 = perf_counter()

        segcm_t = dict()
        filt_per_t = dict()

        segmol_ids = self.find_general_ids(topol_vector)

        args = (segmol_ids, filters, segcm_t, filt_per_t)

        nframes = self.loop_trajectory('segmolCM_t'+option, args)

        filt_per_t = ass.rearrange_dict_keys(filt_per_t)

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return segcm_t,filt_per_t

    def calc_chainCM_t(self,filters=dict(),option=''):
        """Centres of mass of entire chains versus time.

        Parameters
        ----------
        filters : dict, optional
            Filters applied on chains (keys are prefixed with
            ``"chain_"`` internally).
        option : str, optional
            Suffix selecting specialised trajectory kernels.

        Returns
        -------
        vec_t : dict
            Time-indexed dictionary of chain centres of mass with shape
            ``(n_chains, 3)`` per frame.
        filt_per_t : dict
            Per-filter, per-time boolean masks.
        """
        t0 = perf_counter()
        filters = {'chain_'+k: v for k,v in filters.items()} #Need to modify when considering chains

        vec_t = dict()
        filt_per_t = dict()

        args = (filters,vec_t,filt_per_t)

        nframes = self.loop_trajectory('chainCM_t'+option, args)

        filt_per_t = ass.rearrange_dict_keys(filt_per_t)

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return vec_t,filt_per_t

    def calc_coords_t(self,ids,filters=dict()):
        """Return raw coordinates for a set of atoms versus time.

        Parameters
        ----------
        ids : array_like of int
            Atom indices whose coordinates are tracked.
        filters : dict, optional
            Filters applied when extracting coordinates.

        Returns
        -------
        c_t : dict
            Time-indexed dictionary of coordinate arrays.
        filt_t : dict
            Per-filter, per-time boolean masks.
        """
        t0 = perf_counter()

        c_t = dict()
        filt_t = dict()
        args = (filters,ids,c_t,filt_t)
        nframes = self.loop_trajectory('coords_t', args)
        filt_t = ass.rearrange_dict_keys(filt_t)

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return c_t,filt_t

    def calc_segCM_t(self,topol_vector, filters=dict()):
        """Centres of mass of polymer segments versus time.

        Parameters
        ----------
        topol_vector : sequence of str
            Pattern used to define segment edges.
        filters : dict, optional
            Filters applied when computing segment centres of mass.
        Returns
        -------
        vec_t : dict
            Time-indexed dictionary of segment centres of mass.
        filt_per_t : dict
            Per-filter, per-time boolean masks.
        """
        t0 = perf_counter()

        ids1,ids2 = self.find_vector_ids(topol_vector)

        segmental_ids = self.find_segmental_ids(ids1, ids2)

        vec_t = dict()
        filt_per_t = dict()

        args = (filters,segmental_ids, ids1,ids2, vec_t,filt_per_t)

        nframes = self.loop_trajectory('segCM_t', args)

        filt_per_t = ass.rearrange_dict_keys(filt_per_t)

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return vec_t,filt_per_t

    def Dynamics(self,prop,xt,filt_t=None,weights_t=None,
                 filt_option='simple', block_average=False,
                 multy_origin=True,every=1,q=None):
        """Compute generic single-particle dynamical correlation functions.

        Supported observables include first/second-order Legendre
        reorientational correlators (``"P1"``, ``"P2"``), mean-square
        displacements (``"MSD"``), scalar products (``"scalar"``) and
        the self-intermediate scattering function (``"Fs"``).

        Parameters
        ----------
        prop : str
            Observable identifier (e.g. ``"P1"``, ``"P2"``, ``"MSD"``,
            ``"Fs"``).
        xt : dict
            Time-indexed dictionary of input vectors/positions with
            shape ``(N, d)``.
        filt_t : dict or None, optional
            Optional time-indexed boolean masks for sub‑populations.
        weights_t : dict or None, optional
            Optional time-indexed weights for each particle.
        filt_option : {"simple", "strict", "change", "const"} or None
            Strategy for combining ``filt_t`` at origin and lag times.
        block_average : bool, optional
            If ``True``, perform a two-step average (over population
            then over time origins).
        multy_origin : bool, optional
            If ``True``, use multiple time origins in the averaging.
        every : int, optional
            Stride in time origins used for the kernel.
        q : float or array_like, optional
            Wave-vector magnitude used for ``"Fs"``.

        Returns
        -------
        dict or None
            Dictionary with keys ``"time"`` and ``prop.lower()``. If a
            division-by-zero occurs due to incompatible filters or
            weights, ``None`` is returned.
        """
        tinit = perf_counter()  # overall timing for this dynamics call

        Prop_nump,nv = self.init_prop(xt)  # allocate output array and infer population size
        x_nump = self.init_xt(xt)  # pack input dictionary into contiguous NumPy array




        if filt_t is not None:
            f_nump = self.init_xt(filt_t,dtype=bool)
            if filt_option is None:
                filt_option = 'simple'
            elif filt_option =='const':
                filt_t = ass.stay_True(filt_t)
                filt_option='strict'
        else:
            f_nump = None
            filt_option = None

        if weights_t is not None:
            w_nump = self.init_xt(weights_t)  # time-dependent weights stored in same layout as xt
        else:
            w_nump = None


        # Select the appropriate low-level kernels and argument builders for this observable
        func,func_args,func_inner = \
        self.get_Dynamics_inner_kernel_functions(prop,filt_option,weights_t)

        if prop.lower() == 'fs':
            if q is None:
                raise Exception('For {} calculation you must give a "q" value '.format(prop))
            kernel_args =(q,)
        else:
            kernel_args = tuple()

        # Collect all state and configuration into a flat argument tuple for the numba kernel
        args = (func,func_args,func_inner,
                Prop_nump,nv,
                x_nump,f_nump,w_nump,
                block_average,
                multy_origin,
                every,kernel_args)


        try:
            #prop_kernel(prop_nump, nv, x_nump, f_nump, nfr)
            DynamicProperty_kernel(*args)  # main numba kernel computing the correlation
        except ZeroDivisionError as err:
            logger.error('Dynamics Run {:s} --> There is a {} --> Check your filters or weights'.format(prop,err))
            return None


        #tf2 = perf_counter()
        if prop.lower() =='p2':
            Prop_nump = 0.5*(3*Prop_nump-1.0)
        t = ass.numpy_keys(xt)
        dynamical_property = {'time':t-t.min(), prop : Prop_nump }  # shift time axis to start at zero
        #tf3 = perf_counter() - tf2

        tf = perf_counter()-tinit
        #logger.info('Overhead: {:s} dynamics computing time --> {:.3e} sec'.format(prop,overheads+tf3))
        ass.print_time(tf,inspect.currentframe().f_code.co_name +'" ---> Property: "{}'.format(prop))
        return dynamical_property

    def multy_tau_average(self,xt,every=1):
        """Average a scalar time series over multiple time origins.

        Parameters
        ----------
        xt : dict
            Time-indexed dictionary of scalar or vector values.
        every : int, optional
            Stride in origin times used for the averaging.

        Returns
        -------
        dict or None
            Dictionary containing ``"time"`` and ``"corr"``. Returns
            ``None`` if a division-by-zero occurs.
        """
        tinit = perf_counter()

        Prop_nump,nv = self.init_prop(xt)
        x_nump = self.init_xt(xt)

        try:
            scalar_time_origin_average(Prop_nump,nv,x_nump,every)  # numba kernel performing origin averaging
        except ZeroDivisionError as err:
            logger.error('multy tau run  --> There is a {} --> Check your filters or weights'.format(err))
            return None


        t = ass.numpy_keys(xt)
        dynamical_property = {'time':t-t.min(), 'corr' : Prop_nump }  # time-shifted correlation

        tf = perf_counter()-tinit
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return dynamical_property

    def get_TACF_inner_kernel_functions(self,prop,filt_option,weights_t):
        """Select kernel functions for trigonometric autocorrelations.

        Parameters
        ----------
        prop : {"cos", "sin"}
            Type of trigonometric correlation to compute.
        filt_option : {"simple", "strict", "change", "const"} or None
            Filtering strategy used for the underlying dynamics.
        weights_t : dict or None
            Optional time-dependent weights.

        Returns
        -------
        tuple
            Seven callable objects required by :func:`TACF_kernel`:
            main kernel, argument builders, moment kernels and
            zero-time contributions.
        """
        inner_mapper = {'cos':'cosCorrelation_kernel',
                        'sin':'sinCorrelation_kernel'}  # main correlation kernels for cos/sin
        inner_zero_mapper = {'cos':'fcos_kernel',
                             'sin':'fsin_kernel'}  # zero-time contributions for cos/sin

        inner_func_name = inner_mapper[prop.lower()]
        inner_zero_func_name = inner_zero_mapper[prop.lower()]

        name = 'dynprop'
        af_name = 'get'
        args_z_name = 'get_zero'
        mean_func_name = 'mean'
        secmom_func_name = 'secmoment'


        if filt_option is not None:
            ps = '_{:s}'.format(filt_option)
            pz = '_filt'
            name += ps
            af_name+= ps
            args_z_name +=pz
            mean_func_name+=pz
            secmom_func_name+=pz
        if weights_t is not None:
            ps = '_weighted'
            args_z_name += ps
            name += ps
            af_name += ps
            mean_func_name+=ps
            secmom_func_name+=ps

        func_name = '{:s}__kernel'.format(name)
        args_func_name = '{:s}__args'.format(af_name)
        args_zero_func_name = '{:s}__args'.format(args_z_name)
        mean_func_name = '{:s}__kernel'.format(mean_func_name)
        secmom_func_name = '{:s}__kernel'.format(secmom_func_name)

        func_names = [func_name,args_func_name, inner_func_name,
                      mean_func_name,secmom_func_name,
                      args_zero_func_name,
                      inner_zero_func_name]  # all building blocks required by TACF_kernel

        s = ''.join( ['f{:d}={:s} \n'.format(i,f) for i,f in enumerate(func_names)] )

        logger.info(s)

        funcs = tuple([ globals()[f] for f in func_names])  # resolve function objects from their names
        return funcs

    def TACF(self,prop,xt,filt_t=None,
             wt=None,filt_option=None,block_average=False):
        """Compute trigonometric time autocorrelation functions.

        Parameters
        ----------
        prop : {"cos", "sin"}
            Choice of correlation kernel.
        xt : dict
            Time-indexed dictionary of scalar values.
        filt_t : dict or None, optional
            Optional boolean masks per time.
        wt : dict or None, optional
            Optional time-dependent weights.
        filt_option : {"simple", "strict", "change", "const"} or None
            Filtering strategy.
        block_average : bool, optional
            If ``True``, perform a two-step average over population and
            time origins.

        Returns
        -------
        dict
            Dictionary with keys ``"time"`` and ``"tacf"`` containing
            the autocorrelation function.
        """
        tinit = perf_counter()  # overall timing for TACF computation

        Prop_nump,nv = self.init_prop(xt)  # main correlation accumulator
        mu_val,mu_num = self.init_prop(xt)  # first moment accumulator and normalisation
        secmom_val,secmom_num = self.init_prop(xt)  # second moment accumulator and normalisation

        x_nump = self.init_xt(xt,dtype=float)  # pack scalar time series into NumPy array

        if filt_t is not None:
            f_nump = self.init_xt(filt_t,dtype=bool)
            if filt_option is None:
                filt_option= 'simple'
            if filt_option =='const':
                filt_option = 'strict'
                filt_t = ass.stay_True(filt_t)
        else:
            f_nump = None
            filt_option = None

        if wt is not None:
            w_nump = self.init_xt(wt)  # optional time-dependent weights
        else:
            w_nump = None

        # Select appropriate inner kernels and argument builders given prop / filters / weights
        func_name, func_args, inner_func,\
        mean_func, secmoment_func, func_args_zero, inner_func_zero \
        = self.get_TACF_inner_kernel_functions(prop,filt_option,wt)


        args = (func_name, func_args, inner_func,
              mean_func, secmoment_func, func_args_zero, inner_func_zero,
              Prop_nump,nv,
              mu_val,mu_num,secmom_val,secmom_num,
              x_nump, f_nump, w_nump,
              block_average)

        TACF_kernel(*args)
        #print(Prop_nump)
        t = ass.numpy_keys(xt)
        TACF_property = {'time':t -t.min(),'tacf':Prop_nump}
        tf = perf_counter()-tinit

        ass.print_time(tf,inspect.currentframe().f_code.co_name)

        return TACF_property

    def get_Kinetics_inner_kernel_functions(self,wt):
        """Select inner kernel functions for :meth:`Kinetics`.

        Parameters
        ----------
        wt : dict or None
            Optional time-dependent weights. If ``None``, an
            unweighted kernel is used.

        Returns
        -------
        tuple
            ``(kernel_func, args_builder)`` to be passed to
            :func:`Kinetics_kernel`.
        """
        if wt is None:
            func_args = 'get__args'
            func_name = 'Kinetics_inner__kernel'
        else:
            func_args = 'get_weighted__args'
            func_name = 'Kinetics_inner_weighted__kernel'

        logger.info('func name : {:s} , argsFunc name : {:s}'.format(func_name,func_args))

        # Resolve the selected kernel and its argument-builder from the global namespace
        return globals()[func_name],globals()[func_args]

    def Kinetics(self,xt,wt=None,block_average=False,
                 multy_origin=True):
        """Compute a simple two-state kinetic observable.

        This routine interprets the boolean time series in ``xt`` as a
        population in a given state and evaluates how that population
        decays or transforms over time. Optionally, time-dependent
        weights ``wt`` can be supplied.

        Parameters
        ----------
        xt : dict
            Dictionary mapping time indices to boolean arrays of shape
            ``(N,)`` that indicate whether each entity (e.g. segment or
            chain) is in the state of interest.
        wt : dict or None, optional
            Dictionary mapping time indices to weight arrays of shape
            ``(N,)``. When provided, the kinetic observable is computed
            as a weighted average.
        block_average : bool, optional
            If ``True``, averages are first taken over the population
            and then over time origins; if ``False`` both are averaged
            simultaneously.
        multy_origin : bool, optional
            If ``True`` (default), a multiple-time-origin scheme is
            used; otherwise only a single origin is considered.

        Returns
        -------
        dict
            Dictionary with keys ``"time"`` (time axis shifted to
            start at zero) and ``"K"`` (kinetic observable as a
            function of time).
        """
        tinit = perf_counter()  # overall timing for kinetics calculation

        Prop_nump,nv = self.init_prop(xt)  # allocate accumulator and infer population size
        x_nump = self.init_xt(xt,dtype=bool)  # pack boolean state trajectory into NumPy array

        if wt is not None:
            w_nump = self.init_xt(wt)
        else:
            w_nump = None

        func_name,func_args = self.get_Kinetics_inner_kernel_functions(wt)

        # Bundle state, configuration and chosen kernels into arguments for the numba kernel
        args = (func_name,func_args,
                Prop_nump,nv,
                x_nump,w_nump,
                block_average,
                multy_origin)
        Kinetics_kernel(*args)
        t = ass.numpy_keys(xt)
        kinetic_property = {'time':t-t.min(),'K': Prop_nump}  # shift time axis to start at zero
        tf = perf_counter()-tinit
        #logger.info('Overhead: {:s} dynamics computing time --> {:.3e} sec'.format(prop,overheads+tf3))
        ass.print_time(tf,inspect.currentframe().f_code.co_name +'" ---> Property: "Kinetics')
        return kinetic_property




class Analysis_Confined(Analysis):
    """Analysis class specialised for confined geometries.

    Extends :class:`Analysis` with helpers for systems where a subset
    of molecules (``particle``) acts as a confining object and another
    subset (``polymer``) is analysed relative to that object. Typical
    examples include polymer chains adsorbed on planar, cylindrical or
    spherical surfaces.

    The confinement type is selected via the ``conftype`` keyword and
    must correspond to one of the methods implemented in
    :class:`Distance_Functions`, :class:`Box_Additions`,
    :class:`bin_Volume_Functions` and :class:`unit_vector_Functions`
    (e.g. ``"zdir"``, ``"zcylindrical"``, ``"sherical"``).
    """

    def __init__(self,    topol_file,
                 connectivity_info,
                 memory_demanding=False,
                 reference_point_method = 'origin',
                 **kwargs):
        super().__init__(topol_file,
                         connectivity_info,
                         reference_point_method = reference_point_method,
                         memory_demanding = memory_demanding,**kwargs)
        known_kwargs = ['conftype','adsorption_interval','particle_method','polymer_method',
                        'polymer','particle','cylinder_length','types_from_itp', 'fftop',
                        ]
        defaults = {'particle_method':'molname',
                    'polymer_method':'molname'
                    }
        #Obligatory keywords
        if 'conftype' in kwargs:
            self.conftype = kwargs['conftype']
        else:
            raise Exception('"conftype" has no default value. Available options are:\n zdir\n ydir\n xdir\n zcylindrical\nsherical_particle')

        for k in ['particle','polymer']:
            if k not in kwargs:
                raise Exception('You need to pass the keyword "{:s}" since it does not have a default value'.format(k))

        #Unkown keywords
        for k in kwargs:
            if k not in known_kwargs:
                raise Exception('You have  passed the variable {:s} but I dont know what to do with it.\n Check also for typos'.format(k))

        #Special keywords/might have default
        if 'adsorption_interval' in kwargs:
           self.setup_adsorption(kwargs['adsorption_interval'])
        else:
            self.adsorption_interval = ((0,0),)

        #Keywords with default values
        for k,d in defaults.items():
            if k not in kwargs:
                setattr(self,k,d)
            else:
                setattr(self,k,kwargs[k])

        if 'cylinder_length' in kwargs:
            self.cylinder_length = kwargs['cylinder_lenght']
            try:
                self.ztrain = self.kwargs['ztrain']
            except KeyError as e:
                raise e('when you give a finite length of a cylinder you need to provide also the "ztrain", which corresponds to the adsorbed distance of the polymer above and below the cylinder')
            self.train_specific_method = self.train_in_finite_cylinder
        self.kwargs = kwargs


        self.confined_system_initialization()

        self.polymer_ids = np.where(self.polymer_filt)[0]
        self.particle_ids = np.where(self.particle_filt)[0]
        return

    def setup_adsorption(self,adsorption_interval):
        if not ass.iterable(adsorption_interval[0]):
                self.adsorption_interval = (tuple(float(x) for x in adsorption_interval),)
        else:
            self.adsorption_interval = tuple(tuple(ai) for ai in adsorption_interval)
        for ai in self.adsorption_interval:
            if len(ai) != 2:
                raise ValueError('Wrong value of Adsorption interval = {} --> must be of up and low value'.format(ai))
    ############## General Supportive functions Section #####################

    def find_connectivity_per_chain(self):
        cargs =dict()
        x = self.sorted_connectivity_keys
        for j,args in self.chain_args.items():
            f1 = np.isin(x[:,0],args)
            f2 = np.isin(x[:,1],args)
            f = np.logical_or(f1,f2)
            cargs[j] = x[f]
        self.connectivity_per_chain = cargs
        return

    def find_systemic_filter(self,name):
        method = getattr(self,name+'_method')

        if method == 'molname':
            compare_array = self.mol_names
        elif method == 'atomtypes':
            compare_array = self.at_types
        elif method == 'molids':
            compare_array = self.mol_ids
        elif method =='atomids':
            compare_array = self.at_ids
        else:
            raise NotImplementedError('method "{}" is not Implemented'.format(method))

        try:
            look_name_s = self.kwargs[name]
        except KeyError:
            raise Exception('You need to provide the key word "{:s}"'.format(name))
        else:
            t1 = type(look_name_s) is str
            t2 = type(look_name_s) is list
            t3 = type(look_name_s) is np.ndarray
            if t1 or t2 or t3:
                if t2 or t3:
                    for i in range(1,len(look_name_s)):
                        if type(look_name_s[i-1]) != type(look_name_s[i]):
                                raise Exception('elements in variable {:s} must be of the same type'.format(name))
            else:
                raise NotImplementedError('{:s} variable is allowed to be either string, list or array'.format(name))

        filt = np.isin(compare_array,look_name_s)
        setattr(self,name+'_filt',filt)

        return

    def find_particle_filt(self):
        self.find_systemic_filter('particle')
        return

    def find_polymer_filt(self):
        self.find_systemic_filter('polymer')
        return

    def translate_particle_in_box_middle(self,coords,box):
        particle_cm = self.get_particle_cm(coords)
        cds = coords.copy()
        cds += box/2 - particle_cm
        cds = implement_pbc(cds,box)
        return cds

    def translated_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords, box)
        return coords

    def get_whole_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords, box)
        coords = self.unwrap_coords(coords, box)
        return coords

    def confined_system_initialization(self):
        """Initialise derived data structures for a confined system.

        This method prepares all attributes that depend on the chosen
        confinement type:

        * identifies particle/polymer atoms and their masses,
        * selects distance, box and volume helper functions from
          :class:`Distance_Functions`, :class:`Box_Additions`,
          :class:`bin_Volume_Functions` and
          :class:`unit_vector_Functions`,
        * builds per-chain and per-particle index groups.

        It is called once during construction to keep later analysis
        routines lightweight.
        """
        t0 = perf_counter()
        self.find_particle_filt()
        self.nparticle = self.mol_ids[self.particle_filt].shape[0]

        self.find_polymer_filt()
        self.npol = self.mol_ids[self.polymer_filt].shape[0]

        self.particle_mass = self.atom_mass[self.particle_filt]
        self.polymer_mass = self.atom_mass[self.polymer_filt]

        #self.find_masses()
        self.unique_atom_types = np.unique(self.at_types)

        #Getting the prober functions
        self.dfun = self.get_class_function(Distance_Functions,self.conftype)
        self.box_add = self.get_class_function(Box_Additions,self.conftype)
        self.volfun = self.get_class_function(bin_Volume_Functions,self.conftype)
        self.unit_vectorFun = self.get_class_function(unit_vector_Functions,self.conftype)
        ##########

        self.find_args_per_residue(self.polymer_filt,'chain_args')
        self.find_connectivity_per_chain()
        self.find_args_per_residue(self.particle_filt,'particle_args')
        self.nparticles = len(self.particle_args.keys())

        self.all_args = np.arange(0,self.natoms,1,dtype=int)

        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def get_class_function(self,_class,fun,inplace=False):
        fun = getattr(_class,fun)
        if inplace:
            attr_name = _class+'_function'
            setattr(self,attr_name, fun)
        return fun

    def get_frame_basics(self,frame):
        box  = self.get_box(frame)
        coords = self.get_coords(frame)%box
        rf = self.get_reference_point(self.current_frame)%box
        d = self.get_distance_from_ref(coords, rf)
        return coords,box,d

    def get_whole_frame_basics(self,frame):
        coords = self.get_whole_coords(frame)
        box  = self.get_box(frame)
        cm = self.get_particle_cm(coords)
        d = self.dfun(self,coords,cm)
        return coords,box,d,cm



    def get_particle_cm(self,coords):
        """Return the centre of mass of the confining particle.

        Parameters
        ----------
        coords : ndarray, shape (N, 3)
            System coordinates for the current frame.

        Returns
        -------
        ndarray, shape (3,)
            Centre-of-mass position of all atoms selected by
            ``self.particle_filt``.
        """
        cm =CM ( coords[self.particle_filt], self.particle_mass)
        return cm



    ###############End of General Supportive functions Section#########

    ############### Conformation Calculation Supportive Functions #####




    def is_bridge(self,coords,istart,iend,periodic_image_args):
        # used to distinguish loops from bridges

        # check if one belong to the periodic_images and other not --> this means bridge
        e = iend in periodic_image_args
        s = istart in periodic_image_args
        if (e and not s) or (s and not e):
            return True

        # check if belong to different particle
        if hasattr(self, 'many_particles') and self.many_particles == True:
            r0 = coords[istart]
            re = coords[iend]
            part_ids = np.unique(self.mol_ids[self.particle_filt])
            #dm0 = dict()
            #dm1 = dict()
            dcrit = abs(self.adsorption_interval[0][1] - self.adsorption_interval[0][0])
            for m in part_ids:
                c = coords[m == self.mol_ids]
                d0 = Distance_Functions.minimum_distance(r0.reshape(1,3),c)
                de = Distance_Functions.minimum_distance(re.reshape(1,3),c)
                #dm0[m] = d0
                #dm1[m] = dm1
                if d0[0] < dcrit and de[0]>dcrit:
                    return True
        return False

    def train_in_finite_cylinder(self,coords,cmp,ftrain):

        Lc = self.cylinder_length
        zd_train = self.ztrain

        zd = Distance_Functions.zdir(None,coords,cmp)

        fz = np.logical_and(zd >=Lc/2,zd<=Lc/2+zd_train)                                 #
        fud_surf = np.logical_and(fz,ftrain)

        return np.logical_or(ftrain,fud_surf)


    def get_filt_train(self):
        """Build boolean filters for adsorbed trains and image trains.

        The method determines which polymer atoms belong to adsorbed
        trains based on the configured ``adsorption_interval`` and the
        confinement geometry. It also marks atoms that are trains only
        in periodic images (``image_trains``).

        Returns
        -------
        ftrain : ndarray of bool, shape (N,)
            Mask selecting atoms that are part of an adsorbed train in
            the minimum-image representation.
        image_trains : ndarray of bool, shape (N,)
            Mask selecting atoms that are trains only through periodic
            images (i.e. adsorbed in a translated image but not in the
            primary box).
        """

        coords = self.get_coords(self.current_frame)
        box = self.get_box(self.current_frame)
        rf = self.get_reference_point(self.current_frame)

        rf_pbc = rf%box
        coords_pbc = coords%box

        d_pbc = self.get_distance_from_ref(coords_pbc, rf_pbc)
        d_raw = self.get_distance_from_ref(coords, rf)

        ftrain = False
        nonp_ftrain = False

        for interval in self.adsorption_interval:

            dlow = interval[0]
            dup = interval[1]

            fin = filt_uplow( d_pbc, dlow, dup)

            ftrain = np.logical_or(ftrain, fin)

            fin_nonp = filt_uplow(d_raw,dlow,dup)
            nonp_ftrain = np.logical_or(nonp_ftrain,fin_nonp)

        ftrain = np.logical_and(ftrain, self.polymer_filt)
        nonp_ftrain = np.logical_and(nonp_ftrain, self.polymer_filt)

        image_trains = np.logical_and(ftrain,np.logical_not(nonp_ftrain))

        return ftrain,image_trains

    def get_distance_from_particle(self,r=None,ids=None):
        """Minimum-image distance between atoms and the confining particle.

        Exactly one of ``r`` or ``ids`` may be specified; if both are
        ``None`` the full coordinate array of the current frame is
        used.

        Parameters
        ----------
        r : ndarray, shape (n, 3), optional
            Coordinates at which distances are evaluated.
        ids : array_like of int, optional
            Atom indices whose coordinates are taken from the current
            frame.

        Returns
        -------
        ndarray, shape (n,)
            Minimum-image distance from each point to the confining
            particle, as defined by ``self.dfun`` for the chosen
            confinement type.
        """
        frame = self.current_frame

        box = self.get_box(frame)
        coords = self.get_coords(frame)
        if r is None:
            if ids is not None:
                r = coords[ids]
            else:
                r = coords
        cm = self.get_particle_cm(coords)

        d = 1e16
        for L in self.box_add(box):
            d = np.minimum(d,self.dfun(self,r,cm+L))
        return d

    def get_distance_from_ref(self,coords, cref):
        """Get distance of the coords based on the confinment type
        ----------
        coords : ndarray, shape (n, 3), optional
            Coordinates at which distances are evaluated.
        cref : array_like (3,) or (n,3) for multiple reference points
        Returns
        -------
        ndarray, shape (n,)
            Distance of coords to cref  for the chosen
            confinement type.
        """
        if hasattr(self,'conftype'):
            dfun = getattr(Distance_Functions, self.conftype)
        else:
            dfun = getattr(Distance_Functions, 'minimum_distance')
            if cref.shape == (3,):
                cref = cref.reshape((1,3))
        d = dfun (coords, cref)
        return d

    def conformations(self):
        """Classify polymer conformations into trains, tails, loops, bridges.

        The classification is performed for the current frame using the
        adsorption filters from :meth:`get_filt_train` and the
        connectivity graph. Chunks along adsorbed chains are labelled
        according to whether they are directly adsorbed (trains), form
        loops, terminate away from the surface (tails), or connect
        different adsorption regions (bridges).

        Returns
        -------
        ads_chains : ndarray of int
            Indices of adsorbed chains.
        args_train : ndarray of int
            Atom ids belonging to trains.
        args_tail : ndarray of int
            Atom ids belonging to tails.
        args_loop : ndarray of int
            Atom ids belonging to loops.
        args_bridge : ndarray of int
            Atom ids belonging to bridges.
        """
        box = self.get_box(self.current_frame)

        ftrain, image_trains = self.get_filt_train()
        args_train = np.nonzero(ftrain)[0]
        periodic_image_args = set(np.nonzero(image_trains)[0])
        #logger.debug('Number of periodic image trains ={:d}\n Number of trains = {:d}'.format(len(periodic_image_args),args_train.shape[0]))
        #ads_chains
        ads_chains = np.unique(self.mol_ids[ftrain])

        fads_chains = np.isin(self.mol_ids,ads_chains)
        args_ads_chain_atoms = np.nonzero(fads_chains)[0]


        args_tail = np.empty(0,dtype=int) ;
        args_bridge = np.empty(0,dtype=int) ;
        args_loop  = np.empty(0,dtype=int)

        coords = self.get_coords(self.current_frame) # need to identify bridges over loops

        for j in ads_chains:
            #args_chain = self.chain_args[j]
            connectivity_args = self.connectivity_per_chain[j]
            #nch = args_chain.shape[0]

            #1) Identiyfy connectivity nodes
            nodes = []
            for c in connectivity_args:
                ft = ftrain[c]
                if ft[0] and not ft[1]: nodes.append( (c[1],c[0]) )
                if not ft[0] and ft[1]: nodes.append( (c[0],c[1]) )

            #2) Loop over nodes and identify chunks
            for node in nodes:

                chunk = {node[0]}
                istart = node[1]

                old_chunk = set() ; loopBridge = False ; found_iend = False
                new_neibs = chunk.copy()

                while old_chunk != chunk:
                    old_chunk = chunk.copy()
                    new_set = set()
                    for ii in new_neibs:
                        for neib in self.neibs[ii]:

                            if neib == istart: continue

                            if ftrain[neib]:
                                loopBridge = True

                                if not found_iend:
                                    iend = neib
                                    found_iend = True
                                continue

                            if neib not in chunk:
                                new_set.add(neib)
                                chunk.add(neib)
                    new_neibs = new_set

                if not found_iend:
                    try: del iend
                    except: pass


                chunk = np.array(list(chunk),dtype=int)

                assert not ftrain[chunk].any(), \
                    'chunk in train chain {:d}, node {}, chunk size = {:d}\
                    ,istart = {:d}, iend ={:d} \n\n chunk =\n {} \n\n'.format(j,
                    node,chunk.shape[0],istart,iend,chunk)
                assert fads_chains[chunk].all(),\
                    'chunk out of adsorbed chains, j = {:d} ,\
                        node = {} n\n chunk \n {} \n\n'.format(j,
                        node,chunk)

                if loopBridge:
                    if not self.is_bridge(coords,istart,iend,periodic_image_args):
                        args_loop = np.concatenate( (args_loop, chunk) )
                    else:
                        #logger.debug('chain = {:d}, chunk | (istart,iend) = ({:d}-{:d}) is bridge'.format(j,istart,iend))
                        args_bridge = np.concatenate( (args_bridge, chunk) )
                else:
                    args_tail = np.concatenate( (args_tail, chunk) )

        #args_tail = np.unique(args_tail)
        args_loop = np.unique(args_loop)
        args_bridge = np.unique(args_bridge)
        assert not np.isin(args_train,args_tail).any(),\
        'tails in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_tail)))

        assert not np.isin(args_train,args_loop).any(),\
        'loops in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_loop)))

        assert not np.isin(args_tail,args_loop).any(),\
        'loops in tails, there are {:d}'.format(np.count_nonzero(np.isin(args_tail,args_loop)))


        assert not np.isin(args_train,args_bridge).any(),\
        'bridge in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_bridge)))

        assert not np.isin(args_tail,args_bridge).any(),\
        'bridge in tails, there are {:d}'.format(np.count_nonzero(np.isin(args_tail,args_bridge)))

        assert args_train.shape[0] + args_tail.shape[0] +\
               args_loop.shape[0] + args_bridge.shape[0] \
               == np.count_nonzero(fads_chains) , 'Confomormations do not sum up correnctly,\
                   correct sum = {:d}, sum = {:d}'.format( args_ads_chain_atoms.shape[0],
                   args_train.shape[0] + args_tail.shape[0] +\
                   args_loop.shape[0] + args_bridge.shape[0]
                                                      )


        return ads_chains,args_train,args_tail,args_loop,args_bridge

    def connected_chunks(self,args):
        """Split a set of atom ids into connectivity-based chunks.

        Parameters
        ----------
        args : array_like of int
            Atom indices belonging to a subset of chains.

        Returns
        -------
        list of set
            Each set contains the atom indices belonging to one
            connected component within ``args`` according to
            ``self.neibs``.
        """
        #t0 = perf_counter()
        set_args = set(args)
        chunks = []
        #aold = -1

        while(len(set_args)>0):

            a = set_args.pop()
            set_args.add(a)
            #assert a != aold,'while loop stuck on argument = {}'.format(a)
            old_set_a = set()
            new_set_a = {a}

            new_neibs = new_set_a.copy()

            while new_set_a != old_set_a:
                old_set_a = new_set_a.copy()

                for j in new_neibs.copy():
                    new_neibs = set()
                    for neib in self.neibs[j]:
                        if neib in set_args:
                            new_set_a.add(neib)
                            if neib not in old_set_a:
                                new_neibs.add(neib)

            chunks.append(new_set_a)
            set_args.difference_update(new_set_a)

            #aold # (for debugging-combined with assertion above)
        #ass.print_time(perf_counter()-t0,'connected_chunks')
        return chunks

    def length_of_connected_chunks(self,args,coords,exclude_bonds=None):
        # this currently is suitable for linear chains in corse-grained or united atom represtation
        # if there is an all atom representation exclude_bonds must be used
        bonds = self.sorted_connectivity_keys
        f1 = np.isin(bonds[:,0],args)
        f2 = np.isin(bonds[:,1],args)
        f = np.logical_and(f1,f2)
        cb = bonds[f]
        coords1 = coords[cb[:,0]]
        coords2 = coords[cb[:,1]]
        r = coords2-coords1
        dists = np.sqrt(np.sum(r*r,axis=1))
        return dists.sum()
    ############### End of Conformation Calculation Supportive Functions #####

    ######## Main calculation Functions for structural properties ############

    def get_bins(self,binl,dmax,offset=0):
        bins =np.arange(offset,offset+dmax+binl, binl)
        return bins

    def calc_density_profile(self,binl,dmax,offset=0,
                             option='',mode='mass',flux=False,types=None):
        """Compute one-dimensional density profiles relative to the confining object.

        Parameters
        ----------
        binl : float
            Bin width for the distance coordinate.
        dmax : float
            Maximum distance up to which the profile is accumulated.
        offset : float, optional
            Shift applied to the distance axis (useful for changing
            the origin).
        option : {"", "pertype", "bymol", "2side", "conformations"}, optional
            Controls how the density is decomposed:

            * ``""``: total density only.
            * ``"pertype"``: contributions per atom type.
            * ``"bymol"``: contributions per molecule type.
            * ``"2side"``: separate profiles on both sides of the
              confining surface (for one-directional confinement).
            * ``"conformations"``: mass/number density split into
              conformational states (trains, tails, loops, bridges,
              free).
        mode : {"mass", "number", "massnumber"}, optional
            Density type to accumulate. ``"mass"`` gives mass density,
            ``"number"`` number density, and ``"massnumber"`` is used
            internally for the conformations mode.
        flux : bool, optional
            If ``True``, compute density fluctuations (variance) instead
            of the mean density by performing a second pass over the
            trajectory.
        types : sequence of str or None, optional
            Subset of atom types to include when ``option="pertype"``.

        Returns
        -------
        dict
            Dictionary containing the distance axis under ``"d"``,
            the total density ``"rho"`` and, depending on ``option``,
            per-type/per-molecule or per-conformation contributions.
            When ``flux`` is enabled, an additional key ``"rho_flux"``
            holds the density fluctuations.
        """
        t0 = perf_counter()

        #initialize
        ##############
        scale = 1.660539e-3 if mode == 'mass' else 1.0
        density_profile = dict()

        if dmax is None:
            NotImplemented
        bins  =   self.get_bins(binl,dmax,offset)
        nbins = len(bins)-1
        rho = np.zeros(nbins,dtype=float)
        mass = self.atom_mass


        if option =='':
            args = (nbins,bins,rho)
            contr=''
        elif option == 'pertype':
            if types is None:
                types = self.unique_atom_types
            elif type(types) is list or type(types) is tuple:
                types = types
            elif type(types) is str:
                types = [ types ]
            else:
                raise Exception('Wrong types')
            for ty in types:
                if ty not in self.unique_atom_types:
                    raise ValueError('{:s} is not regognized as an atom type'.format(ty))

            rho_per_atom_type = {t:np.zeros(nbins,dtype=float) for t in types }
            ftt = {t: t == self.at_types for t in rho_per_atom_type.keys() }

            args = (nbins,bins,rho,rho_per_atom_type,ftt)
            contr = '__pertype'

        elif option == 'bymol':
            rho_per_mol = {m:np.zeros(nbins,dtype=float) for m in np.unique(self.mol_names) }
            ftt = {m:  m == self.mol_names for m in rho_per_mol.keys() }
            args = (nbins,bins,rho,rho_per_mol,ftt)
            contr = '__bymol'

        elif option =='2side':
            rho_down = np.zeros(nbins,dtype=float)
            args =(nbins,bins,rho,rho_down)
            contr ='__2side'
        elif option =='conformations':
            confdens = {nm+k:np.zeros(nbins,dtype=float) for nm in ['m','n']
                        for k in ['rho','train','tail','loop','bridge','free']
                        }
            stats = { k : 0 for k in ['adschains','train','looptailbridge',
                                  'tail','loop','bridge']}
            dlayers = [(bins[i],bins[i+1]) for i in range(nbins)]
            args = (dlayers, confdens, stats)
            contr='__conformations'
            mode='massnumber'

        if mode =='mass':
            args =(*args,mass)


        if flux:
            if mode =='number':
                raise NotImplementedError('number density fluxations are not implemented.Please use mass density and then rescale to number')
            density_profile.update(self.calc_density_profile(binl,dmax,
                                 offset=offset,mode=mode,option=option))
            rho_mean = density_profile['rho'].copy()
            rho_mean/=scale
            args = (nbins,bins,rho,mass,rho_mean**2)
            func =  'mass'+'_density_profile'+'_flux'
            if mode !='mass' or option!='':
                logger.warning('mass mode and total density fluxuations are calculated')
        else:
            func =  mode+'_density_profile'+contr


        #############

        #calculate
        #############
        nframes = self.loop_trajectory(func, args)
        #############

        #post_process
        ##############

        if flux is not None and flux !=False:
            density_profile['rho_flux'] = rho*scale**2/nframes
        else:
            rho*=scale/nframes
            d_center = center_of_bins(bins)
            density_profile.update({'d':d_center-offset})

            density_profile.update({'rho':rho})
            if option =='pertype':
                for t,rhot in rho_per_atom_type.items():
                    density_profile[t] = rhot*scale/nframes
            elif option=='bymol':
                for m,rhom in rho_per_mol.items():
                    density_profile[m] = rhom*scale/nframes
            elif option =='2side':
                rho_down *=scale/nframes
                density_profile.update({'rho_up':rho,'rho_down':rho_down,'rho':0.5*(rho+rho_down)})
            elif option =='conformations':
                stats = {k+'_perc':v/self.npol/nframes if 'adschains'!=k else v/len(self.chain_args)/nframes
                         for k,v in stats.items()}
                dens= {k:v*scale/nframes for k,v in confdens.items()}

                for k in ['train','tail','loop','bridge','free']:
                    dens['mrho'] += dens['m'+k]
                    dens['nrho'] += dens['n'+k]

                dens['rho'] = dens['mrho']

                density_profile.update(dens)
                density_profile.update(stats)

        #############

        tf = perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return density_profile



    def calc_P2(self,topol_vector,binl,dmax,offset=0,option=''):
        """Compute the second Legendre order parameter :math:`P_2` versus distance.

        The bond vectors are defined by ``topol_vector`` and the
        reference direction is implied by the confinement type (e.g.
        normal to a planar surface).

        Parameters
        ----------
        topol_vector : int or sequence of str
            If an integer (2, 3, 4), selects 1–2, 1–3 or 1–4 bond
            vectors, respectively. If a sequence of atom types, the
            corresponding connectivity/angles/dihedrals are used to
            define the vectors.
        binl : float
            Bin width for the distance coordinate.
        dmax : float
            Maximum distance considered.
        offset : float, optional
            Shift applied to the distance axis.
        option : {"", "conformation", "conformations"}, optional
            If empty, a single ``P2(d)`` is returned. When set to
            ``"conformation"``/``"conformations"``, separate curves
            are computed for different polymer conformations (trains,
            loops, bridges, tails, free).

        Returns
        -------
        dict
            Dictionary with at least ``"d"`` (bin centres) and ``"P2"``
            / ``"P2(std)"``. When ``option`` selects conformations,
            additional keys such as ``"P2train"`` and
            ``"P2train(std)"`` are included.
        """

        t0 = perf_counter()

        bins  =   self.get_bins(binl,dmax,offset)

        dlayers=[]
        for i in range(0,len(bins)-1):
            dlayers.append((bins[i],bins[i+1]))
        d_center = np.array([0.5*(b[0]+b[1]) for b in dlayers])

        ids1, ids2 = self.find_vector_ids(topol_vector)
        nvectors = ids1.shape[0]
        logger.info('topol {}: {:d} vectors  '.format(topol_vector,nvectors))



        if option in ['conformation','conformations']:
            confs = ['train','loop','bridge','tail','free']
            costh_unv = {k:[[] for i in range(len(dlayers))] for k in confs }
            args = (ids1,ids2,dlayers,nvectors,costh_unv)

            s = '_conformation'
        elif option=='':
            s =''
            costh_unv = [[] for i in range(len(dlayers))]
            costh = np.empty(nvectors,dtype=float)
            args = (ids1,ids2,dlayers,costh,costh_unv)
        else:
            raise NotImplementedError('option "{}" not Implemented.\n Check your spelling when you give strings'.format(option))


        nframes = self.loop_trajectory('P2'+s, args)

        if option =='':

            costh2_mean = np.array([ np.array(c).mean() for c in costh_unv ])
            costh2_std  = np.array([ np.array(c).std()  for c in costh_unv ])

            s='P2'
            orientation = {'d':  d_center}
            orientation.update({s: 1.5*costh2_mean-0.5, s+'(std)' : 1.5*costh2_std-0.5 })

        elif option in ['conformation','conformations']:

            orientation = {'d':  d_center}

            for j in confs:
                costh2_mean = np.array([ np.array(c).mean() for c in costh_unv[j] ])
                costh2_std  = np.array([ np.array(c).std()  for c in costh_unv[j] ])
                s='P2'+j
                orientation.update({s: 1.5*costh2_mean-0.5, s+'(std)' : 1.5*costh2_std-0.5 })

            orientation.update(self.calc_P2(topol_vector,binl,dmax,offset))

        tf = perf_counter() - t0

        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return orientation

    def calc_particle_size(self):
        t0 = perf_counter()
        part_size = np.zeros(3)
        args = (part_size,)
        nframes = self.loop_trajectory('particle_size',args)
        part_size /= nframes
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return part_size

    def calc_dihedral_distribution(self,phi,filters=dict()):
        """Compute dihedral angle distributions under different filters.

        Parameters
        ----------
        phi : sequence of str
            Dihedral type key passed to :meth:`calc_dihedrals_t`.
        filters : dict, optional
            Mapping from filter name to filter definition; used to build
            sub-population distributions.

        Returns
        -------
        dict
            Dictionary mapping each filter name (and ``"system"`` for
            the full population) to an array of dihedral angles in
            degrees.
        """
        t0 = perf_counter()
        diht,ft = self.calc_dihedrals_t(phi,filters = filters)
        distrib = {k: [] for filt in filters.values() for k in filt }

        for k in distrib:
            for t,dih in diht.items():
                distrib[k].extend(dih[ft[k][t]])

        distrib['system'] = []
        for t,dih in diht.items():
            distrib['system'].extend(dih)

        for k in distrib:
            distrib[k] = np.array(distrib[k])*180/np.pi

        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return distrib

    def calc_chain_characteristics(self,binl,dmax,offset=0):
        """Chain shape characteristics as a function of distance from the surface.

        Parameters
        ----------
        binl : float
            Bin width for the chain centre-of-mass distance.
        dmax : float
            Maximum distance considered.
        offset : float, optional
            Shift applied to the distance axis.

        Returns
        -------
        dict
            Dictionary with distance axis ``"d"`` and several shape
            measures (e.g. ``"k2"``, ``"Rg2"``, ``"Ree2"``,
            ``"asph"``, ``"acyl"``) each accompanied by a ``"(std)"``
            entry with the corresponding standard deviation.
        """
        t0 = perf_counter()

        bins  =   self.get_bins(binl,dmax,offset)
        dlayers = [(bins[i],bins[i+1]) for i in range(len(bins)-1)]
        d_center = [0.5*(b[0]+b[1]) for b in dlayers]
        nl = len(dlayers)

        chars_strlist = ['k2','Rg2','Rg','Ree2','asph','acyl', 'Rgxx_plus_yy', 'Rgxx_plus_yy', 'Rgyy_plus_zz']
        chars = {k:[[] for d in dlayers] for k in chars_strlist }

        chain_args = self.chain_args

        # calculate
        args  = chain_args,dlayers,chars
        nframes = self.loop_trajectory('chain_characteristics', args)

        #post_process
        chain_chars = {'d':np.array(d_center)-offset}

        for k,v in chars.items():
            chain_chars[k] = np.array([ np.mean(chars[k][i]) for i in range(nl) ])
            chain_chars[k+'(std)'] = np.array([ np.std(chars[k][i]) for i in range(nl) ])

        tf= perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return chain_chars

    ###### End of Main calculation Functions for structural properties ########

    def calc_conformations_t(self,option=''):
        """Compute time evolution of conformational populations.

        Parameters
        ----------
        option : str, optional
            Suffix used to select specialised trajectory kernels.

        Returns
        -------
        dict
            Dictionary with a time axis under ``"time"`` and arrays for
            each conformation (e.g. ``"train"``, ``"loop"``,
            ``"tail"``, ``"bridge"``, ``"free"``) giving the number
            of atoms in that state as a function of time.
        """
        t0 = perf_counter()

        confs_t = dict()
        args = (confs_t,)
        nframes = self.loop_trajectory('confs_t'+option, args)
        confs_t = ass.rearrange_dict_keys(confs_t)

        conforms_t = {'time' : ass.numpy_keys( confs_t[ list(confs_t.keys())[0] ] ) }
        conforms_t.update({k: ass.numpy_values(c) for k,c in confs_t.items()})
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)

        return conforms_t


class Filter_Operations():
    def __init__(self):
        pass

    @staticmethod
    def calc_filters(filters, additional_info):
        """Evaluate a set of named filters.

        Parameters
        ----------
        filters : dict
            Mapping from filter name (string matching a ``Filters``
            static method) to filter parameters passed as the first
            argument of that method.
        additional_info: dict
            Additional keyword arguments required by the specific filter
            methods (e.g. vector ids, segmental_ids, coordinates).

        Returns
        -------
        dict
            Dictionary obtained by merging the results of all
            individual filter calls.
        """
        bool_data = dict()
        available_filters = ['x', 'y', 'z', 'space', 'conformations','system',
        'adsorption', 'bonds_to_non_train','bonds_to_train']

        for k in filters:
            if k not in available_filters:
                raise ValueError(f'"{k}" is not in the available filter options.\n Available filters: {available_filters}')

        for func, filt_values in filters.items():
            bool_data.update(  getattr(Filters, func)(filt_values, additional_info)  )

        return bool_data

    @staticmethod
    def filtLayers(layers,d, pref=''):
        """Convert a set of layer intervals into boolean masks.

        Parameters
        ----------
        layers : sequence
            Either a single ``(d_min, d_max)`` tuple or a sequence of
            such tuples.
        d : ndarray
            Distance values for each object to be filtered.
        pref : str,
            prefix on the name (key) of the filter
        Returns
        -------
        dict
            Mapping from each interval (tuple) to a boolean mask over
            ``d`` that is ``True`` for elements inside the interval.
        """

        if ass.iterable(layers):
            if ass.iterable(layers[0]):
                return {f'{pref} layer {j}' : filt_uplow(d , dl[0], dl[1]) for j,dl in enumerate(layers)}
            else:
                return {f'{pref} layer 0': filt_uplow(d , layers[0], layers[1])}
        return dict()
    @staticmethod
    def filt_bothEndsIn(ids1,ids2,args):
        """Return mask where both bond ends lie in a given set of ids."""
        f1 = np.isin(ids1, args)
        f2 = np.isin(ids2, args)
        return np.logical_and(f1,f2)


    @staticmethod
    def combine_filts(filts,filtc):
        """Combine two sets of filters in a Cartesian product.

        Parameters
        ----------
        filts, filtc : dict
            Dictionaries of boolean masks with identical key sets.

        Returns
        -------
        dict
            Nested filters ``filt_sc[(s, c)]`` combining spatial and
            conformational selections.
        """
        filt_sc = dict()
        for s,fs in filts.items():
            for c,fc in filtc.items():
                filt_sc[(s,c)] = {k:np.logical_and(fs[k],fc[k]) for k in fs.keys()}
        return filt_sc


    @staticmethod
    def get_ads_degree(obj_an, segmental_ids):
        """Return adsorption degree and adsorption flags for each segment.

        Parameters
        ----------
        obj_an: analysis object providing ``get_filt_train()``.
        segmental_ids: (Nsegments, seg_size) int array


        Returns
        -------
        degree : ndarray of float, shape (Nsegments,)
            Fraction of adsorbed atoms per segment.
        ads : ndarray of bool, shape (Nsegments,)
            ``True`` for segments with at least one adsorbed atom.
        """

        ftrain, _image_trains = obj_an.get_filt_train()

        nseg = segmental_ids.shape[0]
        ads = np.empty(nseg, dtype=bool)
        degree = np.empty(nseg, dtype=float)

        for i, seg_ids in enumerate(segmental_ids):
            f = ftrain[seg_ids]
            ads[i] = f.any()
            degree[i] = np.count_nonzero(f) / f.shape[0]

        return degree, ads


class Filters():
    """Collection of helper functions that build boolean filters.

    The static methods in this class are used together with
    :class:`Analysis_Confined` or `Analysis to generate boolean masks ("filters")
    that classify atoms, segments or chains according to spatial
    position, conformation, or distance from special reference sets
    (trains, end groups, free segments, etc.).
    """
    def __init__(self):
        pass

    @staticmethod
    def x(layers, additional_info):
        """Filter bonds based on their :math:`x`-distance.
        Parameters
        ----------
        layers : sequence
            One or more ``(d_min, d_max)`` intervals.
        additional_info : dict
            Additional information required for this filter.
            Expected keys are:
            - "relative_coords": (N,3) float array
        Returns
        -------
        dict
            Mapping from each layer interval to a boolean array over
            vectors indicating membership.
        """
        d = np.abs(additional_info['relative_coords'][:,0])
        f1 = Filter_Operations.filtLayers(layers, d, pref='x')
        return f1

    @staticmethod
    def y(layers, additional_info):
        """Filter bonds based on their :math:`y`-distance.
        Parameters
        ----------
        layers : sequence
            One or more ``(d_min, d_max)`` intervals.
        additional_info : dict
            Additional information required for this filter.
            Expected keys are:
            - "relative_coords": (N,3) float array
        Returns
        -------
        dict
            Mapping from each layer interval to a boolean array over
            vectors indicating membership.
        """
        d = np.abs(additional_info['relative_coords'][:,1])
        f1 = Filter_Operations.filtLayers(layers, d, pref='y')
        return f1

    @staticmethod
    def z(layers,  additional_info):
        """Filter bonds based on their :math:`z`-distance.
        Parameters
        ----------
        layers : sequence
            One or more ``(d_min, d_max)`` intervals.
        additional_info : dict
            Additional information required for this filter.
            Expected keys are:
            - "relative_coords": (N,3) float array
        Returns
        -------
        dict
            Mapping from each layer interval to a boolean array over
            vectors indicating membership.
        """
        d = np.abs(additional_info['relative_coords'][:,2])
        f1 = Filter_Operations.filtLayers(layers, d, pref='z')
        return f1

    @staticmethod
    def space(layers, additional_info):
        """Filter bonds based on their :math:`3d`-distance.
        Parameters
        ----------
        layers : sequence
            One or more ``(d_min, d_max)`` intervals.
        additional_info : dict
            Additional information required for this filter.
            Expected keys are:
            - "relative_coords": (N,3) float array
        Returns
        -------
        dict
            Mapping from each layer interval to a boolean array over
            vectors indicating membership.
        """
        r = additional_info['relative_coords']
        d = np.sqrt(np.sum(r*r, axis=1))
        f1 = Filter_Operations.filtLayers(layers, d, pref='space')
        return f1


    @staticmethod
    def bonds_to_non_train(bondlayers, additional_info):
        """Filter train segments by bond distance from non-train segments.

        Train Segments whose end points are closer (in terms of number of
        bonds) to non-train segments than the specified intervals are
        selected.

        Parameters
        ----------
        bondlayers : sequence of tuple
            Intervals in bond-count space.
        additional_info : dict
            Additional information required for this filter.
            Expected keys are:

            - ``"ids1"``: (N,) int array
            - ``"ids2"``: (N,) int array
            - ``"obj"``: analysis object providing ``conformations()`` and
              ``nbonds_of_ids_from_other_ids(ids, other_ids)``.

        Returns
        -------
        dict
            Mapping from each interval to a boolean mask over bonds.
        """

        ids1 = additional_info['ids1']
        ids2 = additional_info['ids2']
        obj_an = additional_info['obj']
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = obj_an.conformations()

        args_rest_train = np.concatenate( (args_tail,args_loop,args_bridge ) )
        nbonds1 = obj_an.nbonds_of_ids_from_other_ids(ids1,args_rest_train)
        nbonds2 = obj_an.nbonds_of_ids_from_other_ids(ids2,args_rest_train)
        nbonds = np.minimum(nbonds1,nbonds2)

        return Filter_Operations.filtLayers(bondlayers,nbonds, pref= 'btnt')

    @staticmethod
    def bonds_to_train(bondlayers, additional_info):
        """Filter vectors by bond-distance from train segments.

        Parameters
        ----------
        bondlayers : sequence of tuple
            Intervals in bond-count space.
        additional_info : dict
            Additional information required for this filter.

            Expected keys are:

            - ``"ids1"``: (N,) int array
            - ``"ids2"``: (N,) int array
            - ``"obj"``: analysis object providing ``conformations()`` and
              ``nbonds_of_ids_from_other_ids(ids, other_ids)``.

        Returns
        -------
        dict
            Mapping from each interval to a boolean mask over vectors.
        """

        ids1 = additional_info['ids1']
        ids2 = additional_info['ids2']
        obj_an = additional_info['obj']

        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = obj_an.conformations()

        nbonds1 = obj_an.nbonds_of_ids_from_other_ids(ids1, args_train)
        nbonds2 = obj_an.nbonds_of_ids_from_other_ids(ids2, args_train)
        nbonds = np.minimum(nbonds1, nbonds2)

        return Filter_Operations.filtLayers(bondlayers, nbonds, pref= 'btt')

    @staticmethod
    def conformations(fconfs, additional_info):
        """Build simple filters selecting vectors inside conformational sets.

        Parameters
        ----------
        fconfs : sequence of str
            Conformation labels (e.g. ``"train"``, ``"tail"``, ``"loop"``,
            ``"bridge"``, ``"free"``).
        additional_info : dict
            Additional information required for this filter.

            Expected keys are:

            - ``"ids1"``: (N,) int array
            - ``"ids2"``: (N,) int array
            - ``"obj"``: analysis object providing ``conformations()`` and
              the atom id population via ``all_args``.

        Returns
        -------
        dict
            Mapping from conformation label to a boolean mask over vectors where
            both ends lie in the corresponding atom set.
        """

        ids1 = additional_info['ids1']
        ids2 = additional_info['ids2']
        obj_an = additional_info['obj']

        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = obj_an.conformations()

        all_not_free = np.concatenate((args_train, args_tail, args_loop, args_bridge))
        all_args = np.array( list( set(ids1) | set(ids2) ) )
        args_free = all_args[np.logical_not(np.isin(all_args, all_not_free))]

        conf_map = {
            'train': args_train,
            'tail': args_tail,
            'loop': args_loop,
            'bridge': args_bridge,
            'free': args_free,
        }

        filt = dict()
        for conf in fconfs:
            if conf not in conf_map:
                raise ValueError(f'Unknown conformation "{conf}" in filter values. Available: {list(conf_map.keys())}')
            filt[conf] = Filter_Operations.filt_bothEndsIn(ids1, ids2, conf_map[conf])
        return filt

    @staticmethod
    def conformationDistribution(fconfs, additional_info):
        """Compute size distributions for connected conformational chunks.

        Parameters
        ----------
        fconfs : dict
            Mapping from conformation name (``"train"``, ``"loop"``,
            ``"tail"``, ``"bridge"``, ``"free"``) to lists of size
            intervals.
        additional_info : dict
            Additional information required for this filter.
            Expected keys are:

            - ``"ids1"``: (N,) int array
            - ``"ids2"``: (N,) int array
            - ``"obj"``: analysis object providing ``conformations()`` and
              ``connected_chunks(args)``.

        Returns
        -------
        dict
            Contains distribution arrays (``"<conf>:distr"``) and
            per-interval boolean vector masks (``"<conf>:chunk interval <k>"``).
        """

        ids1 = additional_info['ids1']
        ids2 = additional_info['ids2']
        obj_an = additional_info['obj']

        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = obj_an.conformations()

        all_not_free = np.concatenate((args_train, args_tail, args_loop, args_bridge))
        all_args = np.array(list(set(ids1) | set(ids2)))
        args_free = all_args[np.logical_not(np.isin(all_args, all_not_free))]

        conf_map = {
            'train': args_train,
            'tail': args_tail,
            'loop': args_loop,
            'bridge': args_bridge,
            'free': args_free,
        }

        filt = dict()
        for conf, intervals in fconfs.items():
            if conf not in conf_map:
                raise ValueError(f'Unknown conformation "{conf}" in filter values. Available: {list(conf_map.keys())}')

            args = conf_map[conf]
            connected_chunks = obj_an.connected_chunks(args)
            sizes = np.array([len(chunk) for chunk in connected_chunks])

            filt[f'{conf}:distr'] = sizes

            for jk, inter in enumerate(intervals):
                chunk_int = set()
                for chunk, size in zip(connected_chunks, sizes):
                    if inter[0] <= size < inter[1]:
                        chunk_int = chunk_int | chunk

                args_chunk = np.array(list(chunk_int), dtype=int)
                filt[f'{conf}:chunk interval {jk}'] = Filter_Operations.filt_bothEndsIn(ids1, ids2, args_chunk)

        return filt



    @staticmethod
    def adsorption(ads_degrees, additional_info):
        """Build filters based on adsorption degree for any segment set.

        Parameters
        ----------
        ads_degree : sequence of tuple
            Intervals in adsorption degree used to group segments.
        additional_info : dict
            Expected keys are:

            - ``"segmental_ids"``: (Nsegments, seg_size) int array
            - ``"obj"``: analysis object providing ``get_filt_train()``.

        Returns
        -------
        dict
            Per-interval boolean masks over segments together with:
            - ``"ads"``: segments with at least one adsorbed atom
            - ``"free"``: segments with no adsorbed atoms
            - ``"degree"``: adsorption degree per segment
        """

        degree, ads = Filter_Operations.get_ads_degree(
            additional_info['obj'] ,additional_info['segmental_ids'])
        filt_ads = dict()
        filt_ads.update(Filter_Operations.filtLayers(ads_degrees, degree, pref='adsorption'))
        filt_ads.update({'ads': ads, 'free': np.logical_not(ads), 'degree': degree})
        return filt_ads

class coreFunctions():
    """Low-level kernels used inside trajectory loops.

    Methods in this class are thin wrappers around inner kernels that
    populate preallocated arrays or dictionaries while looping over
    frames. High-level analysis methods (typically starting with
    ``calc_``) allocate the data structures and then delegate the
    per-frame work to these helpers via :meth:`Analysis.loop_trajectory`.
    """
    def __init__():
        pass

    @staticmethod
    def stress_per_atom_t(self,filters,atomstress,filt_per_t):
        """Accumulate per-atom stress tensors for the current frame.

        The stress tensor combines kinetic (velocity) and virial
        (force-position) contributions and is stored in ``atomstress``
        indexed by the current time key.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Object providing coordinates, velocities, forces and box.
        filters : dict
            Filter specification passed to :class:`Filters` to generate
            ``filt_per_t``.
        atomstress : dict
            Dictionary that will be filled with stress arrays of shape
            ``(Natoms, 3, 3)`` per frame.
        filt_per_t : dict
            Dictionary that will receive the boolean filters per frame.
        """

        frame = self.current_frame

        coords = self.get_coords(frame)
        ids = np.arange(0,coords.shape[0],1,dtype=int)

        v = self.get_velocities(frame)
        f = self.get_forces(frame)
        m = self.atom_mass

        # Build per-atom 3x3 stress tensor from kinetic (vv/m) and virial (r·f) contributions
        vel_contr = np.array( [v[:,i]*v[:,j]/m for i in range(3) for j in range(3)] )
        virial = np.array([coords[:,i]*f[:,j] for i in range(3) for j in range(3)])
        stress = vel_contr + virial
        stress = stress.reshape(stress.shape[-1::-1])

        key = self.get_key()
        atomstress[key] = stress

        filters_info = self.get_filters_info(filters, ids1=ids, ids2=ids)

        filt_per_t[key] = Filter_Operations.calc_filters(filters, filters_info)
        return
    @staticmethod
    def minmax_size(self,size):
        """Update running box-extents estimate for the system.

        Adds the per-axis span of the coordinates in the current frame
        to ``size``. Typically used to estimate overall system
        dimensions over a trajectory.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)
        size+=coords.max(axis=0) - coords.min(axis=0)
        return
    @staticmethod
    def Sq(self,q,Sq,ids=None):
        """Accumulate the static structure factor :math:`S(q)`.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and box.
        q : ndarray
            Magnitudes of the scattering vectors.
        Sq : ndarray
            Output array that is updated in place with ``S(q)``.
        ids : array_like of int, optional
            Optional subset of atom indices; if given only those atoms
            contribute.
        """
        frame = self.current_frame
        coords =self.get_coords(frame)

        box = self.get_box(frame)

        if ids is not None:
            coords = coords[ids]
        n = coords.shape[0]
        npairs = int(n*(n-1)/2)  # number of unique atom pairs
        v = np.empty((npairs,3),dtype=float)
        pair_vects(coords,box,v)  # all pair displacement vectors with minimum-image convention
        self.v = v
        numba_Sq2(n,v,q,Sq)
        return

    @staticmethod
    def atomic_coordination(self,maxdist,args1,args2,coordination):
        """Compute atomic coordination numbers for a pair of atom sets.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and box.
        maxdist : float
            Cutoff distance for neighbours.
        args1, args2 : array_like of int
            Atom indices defining central and neighbour atoms.
        coordination : ndarray
            Output array updated in place with coordination numbers.
        """

        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords1 = coords[args1]
        coords2 = coords[args2]
        numba_coordination(coords1,coords2,box,maxdist,coordination)
        return

    @staticmethod
    def particle_size(self,part_size):
        """Accumulate bounding-box size of the confining particle.

        The confining particle is first translated to the box centre;
        the difference between max and min coordinates along each axis
        is added to ``part_size``.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords,box)
        part_coords = coords[self.particle_filt]
        part_s = part_coords.max(axis = 0 ) - part_coords.min(axis = 0 )
        #logger.debug('frame = {:d} --> part size = {} '.format(frame,part_s))
        part_size += part_s
        return


    @staticmethod
    def box_mean(self,box):
        """Accumulate the sum of box vectors over frames."""
        frame = self.current_frame
        box+=self.get_box(frame)
        return

    @staticmethod
    def box_var(self,box_var,box_mean_squared):
        """Accumulate variance contribution for box fluctuations."""
        frame = self.current_frame
        box_var += self.get_box(frame)**2 - box_mean_squared
        return

    @staticmethod
    def vector_correlations(self,vec_t,filt_t,bk0,bk1,correlation):
        """Collect instantaneous vector–vector correlations.

        Uses precomputed vectors and filters to build angle
        distributions for selected sub-populations.
        """


        timekey = self.get_key()
        vec = vec_t[timekey]  # precomputed vectors for this frame
        for kf in correlation:
            f = filt_t[kf][timekey]
            for k in correlation[kf]:
              #  t0 = perf_counter()
                b0 = bk0[k]
                b1 = bk1[k]
                f01 = np.logical_and(f[b0],f[b1])
                v0 = vec[b0][f01]
                v1 = vec[b1][f01]

                try:
                    costh = costh__parallelkernel(v0,v1)  # cos(theta) for all selected vector pairs
                    correlation[kf][k].append( costh )
                except ZeroDivisionError:
                    pass
        return
    @staticmethod
    def total_dipole_moment(self,dipoles_t,q=None):
        """Accumulate the total dipole moment of the system.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and partial charges.
        dipoles_t : dict
            Dictionary updated with the total dipole vector per frame.
        q : ndarray or None, optional
            Optional projection/scaling vector for the dipole.
        """

        frame = self.current_frame
        coords = self.get_coords(frame)

        key = self.get_key()

        pc = self.partial_charge

        dipoles = np.sum(pc*coords,axis=0).reshape((1,3))  # sum q*r over all atoms
        if q is not None:
            dipoles = np.sum(q*dipoles)*q
        dipoles_t[key] = dipoles

        return

    @staticmethod
    def segmental_dipole_moment(self,filters,ids1,ids2,
                                segmental_ids,dipoles_t,filt_per_t):
        """Compute dipole moments for user-defined segments.

        Segmental dipoles are accumulated in ``dipoles_t`` and
        corresponding filters are stored in ``filt_per_t``.
        """

        frame = self.current_frame
        coords = self.get_coords(frame)

        key = self.get_key()

        n = segmental_ids.shape[0]

        pc = self.atom_charge.reshape(self.natoms,1)

        relc = coords[segmental_ids] - self.segs_CM(coords, segmental_ids)

        if isinstance(segmental_ids, np.ndarray) and segmental_ids.ndim == 2:
            dipoles = np.sum(pc[segmental_ids] * relc, axis=1)
        else:
            dipoles = np.empty((n,3),dtype=float)
            for i,sa in enumerate(segmental_ids):
                dipoles[i] = np.sum(pc[sa]*relc[i],axis=0)

        dipoles_t[key] = dipoles


        filters_info = self.get_filters_info(filters, segmental_ids = segmental_ids)

        filt_per_t[key] = Filter_Operations.calc_filters(filters, filters_info)

        return

    @staticmethod
    def mass_density_profile(self,nbins,bins,
                                  rho,mass):
        """Kernel for mass density profiles relative to the confining object.

        Parameters
        ----------
        self : Analysis_Confined
            Provides frame coordinates, box and ``volfun``.
        nbins : int
            Number of distance bins.
        bins : ndarray
            Bin edges of length ``nbins + 1``.
        rho : ndarray
            Output array of length ``nbins`` incremented in place with
            mass density contributions.
        mass : ndarray
            Per-atom masses used to weight the density.
        """
        frame = self.current_frame

        coords,box,d = self.get_frame_basics(frame)

        # Loop over distance bins: compute shell volume and add mass/volume contribution
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.sum(mass[fin_bin])/vol_bin
        return

    @staticmethod
    def mass_density_profile__bymol(self,nbins,bins,
                                  rho,rho_per_mol,ftt,mass):
        """Kernel for mass density decomposed by molecule type.

        Parameters
        ----------
        self : Analysis_Confined
            Provides geometry and volume information.
        nbins : int
            Number of distance bins.
        bins : ndarray
            Bin edges of length ``nbins + 1``.
        rho : ndarray
            Total mass density profile, updated in place.
        rho_per_mol : dict
            Mapping from molecule label to density array of length
            ``nbins`` updated in place.
        ftt : dict
            Boolean masks selecting atoms belonging to each molecule
            type.
        mass : ndarray
            Per-atom masses.
        """

        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)

        # Loop over distance bins and molecule types: accumulate per-molecule mass/volume
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += numba_sum(mass[fin_bin])/vol_bin

            for t in rho_per_mol.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_mol[t][i] += numba_sum(mass[ft])/vol_bin
        return

    @staticmethod
    def mass_density_profile__pertype(self,nbins,bins,
                                  rho,rho_per_atom_type,ftt,mass):
        """Kernel for mass density decomposed by atom type.

        Parameters
        ----------
        self : Analysis_Confined
            Provides geometry and volume information.
        nbins : int
            Number of distance bins.
        bins : ndarray
            Bin edges of length ``nbins + 1``.
        rho : ndarray
            Total mass density profile, updated in place.
        rho_per_atom_type : dict
            Mapping from atom type label to density array of length
            ``nbins`` updated in place.
        ftt : dict
            Boolean masks per type, typically indexed as ``ftt[type]``.
        mass : ndarray
            Per-atom masses.
        """

        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)

        # Loop over bins and atom types: reuse mass_bin and slice per-type masks in that bin
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            mass_bin = mass[fin_bin]
            rho[i] += numba_sum(mass_bin)/vol_bin

            for t in rho_per_atom_type.keys():
                #ft = np.logical_and( fin_bin,ftt[t])
                ft = ftt[t][fin_bin]
                rho_per_atom_type[t][i] += numba_sum(mass_bin[ft])/vol_bin
        return



    @staticmethod
    def mass_density_profile_flux(self,nbins,bins,
                                  rho,mass,rho_mean_sq):
        """Kernel for mass-density fluctuations (variance) profiles.

        Parameters
        ----------
        self : Analysis_Confined
            Provides geometry and volume information.
        nbins : int
            Number of distance bins.
        bins : ndarray
            Bin edges of length ``nbins + 1``.
        rho : ndarray
            Output array of length ``nbins`` storing variance-like
            contributions ``\langle \rho^2 \rangle - (\rho_\text{mean})^2``.
        mass : ndarray
            Per-atom masses.
        rho_mean_sq : ndarray
            Pre-computed square of the mean density per bin.
        """
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)

        # For each bin, accumulate <rho^2> - rho_mean_sq from instantaneous mass/volume
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += (np.sum(mass[fin_bin])/vol_bin)**2-rho_mean_sq[i]
        return

    @staticmethod
    def number_density_profile__2side(self,nbins,bins,
                                  rho_up,rho_down):
        """Kernel for number density on both sides of a symmetric interface.

        Parameters
        ----------
        self : Analysis_Confined
            Provides translated coordinates, confinement type and
            ``volfun``.
        nbins : int
            Number of distance bins per side.
        bins : ndarray
            Bin edges (positive distances) of length ``nbins + 1``.
        rho_up, rho_down : ndarray
            Arrays of length ``nbins`` updated in place with number
            densities on the ``+`` and ``-`` sides respectively.
        """
        frame = self.current_frame
        coords = self.translated_coords(frame)

        cs = self.get_particle_cm(coords)

        dfun = getattr(Distance_Functions,self.conftype +'__2side')
        d = dfun(self,coords,cs)
         # needed because in volfun the volume of each bin is multiplied by 2
        # For each |z| layer accumulate number density separately above and below the interface

        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])*0.5
            fin_bin_up =   filt_uplow(d,bins[i],bins[i+1])
            fin_bin_down = filt_uplow(d,-bins[i+1],-bins[i])
            rho_up[i] += np.count_nonzero(fin_bin_up)/vol_bin
            rho_down[i] += np.count_nonzero(fin_bin_down)/vol_bin

        return


    @staticmethod
    def mass_density_profile__2side(self,nbins,bins,
                                  rho_up,rho_down,mass):
        """Kernel for mass density on both sides of a symmetric interface.

        Parameters
        ----------
        self : Analysis_Confined
            Provides translated coordinates, confinement type and
            ``volfun``.
        nbins : int
            Number of distance bins per side.
        bins : ndarray
            Bin edges (positive distances) of length ``nbins + 1``.
        rho_up, rho_down : ndarray
            Arrays of length ``nbins`` updated in place with mass
            densities on the ``+`` and ``-`` sides respectively.
        mass : ndarray
            Per-atom masses.
        """
        frame = self.current_frame
        coords = self.translated_coords(frame)

        cs = self.get_particle_cm(coords)

        dfun = getattr(Distance_Functions,self.conftype +'__2side')
        d = dfun(self,coords,cs)
         # needed because in volfun the volume of each bin is multiplied by 2
        # As above, but with mass/volume instead of counts

        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])*0.5
            fin_bin_up =   filt_uplow(d,bins[i],bins[i+1])
            fin_bin_down = filt_uplow(d,-bins[i+1],-bins[i])
            rho_up[i] += np.sum(mass[fin_bin_up])/vol_bin
            rho_down[i] += np.sum(mass[fin_bin_down])/vol_bin

        return

    @staticmethod
    def number_density_profile__pertype(self,nbins,bins,
                                  rho,rho_per_atom_type,ftt):
        """Kernel for number density decomposed by atom type.

        Parameters
        ----------
        self : Analysis_Confined
            Provides geometry and volume information.
        nbins : int
            Number of distance bins.
        bins : ndarray
            Bin edges of length ``nbins + 1``.
        rho : ndarray
            Total number density profile, updated in place.
        rho_per_atom_type : dict
            Mapping from atom type label to number-density arrays of
            length ``nbins`` updated in place.
        ftt : dict
            Boolean masks per atom type.
        """
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)

        # Loop over bins and atom types: counts per bin and per type
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin

            for t in rho_per_atom_type.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_atom_type[t][i] += np.count_nonzero(ft)/vol_bin

    @staticmethod
    def number_density_profile(self,nbins,bins,rho):
        """Kernel for total number density profiles.

        Parameters
        ----------
        self : Analysis_Confined
            Provides geometry and volume information.
        nbins : int
            Number of distance bins.
        bins : ndarray
            Bin edges of length ``nbins + 1``.
        rho : ndarray
            Output array of length ``nbins`` incremented in place with
            number density contributions.
        """
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)

        # Standard 1D number-density histogram over distance from the confining object
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin
        return

    @staticmethod
    def massnumber_density_profile__conformations(self,dlayers,dens,stats):
        """Update mass/number density split by conformational state.

        Parameters
        ----------
        self : Analysis_Confined
            Provides conformational classification and volume function.
        dlayers : sequence of tuple
            Distance intervals ``(d_min, d_max)`` over which densities
            are accumulated.
        dens : dict
            Dictionary of arrays (per layer) for mass and number
            densities of trains, tails, loops, bridges and free
            segments (e.g. ``"ntrain"``, ``"mtrain"`` etc.). Updated
            in place.
        stats : dict
            Global counters for conformational populations, updated in
            place by :func:`conformation_stats`.
        """

        #1) ads_chains, trains,tails,loops,bridges
        ads_chains, args_train, args_tail, args_loop, args_bridge = self.conformations()

        #check_occurances(np.concatenate((args_train,args_tail,args_bridge,args_loop)))

        # Accumulate layer-resolved densities and global statistics for each conformation class
        coreFunctions.conformation_dens(self, dlayers, dens,ads_chains,
                                           args_train, args_tail,
                                           args_loop, args_bridge)

        coreFunctions.conformation_stats(stats,ads_chains, args_train, args_tail,
                             args_loop, args_bridge)
        return

    @staticmethod
    def get_args_free(self,ads_chains):
        """Return indices of polymer atoms that are not adsorbed.

        Parameters
        ----------
        self : Analysis_Confined
            Provides ``polymer_filt`` and ``mol_ids``.
        ads_chains : ndarray of int
            Indices of adsorbed chains.

        Returns
        -------
        ndarray of int
            Atom indices belonging to polymer chains that are not part
            of ``ads_chains``.
        """
        fp = self.polymer_filt
        fnotin = np.logical_not(np.isin(self.mol_ids,ads_chains))
        f = np.logical_and(fp,fnotin)
        args_free = np.where(f)[0]
        return args_free

    @staticmethod
    def conformation_dens(self, dlayers,dens,ads_chains,
                             args_train, args_tail,
                             args_loop, args_bridge):
        """Accumulate conformational densities (number and mass).

        Parameters
        ----------
        self : Analysis_Confined
            Provides geometry, masses and confining volume function.
        dlayers : sequence of tuple
            Distance intervals ``(d_min, d_max)`` over which densities
            are accumulated.
        dens : dict
            Dictionary of arrays for number/mass densities per
            conformation, updated in place.
        ads_chains : ndarray of int
            Indices of adsorbed chains.
        args_train, args_tail, args_loop, args_bridge : ndarray of int
            Atom indices for each conformational category.
        """

        coords,box,d = self.get_frame_basics(self.current_frame)

        args_free = coreFunctions.get_args_free(self,ads_chains)

        d_tail = d[args_tail]
        d_loop = d[args_loop]
        d_bridge = d[args_bridge]
        d_free = d[args_free]
        d_train = d[args_train]
        for l,dl in enumerate(dlayers):  # per-layer selection and accumulation
            args_tl = args_tail[filt_uplow(d_tail, dl[0], dl[1])]
            args_lp = args_loop[filt_uplow(d_loop, dl[0], dl[1])]
            args_br =  args_bridge[filt_uplow(d_bridge, dl[0], dl[1])]
            args_fr = args_free[filt_uplow(d_free,dl[0],dl[1])]
            args_tr = args_train[filt_uplow(d_train,dl[0],dl[1])]

            vol_bin = self.volfun(self,dl[0],dl[1])

            dens['ntrain'][l] += args_tr.shape[0]/vol_bin
            dens['ntail'][l] += args_tl.shape[0]/vol_bin
            dens['nloop'][l] += args_lp.shape[0]/vol_bin
            dens['nbridge'][l] += args_br.shape[0]/vol_bin
            dens['nfree'][l] += args_fr.shape[0]/vol_bin

            dens['mtrain'][l] += np.sum(self.atom_mass[args_tr])/vol_bin
            dens['mtail'][l] += np.sum(self.atom_mass[args_tl])/vol_bin
            dens['mloop'][l] += np.sum(self.atom_mass[args_lp])/vol_bin
            dens['mbridge'][l] += np.sum(self.atom_mass[args_br])/vol_bin
            dens['mfree'][l] += np.sum(self.atom_mass[args_fr])/vol_bin
        return

    @staticmethod
    def conformation_stats(stats,ads_chains, args_train, args_tail,
                             args_loop, args_bridge):
        """Update global statistics for conformational populations.

        Parameters
        ----------
        stats : dict
            Dictionary of scalar counters (e.g. ``"train"``,
            ``"adschains"``, ``"looptailbridge"``) that is updated in
            place.
        ads_chains : ndarray of int
            Indices of adsorbed chains.
        args_train, args_tail, args_loop, args_bridge : ndarray of int
            Atom indices for each conformational category.
        """
        stats['train'] += args_train.shape[0]
        stats['adschains'] += ads_chains.shape[0]
        stats['looptailbridge'] += (args_loop.shape[0]+args_tail.shape[0]+args_bridge.shape[0])
        stats['tail'] += args_tail.shape[0]
        stats['loop'] += args_loop.shape[0]
        stats['bridge'] += args_bridge.shape[0]
        return



    @staticmethod
    def P2(self,ids1,ids2,dlayers,costh,costh_unv):
        """Kernel for computing :math:`P_2` vs distance for all bonds.

        Parameters
        ----------
        self : Analysis_Confined
            Provides distances to the confining object and unit
            vectors.
        ids1, ids2 : ndarray of int
            Atom indices defining bond vectors.
        dlayers : sequence of tuple
            Distance intervals ``(d_min, d_max)``.
        costh : ndarray
            Preallocated array for :math:`\cos(\theta)` values.
        costh_unv : list of list
            Container per distance layer into which :math:`\cos^2` values
            are appended.
        """
        frame = self.current_frame
        #1) coords
        coords = self.get_coords(frame)

        #2) calc_particle_cm
        cs = self.get_particle_cm(coords)

        r1 = coords[ids1]; r2 = coords[ids2]

        rm = 0.5*(r1+r2)

        d = self.get_distance_from_particle(rm)
        uv = self.unit_vectorFun(self,rm,cs)

        costhsquare__kernel(costh,r2-r1,uv)

        for i,dl in enumerate(dlayers):  # bin P2 contributions by distance layer
            filt = filt_uplow(d, dl[0], dl[1])
            costh_unv[i].extend(costh[filt])

        return
    @staticmethod
    def P2_conformation(self,ids1,ids2,dlayers,
                        nvectors,costh_unv):
        """Kernel for :math:`P_2` resolved by polymer conformation.

        Parameters
        ----------
        self : Analysis_Confined
            Provides conformations, geometry and unit vectors.
        ids1, ids2 : ndarray of int
            Atom indices defining bond vectors.
        dlayers : sequence of tuple
            Distance intervals ``(d_min, d_max)``.
        nvectors : int
            Total number of bond vectors.
        costh_unv : dict
            Mapping from conformation label (e.g. ``"train"``) to
            lists-of-lists that receive :math:`\cos^2` values per
            distance layer.
        """
        frame = self.current_frame
        #1) coords
        coords = self.get_coords(frame)
        box =self.get_box(frame)

        #2) calc_particle_cm
        cs = self.get_particle_cm(coords)
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations()

        args_free = coreFunctions.get_args_free(self,ads_chains)

        for j in costh_unv:  # loop over conformation classes (train/tail/loop/bridge/free)
            args = locals()['args_'+j]
            filt_ids = Filters.filt_bothEndsIn(ids1,ids2,args)
            r1 = coords[ids1[filt_ids]]; r2 = coords[ids2[filt_ids]]


            rm = 0.5*(r1+r2)

            d = self.get_distance_from_particle(rm)
            uv = self.unit_vectorFun(self,rm,cs)

            costh = np.empty(d.shape[0],dtype=float)
            costhsquare__kernel(costh,r2-r1,uv)

            for i,dl in enumerate(dlayers):  # bin contributions per distance layer for this class
                filt = filt_uplow(d, dl[0], dl[1])
                costh_unv[j][i].extend(costh[filt])
        return
    @staticmethod
    def Rg__permol(self,rgdict):
        """Accumulate radius of gyration per molecule type for one frame.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates, masses and molecule labels.
        rgdict : dict
            Dictionary with keys ``"<mol>_Rg"`` and ``"<mol>_Rgstd"``
            updated in place with running sums over frames.
        """
        # Rg per molecule name: loop over molecule labels and aggregate chain-based Rg values
        coords = self.get_coords(self.current_frame)
        for m in self.molecules:
            rgframe = []
            for a in self.chain_args.values():
                if m != self.mol_names[a[0]]:
                    if (m == self.mol_names[a]).any():
                        raise Exception('Something is wrong with the arguments. The names do not correspond to the right ids.')
                    continue
                c = coords[a]
                mass = self.atom_mass[a]
                cm = CM(c,mass)
                r = c - cm
                dsq = np.sum(r*r,axis=1)
                rgframe.append( np.average(dsq,weights=mass))

            rgdict[m+'_Rg'] += np.mean(rgframe)
            rgdict[m+'_Rgstd'] += np.std(rgframe)
        return
    @staticmethod
    def Rg(self,rgdict):
        """Accumulate system-averaged radius of gyration for one frame.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates, chain definitions and masses.
        rgdict : dict
            Dictionary with keys ``"Rg"`` and ``"Rgstd"`` updated in
            place with running sums over frames.
        """
        coords = self.get_coords(self.current_frame)
        rgframe = []  # per-chain Rg values in this frame
        for a in self.chain_args.values():
            c = coords[a]
            mass = self.atom_mass[a]
            cm = CM(c,mass)
            r = c - cm
            dsq = np.sum(r*r,axis=1)
            rgframe.append( np.average(dsq,weights=mass))

        rgdict['Rg'] += np.mean(rgframe)
        rgdict['Rgstd'] += np.std(rgframe)
        return

    @staticmethod
    def chain_characteristics(self,chain_args,dlayers,chars):
        """Accumulate chain-shape characteristics binned by distance.

        Used as the per-frame kernel for
        :meth:`Analysis_Confined.calc_chain_characteristics`.

        Parameters
        ----------
        self : Analysis_Confined
            Provides geometry, masses and confinement helpers.
        chain_args : dict
            Mapping from chain id to atom index array.
        dlayers : sequence of tuple
            Distance intervals ``(d_min, d_max)``.
        chars : dict
            Mapping from characteristic name (e.g. ``"Rg2"``, ``"k2"``)
            to lists-of-lists that will be filled with per-layer
            values.
        """
        #1) translate the system
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        cs = self.get_particle_cm(coords)


        for j in chain_args:
            #find chain  center of mass
            c_ch = coords[chain_args[j]]
            at_mass_ch = self.atom_mass[chain_args[j]]
            ch_cm = CM(c_ch,at_mass_ch)

            #compute gyration tensor,rg,asphericity,acylindricity,end-to-end-distance
            Sm = np.zeros((3,3),dtype=float)
            Ree2, Rg2, k2, asph, acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz \
            = chain_characteristics_kernel(c_ch, at_mass_ch,ch_cm,Sm)

            Rg = Rg2**0.5
            local_dict = locals()
            #Assign values
            d =1e16
            for L in self.box_add(box):
                d = np.minimum(d,self.dfun(self,ch_cm.reshape(1,3),cs+L))
            for i,dl in enumerate(dlayers):
                if dl[0]< d[0] <=dl[1]:
                    for char in chars:
                        chars[char][i].append(local_dict[char])
                    break
        return

    @staticmethod
    def dihedrals_t(self,
                dih_ids, ids1, ids2, filters,  dihedrals_t, filt_per_t):
        """Evaluate dihedral angles for the current frame.

        Stores raw dihedral values under ``dihedrals_t`` and updates
        corresponding filters in ``filt_per_t``.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and filter evaluation.
        dih_ids : ndarray of int
            Dihedral index quadruplets.
        ids1, ids2 : ndarray of int
            Atom indices used by the filters.
        filters : dict
            Filter specifications passed to :class:`Filters`.
        dihedrals_t : dict
            Dictionary updated with one array of dihedral values per
            time key.
        filt_per_t : dict
            Dictionary updated with filter masks per time key.
        """

        t0 = perf_counter()
        frame = self.current_frame
        coords = self.get_coords(frame)

        dih_val = np.empty(dih_ids.shape[0],dtype=float) # alloc
        dihedral_values_kernel(dih_ids,coords,dih_val)

        key = self.get_key()
        dihedrals_t[key] = dih_val.copy()

        del dih_val #deallocating for safety
        tm = perf_counter()


        filters_info = self.get_filters_info(filters, ids1 = ids1, ids2 = ids2,
        segmental_ids = dih_ids)

        filt_per_t[key] = Filter_Operations.calc_filters(filters, filters_info)

        tf = perf_counter()
        if frame ==1:
            logger.info('Dihedrals_as_t: Estimate time consuption --> Main: {:2.1f} %, Filters: {:2.1f} %'.format((tm-t0)*100/(tf-t0),(tf-tm)*100/(tf-t0)))
        return

    @staticmethod
    def coords_t(self,filters, ids, c_t,filt_per_t):
        """Store selected coordinates and associated filters for a frame.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and filter evaluation.
        filters : dict
            Filter specifications passed to :class:`Filters`.
        ids : ndarray of int
            Atom indices whose coordinates are stored.
        c_t : dict
            Dictionary updated with coordinate arrays per time key.
        filt_per_t : dict
            Dictionary updated with filter masks per time key.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)


        key = self.get_key()
        c_t[key] = coords[ids]

        filters_info = self.get_filters_info(filters, ids1=ids, ids2=ids)

        filt_per_t[key] = Filter_Operations.calc_filters(filters, filters_info)
        return


    @staticmethod
    def vects_t(self,ids1,ids2,filters,vec_t,filt_per_t):
        """Store bond/segment vectors and associated filters for a frame.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and filter evaluation.
        ids1, ids2 : ndarray of int
            Indices of the two endpoints of each vector.
        filters : dict
            Filter specifications passed to :class:`Filters`.
        vec_t : dict
            Dictionary updated with vector arrays per time key.
        filt_per_t : dict
            Dictionary updated with filter masks per time key.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)

        vec = coords[ids2,:] - coords[ids1,:]

        key = self.get_key()
        vec_t[key] = vec

        filters_info = self.get_filters_info(filters, ids1=ids1, ids2=ids2)
        filt_per_t[key] = Filter_Operations.calc_filters(filters, filters_info )
        return

    @staticmethod
    def confs_t(self,confs_t):
        """Collect instantaneous fractions and counts of conformations.

        Parameters
        ----------
        confs_t : dict
            Dictionary updated with per-frame statistics such as
            fractions (``"x_*"``) and counts (``"n_*"``) for trains,
            tails, loops, bridges and adsorbed chains.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)

        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        ntot = args_train.shape[0] + args_tail.shape[0] +\
               args_loop.shape[0] + args_bridge.shape[0]
        for k in ['train','tail','loop','bridge']:
            args = locals()['args_'+k]
            x['x_'+k] = args.shape[0]/ntot
            x['n_'+k] = args.shape[0]
        x['x_ads_chains'] = ads_chains.shape[0]/len(self.chain_args)
        x['n_ads_chains'] = ads_chains.shape[0]


        key = self.get_key()
        confs_t[key] = x

        return

    @staticmethod
    def confs_t__length(self,confs_t):
        """Collect average contour lengths of conformational chunks.

        Parameters
        ----------
        confs_t : dict
            Dictionary updated with mean and standard deviation of
            contour lengths for each conformation.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)

        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        for args,lab in zip([args_train, args_tail, args_loop, args_bridge],
                        ['l_train','l_tail','l_loop','l_bridge']):
            connected_chunks = self.connected_chunks(args)
            lengths = [self.length_of_connected_chunks(list(ch),coords)
                       for ch in connected_chunks ]

            m = np.mean(lengths)
            std = np.std(lengths)
            x[lab] = m
            x[lab+'(std)'] = std
           #x[lab+'_lenghts'] = lengths

        key = self.get_key()

        confs_t[key] = x

        return

    @staticmethod
    def confs_t__size(self,confs_t):
        """Collect average sizes (in atoms) of conformational chunks.

        Parameters
        ----------
        confs_t : dict
            Dictionary updated with mean and standard deviation of
            sizes (in atoms) for each conformation.
        """

        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        for args,lab in zip([args_train, args_tail, args_loop, args_bridge],
                        ['s_train','s_tail','s_loop','s_bridge']):
            connected_chunks = self.connected_chunks(args)
            sizes = [s.__len__() for s in connected_chunks]
            size_m = np.mean(sizes)
            size_std = np.std(sizes)
            x[lab] = size_m
            x[lab+'(std)'] = size_std
            #x[lab+'_sizes'] = sizes

        key = self.get_key()
        confs_t[key] = x

        return


    @staticmethod
    def confs_t__perchain(self,confs_t):
        """Collect per-chain fractions of atoms in each conformation.

        Parameters
        ----------
        confs_t : dict
            Dictionary updated with per-chain fractions of atoms in
            each conformational state.
        """
        frame = self.current_frame
        coords,box,d,cs = self.get_whole_frame_basics(frame)

        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        for k in ['train','tail','loop','bridge']:
            args = locals()['args_'+k]
            x[k] =  [ np.count_nonzero( np.isin(a, args ) )/a.shape[0]
                                      for a in self.chain_args.values() ]


        key = self.get_key()
        confs_t[key] = x

        return

    @staticmethod
    def segCM_t(self,filters,segment_ids,ids1, ids2, vec_t,filt_per_t):
        """Store segment centres-of-mass relative to a reference point.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates, segment definitions and reference CM.
        filters : dict
            Filter specifications passed to :class:`Filters`.
        segment_ids : ndarray
            Segment definitions passed to :meth:`segs_CM`.
        vec_t : dict
            Dictionary updated with CM vectors per time key.
        filt_per_t : dict
            Dictionary updated with filter masks per time key.
        """

        frame = self.current_frame
        coords = self.get_coords(frame)

        seg_cm = self.segs_CM(coords,segment_ids)

        key = self.get_key()

        vec_t[key] =  seg_cm

        filters_info = self.get_filters_info(filters, relative_coords = seg_cm,
                                             ids1=ids1, ids2=ids2 )

        filt_per_t[key] = Filter_Operations.calc_filters(filters, filters_info)

        return

    @staticmethod
    def find_clusters_given_pair_distances(n,pd,dcut):
        """Identify clusters from pairwise distances and a cutoff.

        Parameters
        ----------
        n : int
            Number of objects (e.g. segments or centres of mass).
        pd : ndarray, shape (n*(n-1)/2,)
            Flattened upper-triangular array of pairwise distances.
        dcut : float
            Distance cutoff below which two objects are considered
            connected.

        Returns
        -------
        sizes : list of int
            Sizes of all identified clusters.
        sets : list of set of int
            Each set contains the indices belonging to one cluster.
        """
        k=0
        neibs = {i:set() for i in range(n)}
        for i in range(n):
            for j in range(i+1,n):
                if pd[k]<=dcut:
                    neibs[i].add(j)
                    neibs[j].add(i)
                k+=1

        args = set()

        all_args = set([i for i in range(n)])

        sci = {0,}
        sizes = []

        sets = []
        while len(args) != n:

            old_set = set()
            while (len(sci) != len(old_set)):
                old_set = sci.copy()
                for i in old_set:
                    sci = sci | neibs[i]

            sizes.append(len(sci))
            sets.append(sci)

            args = args | sci
            jfind = list(all_args - args)
            if len(jfind) ==0:
                continue

            sci = {jfind[0],}

        return sizes, sets

    @staticmethod
    def cluster_size_min(self,segmental_ids,dcut,distribution,clusters):
        """Compute cluster-size distribution using minimal segment distance.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and box.
        segmental_ids : ndarray
            Segment index array passed to distance calculations.
        dcut : float
            Distance cutoff defining cluster connectivity.
        distribution : dict
            Dictionary updated per time key with a list of cluster
            sizes.
        clusters : dict
            Dictionary updated per time key with the cluster membership
            sets.
        """
        key = self.get_key()
        distribution[key] = dict()

        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)

        n = len(segmental_ids)

        pd = np.empty(int(n*(n-1)/2),dtype=float)

        k=0
        for i in range(n):
            ci = coords[segmental_ids[i]]
            for j in range(i+1,n):
                cj = coords[segmental_ids[j]]
                mic = minimum_image_relative_coords(ci-cj,box)
                d = np.sum(mic*mic,axis=1)**0.5
                pd[k] = d.min()
                k+=1

        sizes,sets = coreFunctions.find_clusters_given_pair_distances(n,pd,dcut)

        distribution[key] = sizes
        clusters[key] = sets
        return

    @staticmethod
    def cluster_size_com(self,segmental_ids,dcut,distribution):
        """Compute cluster-size distribution using segment centres-of-mass.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates, box and :meth:`segs_CM`.
        segmental_ids : ndarray
            Segment index array passed to :meth:`segs_CM`.
        dcut : float
            Distance cutoff defining cluster connectivity.
        distribution : dict
            Dictionary updated per time key with a list of cluster
            sizes.
        """
        key = self.get_key()
        distribution[key] = dict()

        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        seg_cm = self.segs_CM(coords,segmental_ids)
        n = len(segmental_ids)

        pd = np.empty(int(n*(n-1)/2),dtype=float)

        pair_dists(seg_cm,box,pd)

        sizes,sets = coreFunctions.find_clusters_given_pair_distances(n,pd,dcut)

        distribution[key] = sizes
        return

    @staticmethod
    def gofr_segments(self,bins,segment_ids,gofr):
        """Accumulate segment–segment pair distances into a :math:`g(r)` histogram.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates, box and :meth:`segs_CM`.
        bins : ndarray
            Radial bin edges.
        segment_ids : ndarray
            Segment definitions.
        gofr : ndarray
            Histogram array updated in place with pair counts per bin.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        seg_cm = self.segs_CM(coords,segment_ids)
        n = len(seg_cm)

        pd = np.empty(int(n*(n-1)/2),dtype=float)

        pair_dists(seg_cm,box,pd)

        pd = pd[pd<=bins.max()]

        numba_bin_count(pd,bins,gofr)

        return

    @staticmethod
    def gofr_pairs(self,ids1,ids2,bins,gofr):
        """Accumulate pair distances between two atom sets into :math:`g(r)`.

        Parameters
        ----------
        self : Analysis or Analysis_Confined
            Provides coordinates and box.
        ids1, ids2 : ndarray of int
            Atom indices defining the two sets whose pair distances are
            considered.
        bins : ndarray
            Radial bin edges.
        gofr : ndarray
            Histogram array updated in place with pair counts per bin.
        """
        frame = self.current_frame
        coords = self.get_coords(frame)

        c1 = coords[ids1]
        c2 = coords[ids2]
        box = self.get_box(frame)


        relc = minimum_image_relative_coords(c2-c1,box)
        pd =np.sum(relc*relc,axis=1)**0.5

        pd = pd[pd<=bins.max()]

        numba_bin_count(pd,bins,gofr)

        return


@jit(nopython=True,fastmath=True)
def fill_property(prop,nv,i,j,value,mi,block_average):
    """Accumulate a correlation property and normalisation factor.

    Parameters
    ----------
    prop : ndarray
        Output array where the property is accumulated for lag
        ``j - i``.
    nv : ndarray
        Normalisation counts per lag, updated in place.
    i, j : int
        Time-origin and current-time indices.
    value : float
        Correlation value for this (i, j) pair.
    mi : float
        Effective multiplicity / weight for this pair.
    block_average : bool
        If ``True``, perform block averaging over contributions; if
        ``False``, accumulate raw sums.
    """

    idx = j-i
    if block_average:
        try:
            prop[idx] +=  value/mi
            nv[idx] += 1.0
        except:
            pass
    else:
        prop[idx] +=  value
        nv[idx] += mi

    return


@jit(nopython=True,fastmath=True,parallel=True)
def Kinetics_kernel(func,func_args,
                    Prop,nv,xt,wt=None,
                    block_average=False,
                    multy_origin=True):
    """Numba kernel for two-state kinetics correlations.

    Parameters
    ----------
    func : callable
        Inner kernel returning ``(value, mi)`` for a given time pair.
    func_args : callable
        Function building the argument tuple for ``func``.
    Prop : ndarray
        Output array for the kinetic observable, updated in place.
    nv : ndarray
        Normalisation counts per lag, updated in place.
    xt : ndarray
        Boolean time series over which kinetics is evaluated.
    wt : ndarray or None, optional
        Optional weights per time and entity.
    block_average : bool, optional
        If ``True``, use block averaging over time origins.
    multy_origin : bool, optional
        If ``True``, use multiple time origins; otherwise a single
        origin is used.
    """

    n = xt.shape[0]

    if multy_origin: mo = n
    else: mo = 1

    for i in range(mo):
        for j in prange(i,n):
            args = func_args(i,j,xt,None,wt)

            value,mi = func(*args)
            fill_property(Prop,nv,i,j,value,mi,block_average)

    for i in prange(n):
        Prop[i] /= nv[i]
    return

@jit(nopython=True,fastmath=True)
def Kinetics_inner__kernel(x0,xt):
    """Unweighted two-state kernel for Kinetics.

    Parameters
    ----------
    x0 : ndarray of bool
        State at the time origin.
    xt : ndarray of bool
        State at time ``t``.

    Returns
    -------
    value : float
        Number of entities that stayed in the state.
    m : float
        Number of entities that were initially in the state.
    """
    value = 0.0 ; m = 0.0
    for i in range(x0.shape[0]):
        if x0[i]:
            if xt[i]:
                value += 1.0
            m += 1.0
    return value,m

@jit(nopython=True,fastmath=True)
def Kinetics_inner_weighted__kernel(x0,xt,w0):
    """Weighted two-state kernel for Kinetics.

    Parameters
    ----------
    x0 : ndarray of bool
        State at the time origin.
    xt : ndarray of bool
        State at time ``t``.
    w0 : ndarray of float
        Weights per entity at the origin.

    Returns
    -------
    value : float
        Weighted number of entities that stayed in the state.
    m : float
        Sum of weights of entities initially in the state.
    """
    value = 0.0 ; m = 0.0
    for i in range(x0.shape[0]):
        if x0[i]:
            wi = w0[i]
            if xt[i]:
                value += wi
            m += wi
    return value,m

@jit(nopython=True,fastmath=True)
def get_zero__args(i,xt,ft,wt):
    """Return arguments for zero-time TACF kernels without filtering."""
    return (xt[i],)

@jit(nopython=True,fastmath=True)
def get_zero_filt__args(i,xt,ft,wt):
    """Return arguments for zero-time TACF kernels with filtering."""
    return (xt[i],ft[i])

@jit(nopython=True,fastmath=True)
def get_zero_weighted__args(i,xt,ft,wt):
    """Return arguments for zero-time weighted TACF kernels."""
    return (xt[i],  wt[i])

@jit(nopython=True,fastmath=True)
def get_zero_filt_weighted__args(i,xt,ft,wt):
    """Return arguments for zero-time filtered and weighted TACF kernels."""
    return (xt[i],ft[i],wt[i])

@jit(nopython = True,fastmath=True)
def mean__kernel(ifunc,x):
    """Compute mean value of a function over an array.

    Parameters
    ----------
    ifunc : callable
        Function applied element-wise to ``x``.
    x : ndarray
        Input data.

    Returns
    -------
    mean : float
        Sum of ``ifunc(x[i])``.
    mi : float
        Number of contributing elements.
    """
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        mean += ifunc(x[i])
    mi = float(x.shape[0])
    return mean, mi

@jit(nopython = True,fastmath=True)
def mean_weighted__kernel(ifunc,x,w):
    """Compute weighted mean of a function over an array."""
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        wi = w[i]
        mean += wi*ifunc(x[i])
        mi += wi
    return mean,mi

@jit(nopython = True,fastmath=True)
def mean_filt__kernel(ifunc,x,f):
    """Compute mean of a function over filtered elements."""
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            mean+=ifunc(x[i])
            mi+=1.0
    return mean, mi

@jit(nopython = True,fastmath=True)
def mean_filt_weighted__kernel(ifunc,x,f,w):
    """Compute weighted mean of a function over filtered elements."""
    mean =0 ; mi=0
    for i in range(x.shape[0]):
        if f[i]:
            wi = w[i]
            mean+=wi*ifunc(x[i])
            mi+=wi
    return mean, mi


@jit(nopython = True,fastmath=True)
def secmoment__kernel(ifunc,x):
    """Compute second moment of a function over an array."""
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        xi = ifunc(x[i])
        sec += xi*xi
    mi = float(x.shape[0])
    return sec, mi

@jit(nopython = True,fastmath=True)
def secmoment_weighted__kernel(ifunc,x,w):
    """Compute weighted second moment of a function over an array."""
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        wi = w[i]
        xi = ifunc(x[i])
        sec += wi*xi*xi
        mi += wi
    return sec,mi

@jit(nopython = True,fastmath=True)
def secmoment_filt__kernel(ifunc,x,f):
    """Compute second moment over filtered elements."""
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            xi = ifunc(x[i])
            sec+=xi*xi
            mi+=1.0
    return sec, mi

@jit(nopython = True,fastmath=True)
def secmoment_filt_weighted__kernel(ifunc,x,f,w):
    """Compute weighted second moment over filtered elements."""
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            wi = w[i]
            xi = ifunc(x[i])
            sec+=wi*xi*xi
            mi+=w[i]
    return sec, mi


@jit(nopython=True,fastmath=True,parallel=True)
def TACF_kernel(func, func_args, inner_func,
              mean_func, secmoment_func, func_args_zero, inner_func_zero,
              Prop,nv,
              mu_val,mu_num,secmom_val,secmom_num,
              xt, ft=None, wt=None,
              block_average=False):
    """Numba kernel for trigonometric time autocorrelation functions.

    Parameters
    ----------
    func : callable
        Main TACF kernel applied to inner function values.
    func_args : callable
        Builder for (i, j) argument tuples of ``func``.
    inner_func : callable
        Low-level kernel (e.g. cosine or sine) applied to data.
    mean_func, secmoment_func : callable
        Kernels computing first and second moments at zero time.
    func_args_zero : callable
        Builder for zero-time argument tuples.
    inner_func_zero : callable
        Inner function evaluated at zero time.
    Prop : ndarray
        Output TACF array updated in place.
    nv : ndarray
        Normalisation counts per lag.
    mu_val, mu_num, secmom_val, secmom_num : ndarray
        Work arrays for mean and second-moment accumulation.
    xt : ndarray
        Time series of input values.
    ft : ndarray or None, optional
        Optional boolean filters.
    wt : ndarray or None, optional
        Optional weights per time and entity.
    block_average : bool, optional
        If ``True``, use block averaging over origins.
    """

    n= xt.shape[0]

    for i in range(n):

        args_zero = func_args_zero(i,xt,ft,wt)

        mu_val[i],mu_num[i] = mean_func(inner_func_zero,*args_zero)

        secmom_val[i],secmom_num[i] = secmoment_func(inner_func_zero,*args_zero)

        for j in prange(i,n):
            # Need to add an empty tuple for consistency of the dynprop__kernel
            kernel_args = tuple()
            args = (*func_args(i,j,xt,ft,wt), kernel_args)

            value,mi = func(inner_func,*args)

            fill_property(Prop,nv,i,j,value,mi,block_average)

    if block_average:
        for i in range(n):

            mui = mu_val[i]/mu_num[i]
            seci = secmom_val[i]/secmom_num[i]

            mui_square = mui*mui
            vari = seci - mui_square

            Prop[i] = (Prop[i]/nv[i] - mui_square)/vari

        return
    else:
        mu =0 ;nmu =0
        sec = 0;nsec =0
        for i in prange(n):
            mu+=mu_val[i]  ; nmu+=mu_num[i]
            sec+= secmom_val[i] ; nsec += secmom_num[i]
        mu/=nmu
        sec/=nsec

        mu_sq = mu*mu
        var = sec-mu_sq

        for i in prange(n):
            Prop[i] = (Prop[i]/nv[i] - mu_sq)/var

        return
@jit(nopython=True,fastmath=True,parallel=True )
def scalar_time_origin_average(Prop,nv,xt,every=1):
    """Average a scalar observable over multiple time origins.

    Parameters
    ----------
    Prop : ndarray
        Output array for the averaged observable.
    nv : ndarray
        Normalisation counts per lag.
    xt : ndarray
        Scalar time series.
    every : int, optional
        Spacing between time origins.
    """
    n = xt.shape[0]
    for i in range(0,n,every):
        s1 = xt[i]
        for j in prange(i,n):

            s2 = xt[j]

            value = s1*s2

            fill_property(Prop,nv,i,j,value,1.0,False)
    for i in prange(n):
        Prop[i] /= nv[i]
    return

@jit(nopython=True,fastmath=True,parallel=True)
def DynamicProperty_kernel(func,func_args,inner_func,
              Prop,nv,xt,ft,wt,
              block_average,multy_origin,
              every,kernel_args):
    """Generic Numba kernel for dynamical correlation properties.

    This routine underlies many single-particle dynamical observables
    (MSD, dipole correlations, segmental dynamics, etc.). It loops
    over time origins and time lags, constructs the appropriate
    arguments via ``func_args`` and applies the user-provided
    ``inner_func`` through ``func`` to accumulate the observable in
    ``Prop`` with normalisation counts ``nv``.

    Parameters
    ----------
    func : callable
        High-level kernel that takes ``inner_func`` and arguments
        built by ``func_args`` and returns ``(value, mi)``.
    func_args : callable
        Function building argument tuples for ``func`` given
        ``(i, j, xt, ft, wt, kernel_args)``.
    inner_func : callable
        Low-level function applied to the underlying vectors/scalars
        (e.g. displacement, orientation).
    Prop : ndarray
        Output array holding the accumulated dynamical property as a
        function of lag time.
    nv : ndarray
        Normalisation counts per lag, updated in place.
    xt : ndarray
        Time-resolved data (e.g. coordinates, vectors, scalars).
    ft : ndarray or None
        Optional boolean filters per time and entity.
    wt : ndarray or None
        Optional weights per time and entity.
    block_average : bool
        If ``True``, use block averaging over time origins.
    multy_origin : bool
        If ``True``, use multiple time origins; otherwise restrict to
        a single origin.
    every : int
        Interval between successive time origins.
    kernel_args : tuple
        Extra arguments forwarded to ``func_args`` and ``func``.
    """
    n = xt.shape[0]

    if multy_origin: mo = n
    else: mo = 1

    for i in range(0,mo,every):
        for j in prange(i,n):

            args = func_args(i,j,xt,ft,wt)
            args=(*args,kernel_args)
            value,mi = func(inner_func,*args)

            fill_property(Prop,nv,i,j,value,mi,block_average)

    for i in prange(n):
        Prop[i] /= nv[i]
    return

@jit(nopython=True,fastmath=True)
def get__args(i,j,xt,ft,wt):
    """Build basic (unfiltered, unweighted) argument tuple for kernels.

    Parameters
    ----------
    i, j : int
        Time-origin and current-time indices.
    xt : ndarray
        Time-resolved data.
    ft, wt : any
        Unused here; kept for a uniform call signature.

    Returns
    -------
    tuple
        ``(xt[i], xt[j])``.
    """
    return (xt[i],xt[j])

@jit(nopython=True,fastmath=True)
def get_weighted__args(i,j,xt,ft,wt):
    """Build weighted argument tuple for kernels.

    Returns
    -------
    tuple
        ``(xt[i], xt[j], wt[i])``.
    """
    return (xt[i], xt[j],  wt[i])

@jit(nopython=True,fastmath=True)
def get_const__args(i,j,xt,ft,wt):
    """Build argument tuple with 'const' filter over an interval [i, j].

    The ``const`` mask is ``True`` only if all intermediate filters
    are ``True`` for a given entity.
    """
    const = np.empty_like(ft[i])
    for k in range(const.shape[0]):
        for t in range(i,j+1):
            const[k] = const[k] and ft[t][k]
    return (xt[i],xt[j],const)

@jit(nopython=True,fastmath=True)
def get_simple__args(i,j,xt,ft,wt):
    """Build argument tuple with simple per-time filter at origin i.

    Returns
    -------
    tuple
        ``(xt[i], xt[j], ft[i])``.
    """
    return (xt[i],xt[j],ft[i])

@jit(nopython=True,fastmath=True)
def get_simple_weighted__args(i,j,xt,ft,wt):
    """Build argument tuple with simple filter and weights at origin i."""
    return (xt[i],xt[j],ft[i],wt[i])


@jit(nopython=True,fastmath=True)
def get_strict__args(i,j,xt,ft,wt):
    """Build argument tuple with strict filtering at both times i and j.

    Returns
    -------
    tuple
        ``(xt[i], xt[j], ft[i], ft[j])``.
    """
    return (xt[i],xt[j],ft[i],ft[j])

@jit(nopython=True,fastmath=True)
def get_strict_weighted__args(i,j,xt,ft,wt):
    """Build argument tuple with strict filtering and weights at origin i.

    Returns
    -------
    tuple
        ``(xt[i], xt[j], ft[i], ft[j], wt[i])``.
    """
    return (xt[i],xt[j],ft[i],ft[j],wt[i])


@jit(nopython=True,fastmath=True)
def get_change__args(i,j,xt,ft,wt):
    """Build argument tuple to detect state changes between i and j.

    Returns
    -------
    tuple
        ``(xt[i], xt[j], ft[i], ft[j])``.
    """
    return (xt[i],xt[j],ft[i],ft[j])

@jit(nopython=True,fastmath=True)
def get_change_weighted__args(i,j,xt,ft,wt):
    """Build argument tuple for weighted state-change kernels.

    Returns
    -------
    tuple
        ``(xt[i], xt[j], ft[i], ft[j], wt[i])``.
    """
    return (xt[i],xt[j],ft[i],ft[j],wt[i])

@jit(nopython=True,fastmath=True)
def dynprop__kernel(inner_kernel,r1,r2,kernel_args):
    """Base kernel for unfiltered, unweighted dynamical properties.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.

    Returns
    -------
    tot : float
        Sum of inner-kernel values over all entities.
    mi : float
        Number of contributing entities.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        inner = inner_kernel(r1[i],r2[i],*kernel_args)
        tot+=inner
        mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_simple__kernel(inner_kernel,r1,r2,ft0,kernel_args):
    """Dynamical kernel restricted to entities selected at origin.

    Only entities with ``ft0[i]`` true contribute.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    ft0 : ndarray of bool
        Selection mask at the origin time.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_strict__kernel(inner_kernel,r1,r2,ft0,fte,kernel_args):
    """Dynamical kernel requiring entities to be selected at both times.

    Only entities with ``ft0[i] and fte[i]`` true contribute.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    ft0, fte : ndarray of bool
        Selection masks at origin and end times respectively.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_change__kernel(inner_kernel,r1,r2,ft0,fte,kernel_args):
    """Dynamical kernel restricted to entities that change state.

    Only entities with ``ft0[i]`` true and ``fte[i]`` false contribute,
    i.e. that are selected at the origin but not at the end time.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    ft0, fte : ndarray of bool
        Selection masks at origin and end times respectively.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_const__kernel(inner_kernel,r1,r2,const,kernel_args):
    """Dynamical kernel restricted to entities with constant state.

    Only entities with ``const[i]`` true contribute.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    const : ndarray of bool
        Boolean mask selecting entities that remain in a given state.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]

    for i in prange(N):
        if const[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_weighted__kernel(inner_kernel,r1,r2,w,kernel_args):
    """Weighted dynamical kernel without additional filtering.

    Each entity contributes regardless of state; its contribution is
    multiplied by the corresponding weight ``w[i]``.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    w : ndarray of float
        Weights applied to each entity.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        inner = inner_kernel(r1[i],r2[i],*kernel_args)
        tot+=w[i]*inner
        mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_simple_weighted__kernel(inner_kernel,r1,r2,ft0,w,kernel_args):
    """Weighted simple dynamical kernel using origin-time selection.

    Only entities with ``ft0[i]`` true contribute; their contributions
    are multiplied by the corresponding weight ``w[i]``.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    ft0 : ndarray of bool
        Selection mask at origin time.
    w : ndarray of float
        Weights applied to each entity.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_strict_weighted__kernel(inner_kernel,r1,r2,ft0,fte,w,kernel_args):
    """Weighted strict dynamical kernel (selected at both times).

    Only entities with ``ft0[i] and fte[i]`` true contribute; their
    contributions are multiplied by the corresponding weight ``w[i]``.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    ft0, fte : ndarray of bool
        Selection masks at origin and end times respectively.
    w : ndarray of float
        Weights applied to each entity.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_change_weighted__kernel(inner_kernel,r1,r2,ft0,fte,w,kernel_args):
    """Weighted dynamical kernel for entities that change state.

    Only entities with ``ft0[i]`` true and ``fte[i]`` false contribute;
    their contributions are multiplied by the corresponding weight
    ``w[i]``.

    Parameters
    ----------
    inner_kernel : callable
        Function applied to each pair ``(r1[i], r2[i], *kernel_args)``.
    r1, r2 : ndarray
        Arrays of shape ``(N, ...)`` containing values at two times.
    ft0, fte : ndarray of bool
        Selection masks at origin and end times respectively.
    w : ndarray of float
        Weights applied to each entity.
    kernel_args : tuple
        Extra arguments forwarded to ``inner_kernel``.
    """
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi




@jit(nopython=True,parallel=True)
def numba_bin_count(d,bins,counter):
    """Count distances into bins using a simple histogram.

    Parameters
    ----------
    d : ndarray
        Distances to be binned.
    bins : ndarray
        Bin edges of length ``nbins + 1``.
    counter : ndarray
        Histogram array of length ``nbins`` updated in place.
    """
    for j in prange(bins.shape[0]-1):
        for i in range(d.shape[0]):
            if bins[j]<d[i] and d[i] <=bins[j+1]:
                counter[j] +=1
    return

@jit(nopython=True,fastmath=True)
def fcos_kernel(x):
    """Element-wise cosine kernel used in TACF and related routines.

    Parameters
    ----------
    x : ndarray or float
        Input angles in radians.

    Returns
    -------
    ndarray or float
        ``np.cos(x)`` with the same shape as ``x``.
    """
    return np.cos(x)

@jit(nopython=True,fastmath=True)
def fsin_kernel(x):
    """Element-wise sine kernel used in TACF and related routines.

    Parameters
    ----------
    x : ndarray or float
        Input angles in radians.

    Returns
    -------
    ndarray or float
        ``np.sin(x)`` with the same shape as ``x``.
    """
    return np.sin(x)

@jit(nopython=True,fastmath=True)
def cosCorrelation_kernel(x1,x2):
    """Compute product of cosines for two angle arrays.

    Parameters
    ----------
    x1, x2 : ndarray
        Angle arrays in radians with compatible shapes.

    Returns
    -------
    ndarray
        ``np.cos(x1) * np.cos(x2)``.
    """

    c1= np.cos(x1)
    c2 = np.cos(x2)
    return c1*c2

@jit(nopython=True,fastmath=True)
def sinCorrelation_kernel(x1,x2):
    """Compute product of sines for two angle arrays.

    Parameters
    ----------
    x1, x2 : ndarray
        Angle arrays in radians with compatible shapes.

    Returns
    -------
    ndarray
        ``np.sin(x1) * np.sin(x2)``.
    """

    s1= np.sin(x1)
    s2 = np.sin(x2)
    return s1*s2

@jit(nopython=True,fastmath=True)
def mult_kernel(r1,r2,*args):
    """Simple element-wise multiplication kernel ``r1 * r2``.

    Parameters
    ----------
    r1, r2 : ndarray
        Arrays with compatible shapes for element-wise multiplication.

    Returns
    -------
    ndarray
        Element-wise product ``r1 * r2``.
    """
    return r1*r2

@jit(nopython=True,fastmath=True)
def cos2th_kernel(r1,r2,*args):
    """Kernel returning :math:`\cos^2(\theta)` between two vectors.

    Parameters
    ----------
    r1, r2 : ndarray
        Vectors whose mutual angle is used.

    Returns
    -------
    float
        :math:`\cos^2(\theta)` for the angle between ``r1`` and ``r2``.
    """
    costh = costh_kernel(r1,r2)
    return costh*costh

@jit(nopython=True,fastmath=True)
def costh_kernel(r1,r2,*args):
    """Compute cosine of the angle between two vectors.

    Parameters
    ----------
    r1, r2 : ndarray
        Vectors whose angle is computed.

    Returns
    -------
    float
        Cosine of the angle between ``r1`` and ``r2``.
    """
    costh=0 ; rn1 =0 ;rn2=0
    n = r1.shape[0]
    for j in range(n):
        costh += r1[j]*r2[j]
        rn1 += r1[j]*r1[j]
        rn2 += r2[j]*r2[j]
    rn1 = rn1**0.5
    rn2 = rn2**0.5
    costh/=rn1*rn2
    return costh

@jit(nopython=True,fastmath=True)
def Fs_kernel(r1,r2,q):
    """Self-intermediate scattering function kernel for a single atom.

    Parameters
    ----------
    r1, r2 : array_like, shape (3,)
        Initial and final positions of a particle.
    q : float
        Scalar magnitude of the scattering vector.

    Returns
    -------
    float
        Contribution :math:`\cos( q (r_2 - r_1) )` using the summed
        Cartesian components, consistent with the original code.
    """
    ri =  q*(r2[0] +r2[1] + r2[2] - r1[0] - r1[1] -r1[2])
    return np.cos(ri)

@jit(nopython=True,fastmath=True)
def norm_square_kernel(r1,r2,*args):
    """Return squared norm of displacement between ``r1`` and ``r2``.

    Parameters
    ----------
    r1, r2 : ndarray, shape (N, ...)
        Coordinate arrays; the displacement is ``r2 - r1``.

    Returns
    -------
    float
        Squared norm :math:`|r_2 - r_1|^2` summed over all entries.
    """

    nm = 0
    for i in range(r1.shape[0]):
        ri = r2[i] - r1[i]
        nm+= ri*ri
    return nm

@jit(nopython=True,fastmath=True,parallel=True)
def costh__parallelkernel(r1,r2,*args):
    """Parallel average of cos(theta) over many vector pairs.

    Parameters
    ----------
    r1, r2 : ndarray, shape (M, D)
        Collections of ``M`` pairs of ``D``-dimensional vectors.

    Returns
    -------
    float
        Mean value of :math:`\cos(\theta)` over all pairs.
    """
    tot = 0
    for i in prange(r1.shape[0]):
        tot += costh_kernel(r1[i],r2[i])
    ave = tot/float(r1.shape[0])
    return ave

@jit(nopython=True,fastmath=True)
def costhsquare__kernel(costh,r1,r2,*args):
    """Fill array with :math:`\cos^2(\theta)` for each vector pair.

    Parameters
    ----------
    costh : ndarray
        Output array; on return ``costh[i] = cos^2(theta_i)``.
    r1, r2 : ndarray, shape (M, D)
        Collections of vector pairs.
    """
    for i in range(r1.shape[0]):
        costh[i] = cos2th_kernel(r1[i],r2[i])

@jit(nopython=True,fastmath=True)
def costhmean__kernel(r1,r2,*args):
    """Mean value of cos(theta) over all vector pairs.

    Parameters
    ----------
    r1, r2 : ndarray, shape (M, D)
        Collections of vector pairs.

    Returns
    -------
    float
        Mean value of :math:`\cos(\theta)` over all pairs.
    """
    mean =0
    for i in range(r1.shape[0]):
        costh = costh_kernel(r1[i],r2[i])
        mean+=costh
    ave = mean/r1.shape[0]
    return ave



@jit(nopython=True,fastmath=True)
def unwrap_coords_kernel(unc,k0,k1,b2,n,dim,box):
    """Unwrap coordinates across periodic boundaries for selected pairs.

    Parameters
    ----------
    unc : ndarray
        Unwrapped coordinates, modified in place.
    k0, k1 : ndarray of int
        Index arrays defining atom pairs.
    b2 : array_like
        Half box lengths per dimension.
    n : iterable of int
        Indices of pairs to process.
    dim : iterable of int
        Dimensions to consider (0, 1, 2).
    box : array_like
        Box lengths per dimension.

    Returns
    -------
    ndarray
        The modified coordinate array ``unc``.
    """
    for j in dim:
        for i in n:
            if unc[k0[i],j] - unc[k1[i],j] > b2[j]:
                unc[k1[i],j] += box[j]
            elif unc[k1[i],j] - unc[k0[i],j] > b2[j]:
                unc[k1[i],j] -= box[j]
    return unc

@jit(nopython=True,fastmath=True)
def end_to_end_distance(coords):
    """Return squared end-to-end distance of a chain.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Ordered chain coordinates.

    Returns
    -------
    float
        Squared end-to-end distance ``|r_0 - r_{N-1}|^2``.
    """
    rel = coords[0]-coords[coords.shape[0]-1]
    Ree2 = np.dot(rel,rel)
    return Ree2
@jit(nopython=True,fastmath=True)
def chain_characteristics_kernel(coords,mass,ch_cm,Gyt):
    """Kernel computing basic chain-shape invariants from coordinates.

    Returns
    -------
    Ree2 : float
        Squared end-to-end distance.
    Rg2 : float
        Squared radius of gyration.
    k2 : float
        Relative shape anisotropy.
    asph : float
        Asphericity.
    acyl : float
        Acylindricity.
    Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz : float
        Selected tensor element combinations used in diagnostics.
    """
    rel = coords[0]-coords[coords.shape[0]-1]
    Ree2 = np.dot(rel,rel)
    rccm = coords-ch_cm
    for i in range(rccm.shape[0]):
        Gyt+=mass[i]*np.outer(rccm[i],rccm[i])
    Gyt/=np.sum(mass)
    S =  np.linalg.eigvals(Gyt)
    S =-np.sort(-S) # sorting in desceanding order
    Rg2 =np.sum(S)
    #Shat = Gyt-Rg2*np.identity(3)/3
    asph = S[0] -0.5*(S[1]+S[2])
    acyl = S[1]-S[2]
    k2 = (asph**2 + 0.75*acyl**2)/Rg2**2

    Rgxx_plus_yy = Gyt[0][0] + Gyt[1][1]
    Rgxx_plus_zz = Gyt[0][0] + Gyt[2][2]
    Rgyy_plus_zz = Gyt[1][1] + Gyt[2][2]

    return Ree2, Rg2,k2,asph,acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz

@jit(nopython=True, fastmath=True,parallel=True)
def dihedral_values_kernel(dih_ids,coords,dih_val):
    """Compute dihedral angles for all quadruplets in ``dih_ids``.

    Parameters
    ----------
    dih_ids : ndarray, shape (M, 4)
        Atom index quadruplets defining dihedrals.
    coords : ndarray, shape (Natoms, 3)
        Cartesian coordinates.
    dih_val : ndarray, shape (M,)
        Output array filled with dihedral angles in radians.
    """
    for i in prange(dih_ids.shape[0]):
        r0 = coords[dih_ids[i,0]]
        r1 = coords[dih_ids[i,1]]
        r2 = coords[dih_ids[i,2]]
        r3 = coords[dih_ids[i,3]]
        dih_val[i] = calc_dihedral(r0,r1,r2,r3)
    return

@jit(nopython=True,fastmath=True)
def numba_sum_wfilt(x,filt):
    """Sum elements of ``x`` where ``filt`` is true.

    Parameters
    ----------
    x : ndarray
        Input array.
    filt : ndarray of bool
        Boolean mask of the same length as ``x``.

    Returns
    -------
    float
        Sum of ``x[i]`` over indices where ``filt[i]`` is true.
    """
    s = 0
    for i in range(x.shape[0]):
        if filt[i]:
            s += x[i]
    return s

@jit(nopython=True,fastmath=True,parallel=True)
def numba_parallel_sum_wfilt(x,filt):
    """Parallel version of :func:`numba_sum_wfilt`.

    Parameters
    ----------
    x : ndarray
        Input array.
    filt : ndarray of bool
        Boolean mask of the same length as ``x``.

    Returns
    -------
    float
        Sum of ``x[i]`` over indices where ``filt[i]`` is true.
    """
    s = 0
    for i in prange(x.shape[0]):
        if filt[i]:
            s += x[i]
    return s

@jit(nopython=True,fastmath=True)
def numba_sum(x):
    """Sum elements of a 1D array using a simple loop.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    float
        Sum of all elements of ``x``.
    """
    s =0
    for i in range(x.shape[0]):
        s+=x[i]
    return s

@jit(nopython=True,fastmath=True,parallel=True)
def numba_parallel_sum(x):
    """Parallel sum of a 1D array using Numba ``prange``.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    float
        Sum of all elements of ``x``.
    """
    s =0
    for i in prange(x.shape[0]):
        s+=x[i]
    return s


class Analysis_Crystals(Analysis):
    """Placeholder subclass of :class:`Analysis` for crystalline systems.

    Currently this simply forwards to :class:`Analysis` and stores a
    ``topol`` flag; it exists to provide a hook for future
    crystal-specific analysis methods.

    Parameters
    ----------
    molecular_system
        Topology / trajectory specification forwarded to
        :class:`Analysis`.
    memory_demanding : bool, optional
        Passed through to :class:`Analysis` to control frame storage.
    topol : bool, optional
        Stored flag indicating whether topology information is
        available.
    """
    def __init__(self,molecular_system,memory_demanding=True,topol=True):
        super().__init__(molecular_system,memory_demanding)
        self.topol =True
        return


def center_of_bins(bins):
    """Return geometric centres of 1D bins.

    Parameters
    ----------
    bins : ndarray
        Array of bin edges of length ``nbins + 1``.

    Returns
    -------
    ndarray
        Array of length ``nbins`` with ``0.5 * (bins[i] + bins[i+1])``.
    """
    nbins = bins.shape[0] - 1
    r = [0.5*(bins[i]+bins[i+1]) for i in range(nbins)]
    return np.array(r)

def filt_uplow(x,yl,yup):
    """Return mask for values strictly between lower and upper bounds.

    Parameters
    ----------
    x : ndarray
        Input data.
    yl, yup : float
        Lower and upper bounds; the mask is ``(yl < x) & (x <= yup)``.
    """

    return np.logical_and(np.greater(x,yl),np.less_equal(x,yup))


def binning(x,bins):
    """Count how many values of ``x`` fall into each 1D bin.

    Parameters
    ----------
    x : ndarray
        Input data to bin.
    bins : ndarray
        Bin edges of length ``nbins + 1``.

    Returns
    -------
    ndarray of int
        Counts per bin, where bin ``i`` corresponds to
        ``(bins[i], bins[i+1]]``.
    """
    nbins = bins.shape[0]-1
    n_in_bins = np.empty(nbins,dtype=int)
    for i in np.arange(0,nbins,1,dtype=int):
        filt = filt_uplow(x,bins[i],bins[i+1])
        n_in_bins[i] = np.count_nonzero(filt)
    return n_in_bins


class maps:
    """Container for simple mapping tables used across the package.

    Attributes
    ----------
    charge_map : dict
        Mapping from internal atom-type labels to default partial
        charges.
    """
    charge_map = {'CD':-0.266,'C':0.154,'CE':0.164,
                  'hC':-0.01,'hCD':0.132,'hCE':-0.01
                      }


@jit(nopython=True,fastmath=True)
def minimum_image_relative_coords(relative_coords,box):
    """Apply minimum-image convention to relative coordinates.

    Parameters
    ----------
    relative_coords : ndarray, shape (N, 3)
        Relative displacement vectors.
    box : array_like, shape (3,)
        Box lengths in each Cartesian direction.

    Returns
    -------
    ndarray
        Array of shape ``(N, 3)`` with each displacement wrapped into
        the range ``[-box[j]/2, box[j]/2]``.
    """
    imaged_rel_coords = relative_coords.copy()
    for i in range(relative_coords.shape[0]):
        for j in range(3):
            if relative_coords[i][j] > 0.5*box[j]:
                imaged_rel_coords[i][j] -= box[j]
            elif relative_coords[i][j] < -0.5*box[j]:
                imaged_rel_coords[i][j] += box[j]
    return imaged_rel_coords


@jit(nopython=True,fastmath=True,parallel=True)
def minimum_image_distance(coords,cref,box):
        """Minimum-image distance from a reference point.

        Parameters
        ----------
        coords : ndarray, shape (N, 3)
            Coordinates of the points.
        cref : ndarray, shape (3,)
            Reference coordinate.
        box : array_like, shape (3,)
            Box lengths in each Cartesian direction.

        Returns
        -------
        ndarray
            Distances of each point to ``cref`` under minimum-image
            convention.
        """
        r = coords - cref

        for j in range(3):
            b = box[j]
            b2 = b/2
            fm = r[:,j] < - b2
            fp = r[:,j] >   b2
            r[:,j][fm] += b
            r[:,j][fp] -= b
        d = np.zeros(r.shape[0],dtype=float)
        for i in prange(r.shape[0]):
            for j in range(3):
                x = r[i,j]
                d[i] += x*x
            d[i] = np.sqrt(d[i])

        return d

@jit(nopython=True,fastmath=True,parallel=True)
def minimum_image_distance_coords(coords,cref,box):
        """Minimum-image distance and wrapped coordinates.

        Parameters
        ----------
        coords : ndarray, shape (N, 3)
            Original coordinates.
        cref : ndarray, shape (3,)
            Reference coordinate.
        box : array_like, shape (3,)
            Box lengths.

        Returns
        -------
        d : ndarray, shape (N,)
            Distances under minimum-image convention.
        imag_coords : ndarray, shape (N, 3)
            Corresponding wrapped coordinates.
        """
        r = coords - cref
        imag_coords = coords.copy()
        for j in range(3):
            b = box[j]
            b2 = b/2
            fm = r[:,j] < - b2
            fp = r[:,j] >   b2

            r[:,j][fm] += b
            imag_coords[:,j][fm] +=b

            r[:,j][fp] -= b
            imag_coords[:,j][fp] -= b
        d = np.zeros(r.shape[0],dtype=float)
        for i in prange(r.shape[0]):
            for j in range(3):
                x = r[i,j]
                d[i] += x*x
            d[i] = np.sqrt(d[i])

        return d,imag_coords

@jit(nopython=True,fastmath=True,parallel=True)
def numba_Sq2(nc,v,q,Sq):
    """Accumulate static structure factor :math:`S(q)` from vectors.

    Parameters
    ----------
    nc : int
        Number of particles.
    v : ndarray, shape (Np, 3)
        Pair displacement vectors.
    q : ndarray, shape (Nq, 3)
        Scattering vectors.
    Sq : ndarray, shape (Nq,)
        Structure-factor array updated in place.
    """
    nq = q.shape[0]
    npairs = v.shape[0]
    nc = float(nc)
    s = np.empty_like(Sq)
    qm = -q
    for i in range(nq):
        s[i] = 0.0
    for j in prange(npairs):
        s += np.cos(np.dot(qm,v[j]))
    Sq += 2*s/nc

    return


@jit(nopython=True,fastmath=True,parallel=True)
def pair_vects(coords,box,v):
    """Compute all unique pair vectors under PBC.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Particle coordinates.
    box : array_like, shape (3,)
        Box lengths.
    v : ndarray, shape (N*(N-1)/2, 3)
        Output array filled with minimum-image pair displacement
        vectors.
    """
    n = coords.shape[0]
    coords = implement_pbc(coords,box)
    for i in prange(n):
        rel_coords = coords[i] - coords[i+1:]
        rc = minimum_image_relative_coords(rel_coords,box)
        idx_i = i*n
        for k in range(0,i+1):
            idx_i-=k
        for j in range(rc.shape[0]):
            v[idx_i+j] = rc[j]
    return



@jit(nopython=True,fastmath=True,parallel=True)
def pair_dists(coords,box,dists):
    """Compute all unique pair distances under PBC.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Particle coordinates.
    box : array_like, shape (3,)
        Box lengths.
    dists : ndarray, shape (N*(N-1)/2,)
        Output array filled with pair distances.
    """
    n = coords.shape[0]
    for i in prange(n):
        rel_coords = coords[i] - coords[i+1:]
        rc = minimum_image_relative_coords(rel_coords,box)
        dist = np.sum(rc*rc,axis=1)**0.5
        idx_i = i*n
        for k in range(0,i+1):
            idx_i-=k
        for j in range(rc.shape[0]):
            dists[idx_i+j] = dist[j]
    return

@jit(nopython=True,fastmath=True,parallel=True)
def numba_coordination(coords1,coords2,box,maxdist,coordination):
    """Compute coordination numbers within a cutoff between two sets.

    Parameters
    ----------
    coords1 : ndarray, shape (N1, 3)
        First set of coordinates.
    coords2 : ndarray, shape (N2, 3)
        Second set of coordinates.
    box : array_like, shape (3,)
        Box lengths.
    maxdist : float
        Cutoff distance.
    coordination : ndarray, shape (N1,)
        Output array with the number of neighbours per particle in
        ``coords1``.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    for i in prange(n1):
        rel_coords = coords1[i] - coords2
        rc = minimum_image_relative_coords(rel_coords,box)
        dist = np.sqrt(np.sum(rc*rc,axis=1))
        for j in range(n2):
            if dist[j]<maxdist:
                coordination[i]+=1

    return

@jit(nopython=True,fastmath=True,parallel=True)
def pair_dists_general(coords1,coords2,box,dists):
    """Compute all pair distances between two sets under PBC.

    Parameters
    ----------
    coords1 : ndarray, shape (N1, 3)
        First set of coordinates.
    coords2 : ndarray, shape (N2, 3)
        Second set of coordinates.
    box : array_like, shape (3,)
        Box lengths.
    dists : ndarray, shape (N1*N2,)
        Output array where block ``i`` contains distances from
        ``coords1[i]`` to all ``coords2`` points.
    """
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    for i in prange(n1):
        rel_coords = coords1[i] - coords2
        #rc=rel_coords
        rc = minimum_image_relative_coords(rel_coords,box)
        dists[i*n2:(i+1)*n2] = np.sum(rc*rc,axis=1)**0.5
        #for j in range(n2):
         #   dists[i*n2+j] = dist[j]
    return

@jit(nopython=True,fastmath=True)
def CM(coords,mass):
    """Centre of mass of a group of atoms.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Atomic coordinates.
    mass : ndarray, shape (N,)
        Atomic masses.

    Returns
    -------
    ndarray, shape (3,)
        Centre-of-mass position.
    """
    cm = np.sum(mass*coords.T,axis=1)/mass.sum()
    return cm

@jit(nopython=True,fastmath=True)
def implement_pbc(coords,boxsize):
    """Wrap coordinates back into the primary simulation box.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Possibly unwrapped coordinates.
    boxsize : array_like or float
        Box lengths; broadcasting rules apply.

    Returns
    -------
    ndarray
        Coordinates mapped into ``[0, boxsize)``.
    """
    cn = coords%boxsize
    return cn

@jit(nopython=True,fastmath=True)
def square_diff(x,c):
    """Relative squared deviation ``((x - c) / c)**2``.

    Parameters
    ----------
    x, c : float or ndarray
        Value and reference.

    Returns
    -------
    float or ndarray
        Relative squared deviation.
    """
    return ((x-c)/c)**2
@jit(nopython=True,fastmath=True)
def norm2(r):
    """Euclidean norm of a 1D vector.

    Parameters
    ----------
    r : ndarray, shape (N,)
        Input vector.

    Returns
    -------
    float
        :math:`|r|`.
    """
    x =0
    for i in range(r.shape[0]):
        x += r[i]*r[i]
    return x**0.5
@jit(nopython=True,fastmath=True)
def norm2_axis1(r):
    """Euclidean norm along axis 1 for a 2D array.

    Parameters
    ----------
    r : ndarray, shape (N, D)
        Input array.

    Returns
    -------
    ndarray, shape (N,)
        Norm of each row.
    """
    return (r*r).sum(axis=1)**0.5


@jit(nopython=True,fastmath=True)
def calc_dist(r1,r2):
    """Distance between two 3D points.

    Parameters
    ----------
    r1, r2 : ndarray, shape (3,)
        Cartesian coordinates.

    Returns
    -------
    float
        Euclidean distance ``|r2 - r1|``.
    """
    r = r2 - r1
    d = np.sqrt(np.dot(r,r))
    return d

@jit(nopython=True,fastmath=True)
def calc_angle(r1,r2,r3):
    """Bond angle formed by three points.

    Parameters
    ----------
    r1, r2, r3 : ndarray, shape (3,)
        Positions of atoms 1, 2, 3.

    Returns
    -------
    float
        Angle at ``r2`` in radians.
    """
    d1 = r1 -r2 ; d2 = r3-r2
    nd1 = np.sqrt(np.dot(d1,d1))
    nd2 = np.sqrt(np.dot(d2,d2))
    cos_th = np.dot(d1,d2)/(nd1*nd2)
    return np.arccos(cos_th)

@jit(nopython=True,fastmath=True)
def calc_dihedral(r1,r2,r3,r4):
    """Dihedral angle defined by four points.

    Parameters
    ----------
    r1, r2, r3, r4 : ndarray, shape (3,)
        Consecutive atom coordinates.

    Returns
    -------
    float
        Dihedral angle in radians using the IUPAC convention.
    """
    d1 = r2-r1
    d2 = r3-r2
    d3 = r4-r3
    c1 = np.cross(d1,d2)
    c2 = np.cross(d2,d3)
    n1 = c1/np.sqrt(np.dot(c1,c1))
    n2 = c2/np.sqrt(np.dot(c2,c2))
    m1= np.cross(n1,d2/np.sqrt(np.dot(d2,d2)))
    x= np.dot(n1,n2)
    y= np.dot(m1,n2)
    dihedral = np.arctan2(y, x)
    return dihedral





class maps:
    charge_map = {'CD':-0.266,'C':0.154,'CE':0.164,
                  'hC':-0.01,'hCD':0.132,'hCE':-0.01
                      }

    elements_mass = {'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,\
                 'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,\
                 'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,\
                 'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,\
                 'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,\
                 'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,\
                 'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,\
                 'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,\
                 'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,\
                 'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,\
                 'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,\
                 'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,\
                 'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,\
                 'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,\
                 'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,\
                 'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,\
                 'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,\
                 'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,\
                 'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,\
                 'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,\
                 'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247, 'Bk' : 247,\
                 'Ct' : 251, 'Es' : 252, 'Fm' : 257, 'Md' : 258, 'No' : 259,\
                 'Lr' : 262, 'Rf' : 261, 'Db' : 262, 'Sg' : 266, 'Bh' : 264,\
                 'Hs' : 269, 'Mt' : 268, 'Ds' : 271, 'Rg' : 272, 'Cn' : 285,\
                 'Nh' : 284, 'Fl' : 289, 'Mc' : 288, 'Lv' : 292, 'Ts' : 294,\
                 'Og' : 294}


@jit(nopython=True,fastmath=True)
def numba_dipoles(pc,coords,segargs,dipoles):
    """Compute segment dipole moments from partial charges and positions.

    Parameters
    ----------
    pc : ndarray
        Partial charges per atom.
    coords : ndarray, shape (Natoms, 3)
        Atomic coordinates.
    segargs : ndarray
        Index lists/arrays defining segments (one per dipole).
    dipoles : ndarray, shape (Nseg, 3)
        Output array filled in place with segment dipole vectors.
    """
    n = segargs.shape[0]
    for i in prange(n):
        cargs = segargs[i]
        dipoles[i] = np.sum(pc[cargs]*coords[cargs],axis=0)
    return
@jit(nopython=True,fastmath=True,parallel=True)
def numba_isin(x1,x2,f):
    """Vectorised membership test ``x1[i] in x2`` written into mask ``f``.

    Parameters
    ----------
    x1 : ndarray
        Values to test.
    x2 : ndarray
        Reference set of values.
    f : ndarray of bool
        Output mask with ``f[i]`` set to True if ``x1[i]`` is found in
        ``x2``.
    """
    for i in prange(x1.shape[0]):
        for x in x2:
            if x1[i] == x:
                f[i] = True
    return
@jit(nopython=True,fastmath=True,parallel=True)
def numba_CM(coords,ids,mass,cm):
    """Compute centres-of-mass for a collection of index groups.

    Parameters
    ----------
    coords : ndarray, shape (Natoms, 3)
        Atomic coordinates.
    ids : ndarray of object or 2D int array
        Per-group atom indices, as used by :func:`CM`.
    mass : ndarray
        Atomic masses.
    cm : ndarray, shape (Ngroups, 3)
        Output array filled with centres-of-mass for each group.
    """
    for i in prange(ids.shape[0]):
        ji = ids[i]
        cm[i] = CM(coords[ji],mass[ji])
    return

@jit(nopython=True,fastmath=True,parallel=True)
def numba_elementwise_minimum(x1,x2):
    """Element-wise minimum of two arrays.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays of the same shape.

    Returns
    -------
    ndarray
        Array ``xmin`` where ``xmin[i] = min(x1[i], x2[i])``.
    """
    xmin = np.empty_like(x1)
    for i in prange(x1.shape[0]):
        if x1[i]<x2[i]:
            xmin[i] = x1[i]
        else:
            xmin[i] = x2[i]
    return xmin
