# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import pandas as pd
import sympy
import re
pg.setConfigOptions(antialias=True)
#pd.options.display.max_rows = 999

# ideas
# front culling of ewald sphere and reciprocal lattice points
# add detector axes, poni, distance (modules?)

# todo
# gui toggles behave erratically
# detector rotation is off upon resetting the crystal
# Add show parcor toggle button to sfx gui

__version__ = 'v0.0.5, 03.11.2025'

def signed_angle(v1: np.ndarray, v2: np.ndarray, up_reference: np.ndarray) -> float:
        cross = np.cross(v1, v2)
        dotprod = np.dot(v1, v2)
        sign = -1 if np.dot(cross, up_reference) < 0 else 1
        return np.arctan2(np.linalg.norm(cross), dotprod) * sign

class Quaternion:
    """
    Copyright (C) 2025 Karl O. R. Juul (kjuul@chem.au.dk) - All Rights Reserved
    This code is part of a larger project that will be published soon.
    """
    __author__ = "Karl O. R. Juul (kjuul@chem.au.dk)"
    # Good resource: https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    def __init__(self, w, x, y, z) -> None:
        self._w = w
        self._x = x
        self._y = y
        self._z = z
    
    # -- Basic getters -- #
    def get(self) -> np.ndarray:
        '''Get an array of the quaternion coordinates (w, x, y, z).'''
        return np.array([self._w, self._x, self._y, self._z])
    
    def real(self) -> float:
        '''The selection function. Returns the real part of the quaternion (i.e., returns w).'''
        return self._w # Same as (q + q.conjugate()) / 2

    def imag(self) -> np.ndarray:
        '''Returns the imaginary part of the quaternion (i.e., returns x, y, z).'''
        return self.get()[1:] # same as (q - q.conjugate()) / 2
    
    def conjugate(self) -> 'Quaternion':
        w, x, y, z = self.get()
        return Quaternion(w, -x, -y, -z)
    
    def identity() -> 'Quaternion':
        return Quaternion(1, 0, 0, 0)

    # -- Operators -- #
    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(*(self.get() + other.get()))
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        return self + (-1) * other
    
    def __truediv__(self, other) -> 'Quaternion':
        if isinstance(other, float) or isinstance(other, int):
            return self.__mul__(other**-1)
        return self.__mul__(other.inverse())

    def __mul__(self, other) -> 'Quaternion':
        if isinstance(other, Quaternion):
            w0, x0, y0, z0 = self.get()
            w1, x1, y1, z1 = other.get()
            return Quaternion(-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                               x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                              -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                               x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)
        elif isinstance(other, float) or isinstance(other, int):
            return Quaternion(*(self.get() * other))

    def __rmul__(self, other) -> 'Quaternion':
        return self.__mul__(other)
    
    def __pow__(self, other: float) -> 'Quaternion':
        return self.pow(other)
        
    def exp(self) -> 'Quaternion':
        # https://math.stackexchange.com/questions/939229/unit-quaternion-to-a-scalar-power
        if np.all(np.isclose(self.get(), 0)):
            return Quaternion.identity()
        
        w: float = self.real()
        u: Quaternion = (self - self.conjugate()) * 0.5
        v = u.imag()
        return np.exp(w) * Quaternion.from_axis_rotation(v / np.linalg.norm(v), 2 * u.magnitude())

    def log(self) -> 'Quaternion':
        # https://math.stackexchange.com/questions/939229/unit-quaternion-to-a-scalar-power
        '''Natural logarithm.'''
        u: Quaternion = (self - self.conjugate()) * 0.5
        u /= u.magnitude()
        theta = np.arccos(self.real())
        return Quaternion(0, *(theta * u.imag()))

    def pow(self, scalar: float) -> 'Quaternion':
        # https://math.stackexchange.com/questions/939229/unit-quaternion-to-a-scalar-power
        return Quaternion.exp(scalar * Quaternion.log(self))
    
    def slerp(q0: 'Quaternion', q1: 'Quaternion', t: float) -> 'Quaternion':
        return q0 * (q0.inverse() * q1)**t
        
    def magnitude(self) -> float:
        '''The magnitude of the quaternion. Rotation quaternions should have unit magnitude.'''
        return np.sqrt(self.sqr_magnitude())

    def sqr_magnitude(self) -> float:
        w, x, y, z = self.get()
        return w * w + x * x + y * y + z * z # Same as self * self.congugate()

    def inverse(self) -> 'Quaternion':
        '''Note that for rotation quaternions, the inverse equals the conjugate.'''
        return self.conjugate() / self.sqr_magnitude()

    # -- Conversions -- #
    def to_euler(self) -> tuple[float]:
        '''https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles'''
        w, x, y, z = self.get()
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = np.sqrt(1 + 2 * round(w * y - x * z, 2))
        cosp = np.sqrt(1 - 2 * round(w * y - x * z, 2))
        pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2

        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def from_euler(roll: float, pitch: float, yaw: float) -> 'Quaternion':
        '''roll (x), pitch (y), yaw (z), angles are in radians.
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles'''
        # Abbreviations for the various angular functions
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion(w, x, y, z)
    
    def from_axis_rotation(axis: np.ndarray, angle: float) -> 'Quaternion':
        '''axis is a unit vector to rotate about. Angle in radians.'''
        # https://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
        axis = axis/np.linalg.norm(axis)
        a2 = angle * 0.5
        x, y, z = axis
        return Quaternion(np.cos(a2), np.sin(a2)*x, np.sin(a2)*y, np.sin(a2)*z)
    
    def random(stream: np.random.Generator) -> 'Quaternion':
        '''Generates a uniform random quaternion'''
        # https://stackoverflow.com/questions/38978441/creating-uniform-random-quaternion-and-multiplication-of-two-quaternions
        # Adapted from James J. Kuffner, 2004
        values = stream.random(3)
        # Magitude ensures that the total length is 1.
        magnutude = np.sqrt(values[0])
        rest_magnitude = np.sqrt(1.0 - values[0])
        angle1 = 2*np.pi*values[1]
        angle2 = 2*np.pi*values[2]
        w = np.cos(angle2)*magnutude
        x = np.sin(angle1)*rest_magnitude
        y = np.cos(angle1)*rest_magnitude
        z = np.sin(angle2)*magnutude
        return Quaternion(w, x, y, z)
        
    def apply_rotation_to_vector(self, vector: np.ndarray) -> np.ndarray:
        '''https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion'''
        v: Quaternion = Quaternion(0, *vector)
        vrot: Quaternion = self * v * self.conjugate()
        return vrot.get()[1:] # Return the x, y, z components of the resulting quaternion.
    
    def to_rotation_matrix(self) -> np.ndarray:
        # https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
        w, x, y, z = self.get()
        ww, wx, wy, wz = w * w, w * x, w * y, w * z
        xx, xy, xz = x * x, x * y, x * z
        yy, yz = y * y, y * z
        zz = z * z
        return 2*np.array([[ww + xx - 0.5, xy - wz, xz + wy],
                           [xy + wz, ww + yy - 0.5, yz - wx],
                           [xz - wy, yz + wx, ww + zz - 0.5]])
    
    def get_axis_angle(self) -> tuple[np.ndarray, float]:
        '''Returns the axis (unit vector) and angle (in radians) of the rotation.'''
        angle = 2 * np.arccos(self.real())
        s = np.sqrt(1 - self.real()**2)
        if s < 1E-20: # If s close to zero, direction of axis not important
            return np.array([1, 0, 0]), angle
        return self.imag() / s, angle

    def get_look_direction_rotation(self, up_vector: np.ndarray, look_at_vector: np.ndarray) -> 'Quaternion':
        # To look in a direction, first rotating to face the desired direction.
        # Then tilt view to have the correct direction up.

        # Unit vectors pointing in up and forward in the end-coordinate system.
        up_axis = up_vector / np.linalg.norm(up_vector)
        look_at_axis = look_at_vector / np.linalg.norm(look_at_vector)
        forward_vector = np.array([1,0,0]) # Global forward.
        up_vector = np.array([0,0,1]) # Global up.

        # To align forwards, first find the rotation vector (one perpendicular to the plane spanned by global forward and the final forward).
        # This vector is normal to the plane spanned by df and final forward.
        df = look_at_axis - forward_vector
        # Find the axis to rotate about in order to align the forwards.
        if np.all(np.isclose(df, 0)):
            # If already looking forward, there will be no rotation. This is just decor for the Quaternion.from_axis_rotation to work
            # as it requires that the axis has unit length.
            forward_align_axis = up_axis
        else:
            # Get a vector perpendicular to the plane of rotation (rotation vector).
            forward_align_axis = np.cross(look_at_axis, df)
            forward_align_axis /= np.linalg.norm(forward_align_axis)
        
        # Rotate to align forwards.
        forward_angle = signed_angle(forward_vector, look_at_axis, up_axis) # Angle between the forward and global forward.
        q1: Quaternion = Quaternion.from_axis_rotation(forward_align_axis, forward_angle)

        # Rotate around forward (which is now the same as Transform.forward in local space) to align up.
        up_angle = signed_angle(q1.conjugate().apply_rotation_to_vector(up_vector), up_axis, forward_align_axis)
        q2: Quaternion = Quaternion.from_axis_rotation(forward_vector, up_angle)
        return q2 * q1

    # -- To string -- #
    def __str__(self) -> str:
        return str(self.get())
    
    def __repr__(self) -> str:
        return self.__str__()

def cart_from_cell(cell):
    """
    Calculate a,b,c vectors in cartesian system from lattice constants.
    :param cell: a,b,c,alpha,beta,gamma lattice constants.
    :return: a, b, c vector
    """
    if cell.shape != (6,):
        raise ValueError('Lattice constants must be 1d array with 6 elements')
    a, b, c = cell[:3]*1E-10
    alpha, beta, gamma = np.radians(cell[3:])
    av = np.array([a, 0, 0], dtype=float)
    bv = np.array([b * np.cos(gamma), b * np.sin(gamma), 0], dtype=float)
    # calculate vector c
    x = np.cos(beta)
    y = (np.cos(alpha) - x * np.cos(gamma)) / np.sin(gamma)
    z = np.sqrt(1. - x**2. - y**2.)
    cv = np.array([x, y, z], dtype=float)
    cv /= np.linalg.norm(cv)
    cv *= c
    return av, bv, cv

def matrix_from_cell(cell):
    """
    Calculate transform matrix from lattice constants.
    :param cell: a,b,c,alpha,beta,gamma lattice constants in
                            angstrom and degree.
    :param lattice_type: lattice type: P, A, B, C, H
    :return: transform matrix A = [a*, b*, c*]
    """
    cell = np.array(cell)
    av, bv, cv = cart_from_cell(cell)
    a_star = (np.cross(bv, cv)) / (np.cross(bv, cv).dot(av))
    b_star = (np.cross(cv, av)) / (np.cross(cv, av).dot(bv))
    c_star = (np.cross(av, bv)) / (np.cross(av, bv).dot(cv))
    A = np.zeros((3, 3), dtype='float')  # transform matrix
    A[:, 0] = a_star
    A[:, 1] = b_star
    A[:, 2] = c_star
    return np.round(A,6)

def calc_hkls(A, res):
    q_cutoff = 1. / res
    max_h = np.ceil(q_cutoff / np.linalg.norm(A[:,0])).astype(np.int8)
    max_k = np.ceil(q_cutoff / np.linalg.norm(A[:,1])).astype(np.int8)
    max_l = np.ceil(q_cutoff / np.linalg.norm(A[:,2])).astype(np.int8)
    # hkl grid
    hh = np.arange(-max_h, max_h+1)
    kk = np.arange(-max_k, max_k+1)
    ll = np.arange(-max_l, max_l+1)
    hs, ks, ls = np.meshgrid(hh, kk, ll)
    hkl = np.ones((hs.size, 3), dtype=np.int8)
    hkl[:,0] = hs.reshape((-1))
    hkl[:,1] = ks.reshape((-1))
    hkl[:,2] = ls.reshape((-1))
    # remove high resolution hkls
    hkl = hkl[(np.linalg.norm(hkl @ A.T, axis=1) <= q_cutoff)]
    hkl = np.delete(hkl, len(hkl)//2, 0)
    return hkl

def get_sym_mat_from_coords(coords_repr):
    '''Takes string with coordinate representations of 
    symmetry elements and translates into array of
    matrices M, which is returned.'''
    # Turn coords repr into a list on the form ["(x,y,z)", ...]
    data = re.split(r"\)\(" , coords_repr)
    new_data = [data[0]+")"]+["(" + i +")" for i in data[1:-1]] + ["("+data[-1]]
    # Turn into a list on the form [["x", "y", "z"], ...]
    new_data = [re.split(',',a.replace('(','').replace(')','')) for a in new_data]
    x,y,z = sympy.symbols('x,y,z')
    M = []
    for vector in new_data:
        m = np.zeros((3,3))
        for i in range(3):
            #For element 1 extract m11,m12 and m13 and so forth for element 2 and 3
            xpr=sympy.parsing.sympy_parser.parse_expr(vector[i])
            a =sympy.Poly(xpr,x)
            b =sympy.Poly(xpr,y)
            c =sympy.Poly(xpr,z)
            try:
                m[i,0] = int(a.coeffs()[0].evalf())
            except TypeError: #There is no coeff. for x - no problem, just keep 0 in m.
                pass
            try:
                m[i,1] = int(b.coeffs()[0].evalf())
            except TypeError: #There is no coeff. for y 
                pass
            try:
                m[i,2] = int(c.coeffs()[0].evalf())
            except TypeError: #There is no coeff. for z 
                pass
        #append matrix representation of symmetry to 
        M.append(m)       
    return M

def PolCorFromkfs(kf: np.array, pol_deg: float) -> np.array:
    '''Get the polarization correction for a group of non-rotated scattered vectors.
    Set pol_deg=0.5 for unpolarized beam and pol. plane normal should be any vector perp. to ki.'''
    # Based on J. Appl. Cryst. (1988). 21, 916-924. Eqn. in Data correction and scaling, section (b).
    PolPlaneNormal = np.array([0,0,1]) # Normal to polarization and direction of propagation.
    dot_pol_plane_normal_kds = PolPlaneNormal @ np.transpose(kf) 
    ki = np.array([-1,0,0])
    dot_kfs_ki = ki @ np.transpose(kf)
    norm_ki = 1
    norm_kfs = np.linalg.norm(kf,axis=1)
    
    pc=((1-2*pol_deg)*(1-(dot_pol_plane_normal_kds / norm_kfs)**2) + pol_deg*(1+(dot_kfs_ki / (norm_kfs * norm_ki))**2) )
    return pc # Divide by this.

def calc_angles(v1, v2):
    return np.arccos(np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1))).reshape(-1,1)

def correction_partiality_geometric(kds, ewald_radius, spectral_width=1, rlp_size=1):
    diff_beam_2theta = calc_angles(kds, np.array([[0,0,1]]))
    diff_beam_length = np.linalg.norm(kds, axis=1).reshape(-1,1)
    corrected_spectral_width = spectral_width - spectral_width * np.cos(diff_beam_2theta)
    scale = np.sqrt(2.0 * corrected_spectral_width * rlp_size / (corrected_spectral_width**2 + rlp_size**2))
    overlap =  - (ewald_radius - diff_beam_length)**2 / (2.0 * (corrected_spectral_width**2 + rlp_size**2))
    return scale * np.exp(overlap)

def overlap_nd(mu1, mu2, s1, s2, dim=None):
    """Overlap of two isotropic n-D normalized Gaussians.
    mu1, mu2: array-like centers (same length = n) or scalars when dim given
    s1, s2: scalar widths (isotropic)
    If dim is provided and mu1/mu2 are scalars, uses that dimension.
    """
    import numpy as np
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    if dim is None:
        D2 = np.sum((mu1 - mu2)**2)
        n = mu1.size
    else:
        D2 = float(mu1 - mu2)**2
        n = int(dim)
    pref = (2.0 * s1 * s2 / (s1**2 + s2**2))**(n/2.0)
    expo = - D2 / (2.0 * (s1**2 + s2**2))
    return pref * np.exp(expo)

class Visualizer(QtWidgets.QMainWindow):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f'Reciprocal Playground ({__version__})')
        central_widget = QtWidgets.QSplitter()
        central_widget.setHandleWidth(10)
        self.setCentralWidget(central_widget)
        self.sigKeyPress.connect(self.keyPressEvent)

        parameter_scroll = QtWidgets.QScrollArea()
        parameter_scroll.setContentsMargins(0,0,0,0)
        #parameter_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #parameter_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        parameter_scroll.setWidgetResizable(True)
        central_widget.addWidget(parameter_scroll)
        self.parameter_box = QtWidgets.QFrame()
        self.parameter_box.setContentsMargins(0,0,0,0)
        self.params_layout = QtWidgets.QVBoxLayout()
        self.params_layout.setSpacing(5)
        self.params_layout.setContentsMargins(0,0,0,0)
        self.params_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.parameter_box.setLayout(self.params_layout)
        parameter_scroll.setWidget(self.parameter_box)

        self.gl3d = gl.GLViewWidget()
        self.gl3d.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.gl3d.setBackgroundColor((10, 20, 20))
        central_widget.addWidget(self.gl3d)

        self.DEBUG = False

        self.orth_proj = False
        self.show_cell_axs = True
        self.show_cell_axs_lbl = True
        self.show_gon_axs = True
        self.show_gon_axs_lbl = True
        self.show_cell_outline = True
        self.show_detector = True
        self.show_ewald = True
        self.show_text_hkls = True

        self.show_scat_vecs = True
        self.show_diff_vecs = True
        self.show_scat_latt = True
        self.show_scat_scan = True
        self.show_scat_data = True
        self.show_scat_symm = True
        
        # SFX
        self.sfx_keep_all_data = False
        self.show_sfx_parcor = False

        self.setGeometry(0, 0, 1920, 1080)
        #self.setGeometry(0, 0, 1280, 768)
        self.plot_fnt_tab = QtGui.QFont('Helvetica', 14)
        self.plot_fnt_lbl = QtGui.QFont('Helvetica', 18)
        self.plot_fnt_hkl = QtGui.QFont('Helvetica', 20)
        self.plot_line_width = 1
        self.plot_line_width_thick = 3

        # Colorsself.color_scat_latt
        self.color_scat_latt = QtGui.QColor(80, 80, 80, 50)
        self.color_scat_symm = QtGui.QColor(111, 164, 175, 200)
        self.color_scat_data = QtGui.QColor(129, 24, 68, 200)
        self.color_scat_scan = QtGui.QColor(245, 173, 24, 200)
        self.color_labl_hkls = QtGui.QColor(200, 200, 200, 255)
        #self.color_labl_hkls = QtGui.QColor(255, 0, 96, 255)
        self.color_scat_vecs = QtGui.QColor(245, 173, 24, 200)
        self.color_diff_vecs = QtGui.QColor(255, 0, 96, 200)
        self.color_detector = QtGui.QColor(128, 128, 128, 128)
        self.color_scat_patt = QtGui.QColor(255, 0, 96, 200)
        self.color_gui_buttons = QtGui.QColor(0, 106, 103, 255)

        self.par = {'sample_cell_a':6,
                    'sample_cell_b':6,
                    'sample_cell_c':6,
                    'sample_cell_alpha':90,
                    'sample_cell_beta':90,
                    'sample_cell_gamma':90,
                    'sample_point_group':'-1',
                    'sample_orientation':None, # None for random
                    'det_poni_x':None,
                    'det_poni_y':None,
                    'det_distance':0.152,
                    'det_pix_s':75.0E-6,
                    'det_pix_x':2068,
                    'det_pix_y':2162,
                    'wavelength':0.59040E-10,# 21.0 keV
                    #'wavelength':0.56356E-10,# 22.0 keV"
                    'ewald_offset':0.01,
                    'max_resolution':None,
                    'scan_axis':'Omega',
                    'scan_step':0.10,
                    'scan_range':90,
                    'scan_speed':25,
                    'gon_phi_axs':np.array([1,0,0]),
                    'gon_chi_axs':np.array([0,0,1]),
                    'gon_omg_axs':np.array([1,0,0]),
                    'gon_tth_axs':np.array([1,0,0]),
                    'gon_ttv_axs':np.array([0,1,0]),
                    'gon_tth_ang':0.0,
                    'gon_omg_ang':0.0,
                    'gon_chi_ang':0.0,
                    'gon_phi_ang':0.0,
                    'plot_max_hkl_labels':50,
                    'sfx_spectral_width':None,
                    'sfx_rlp_size':None,}

        # Set camera
        self.gl3d.setCameraParams(distance=75000, fov=60, elevation=-45, azimuth=0)
        self.gui_set_projection(toggle=self.orth_proj)
        # Set plot detector distance
        # Graphical glitch when getting too close
        # Scattering pattern is misaligned and reaches
        # out further than the detector
        self.plot_det_dist = 100
        if self.par['max_resolution'] is None:
            self.par['max_resolution'] = self.par['wavelength'] / 2
        # Set default poni to center of detector if None
        if self.par['det_poni_x'] is None:
            self.par['det_poni_x'] = self.par['det_pix_x'] * self.par['det_pix_s'] / 2
        if self.par['det_poni_y'] is None:
            self.par['det_poni_y'] = self.par['det_pix_y'] * self.par['det_pix_s'] / 2
        if self.par['sfx_spectral_width'] is None:
            self.par['sfx_spectral_width'] = 2 * self.par['ewald_offset']
        if self.par['sfx_rlp_size'] is None:
            self.par['sfx_rlp_size'] = 2 * self.par['ewald_offset']

        # Get initial orientation matrix
        self.get_orientation_matrix(self.par['sample_orientation'])

        # symmetry operations
        self.sym_dict = {'1': '(x,y,z)',
                         '-1': '(x,y,z)(-x,-y,-z)',
                         '2': '(x,y,z)(-x,y,-z)',
                         'm': '(x,y,z)(x,-y,z)',
                         '2/m': '(x,y,z)(-x,y,-z)(-x,-y,-z)(x,-y,z)',
                         '222': '(x,y,z)(-x,-y,z)(-x,y,-z)(x,-y,-z)',
                         'mm2': '(x,y,z)(-x,-y,z)(x,-y,z)(-x,y,z)',
                         'mmm': '(x,y,z)(-x,-y,z)(-x,y,-z)(x,-y,-z)(-x,-y,-z)(x,y,-z)(x,-y,z)(-x,y,z)',
                         '4': '(x,y,z)(-x,-y,z)(-y,x,z)(y,-x,z)',
                         '-4': '(x,y,z)(-x,-y,z)(y,-x,-z)(-y,x,-z)',
                         '4/m': '(x,y,z)(-x,-y,z)(-y,x,z)(y,-x,z)(-x,-y,-z)(x,y,-z)(y,-x,-z)(-y,x,-z)',
                         '422': '(x,y,z)(-x,-y,z)(-y,x,z)(y,-x,z)(-x,y,-z)(x,-y,-z)(y,x,-z)(-y,-x,-z)',
                         '4mm': '(x,y,z)(-x,-y,z)(-y,x,z)(y,-x,z)(x,-y,z)(-x,y,z)(-y,-x,z)(y,x,z)',
                         '-42m': '(x,y,z)(-x,-y,z)(y,-x,-z)(-y,x,-z)(-x,y,-z)(x,-y,-z)(-y,-x,z)(y,x,z)',
                         '4/mmm': '(x,y,z)(-x,-y,z)(-y,x,z)(y,-x,z)(-x,y,-z)(x,-y,-z)(y,x,-z)(-y,-x,-z)(-x,-y,-z)(x,y,-z)(y,-x,-z)(-y,x,-z)(x,-y,z)(-x,y,z)(-y,-x,z)(y,x,z)',
                         '3': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)',
                         '-3': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-x,-y,-z)(y,-x+y,-z)(x-y,x,-z)',
                         '32': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-y,-x,-z)(-x+y,y,-z)(x,x-y,-z)',
                         '32(321)': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(y,x,-z)(x-y,-y,-z)(-x,-x+y,-z)',
                         '3m': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-y,-x,z)(-x+y,y,z)(x,x-y,z)',
                         '3m(31m': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(y,x,z)(x-y,-y,z)(-x,-x+y,z)',
                         '-3m': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-y,-x,-z)(-x+y,y,-z)(x,x-y,-z)(-x,-y,-z)(y,-x+y,-z)(x-y,x,-z)(y,x,z)(x-y,-y,z)(-x,-x+y,z)',
                         '-3m1':'(x,y,z)(-y,x-y,z)(-x+y,-x,z)(y,x,-z)(x-y,-y,-z)(-x,-x+y,-z)(-x,-y,-z)(y,-x+y,-z)(x-y,x,-z)(-y,-x,z)(-x+y,y,z)(x,x-y,z)',
                         '6': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-x,-y,z)(y,-x+y,z)(x-y,x,z)',
                         '-6': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(x,y,-z)(-y,x-y,-z)(-x+y,-x,-z)',
                         '6/m': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-x,-y,z)(y,-x+y,z)(x-y,x,z)(-x,-y,-z)(y,-x+y,-z)(x-y,x,-z)(x,y,-z)(-y,x-y,-z)(-x+y,-x,-z)' ,
                         '622': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-x,-y,z)(y,-x+y,z)(x-y,x,z)(y,x,-z)(x-y,-y,-z)(-x,-x+y,-z)(-y,-x,-z)(-x+y,y,-z)(x,x-y,-z)',
                         '6mm': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-x,-y,z)(y,-x+y,z)(x-y,x,z)(-y,-x,z)(-x+y,y,z)(x,x-y,z)(y,x,z)(x-y,-y,z)(-x,-x+y,z)',
                         '-6m2': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(x,y,-z)(-y,x-y,-z)(-x+y,-x,-z)(-y,-x,z)(-x+y,y,z)(x,x-y,z)(-y,-x,-z)(-x+y,y,-z)(x,x-y,-z)',
                         '-6m2(-62m)': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(x,y,-z)(-y,x-y,-z)(-x+y,-x,-z)(y,x,-z)(x-y,-y,-z)(-x,-x+y,-z)(y,x,z)(x-y,-y,z)(-x,-x+y,z)',
                         '6/mmm': '(x,y,z)(-y,x-y,z)(-x+y,-x,z)(-x,-y,z)(y,-x+y,z)(x-y,x,z)(y,x,-z)(x-y,-y,-z)(-x,-x+y,-z)(-y,-x,-z)(-x+y,y,-z)(x,x-y,-z)(-x,-y,-z)(y,-x+y,-z)(x-y,x,-z)(x,y,-z)(-y,x-y,-z)(-x+y,-x,-z)(-y,-x,z)(-x+y,y,z)(x,x-y,z)(y,x,z)(x-y,-y,z)(-x,-x+y,z)',
                         '23': '(x,y,z)(-x,-y,z)(-x,y,-z)(x,-y,-z)(z,x,y)(z,-x,-y)(-z,-x,y)(-z,x,-y)(y,z,x)(-y,z,-x)(y,-z,-x)(-y,-z,x)',
                         'm-3': '(x,y,z)(-x,-y,z)(-x,y,-z)(x,-y,-z)(z,x,y)(z,-x,-y)(-z,-x,y)(-z,x,-y)(y,z,x)(-y,z,-x)(y,-z,-x)(-y,-z,x)(-x,-y,-z)(x,y,-z)(x,-y,z)(-x,y,z)(-z,-x,-y)(-z,x,y)(z,x,-y)(z,-x,y)(-y,-z,-x)(y,-z,x)(-y,z,x)(y,z,-x)',
                         '432': '(x,y,z)(-x,-y,z)(-x,y,-z)(x,-y,-z)(z,x,y)(z,-x,-y)(-z,-x,y)(-z,x,-y)(y,z,x)(-y,z,-x)(y,-z,-x)(-y,-z,x)(y,x,-z)(-y,-x,-z)(y,-x,z)(-y,x,z)(x,z,-y)(-x,z,y)(-x,-z,-y)(x,-z,y)(z,y,-x)(z,-y,x)(-z,y,x)(-z,-y,-x)',
                         '-43m': '(x,y,z)(-x,-y,z)(-x,y,-z)(x,-y,-z)(z,x,y)(z,-x,-y)(-z,-x,y)(-z,x,-y)(y,z,x)(-y,z,-x)(y,-z,-x)(-y,-z,x)(y,x,z)(-y,-x,z)(y,-x,-z)(-y,x,-z)(x,z,y)(-x,z,-y)(-x,-z,y)(x,-z,-y)(z,y,x)(z,-y,-x)(-z,y,-x)(-z,-y,x)',
                         'm-3m': '(x,y,z)(-x,-y,z)(-x,y,-z)(x,-y,-z)(z,x,y)(z,-x,-y)(-z,-x,y)(-z,x,-y)(y,z,x)(-y,z,-x)(y,-z,-x)(-y,-z,x)(y,x,-z)(-y,-x,-z)(y,-x,z)(-y,x,z)(x,z,-y)(-x,z,y)(-x,-z,-y)(x,-z,y)(z,y,-x)(z,-y,x)(-z,y,x)(-z,-y,-x)(-x,-y,-z)(x,y,-z)(x,-y,z)(-x,y,z)(-z,-x,-y)(-z,x,y)(z,x,-y)(z,-x,y)(-y,-z,-x)(y,-z,x)(-y,z,x)(y,z,-x)(-y,-x,z)(y,x,z)(-y,x,-z)(y,-x,-z)(-x,-z,y)(x,-z,-y)(x,z,y)(-x,z,-y)(-z,-y,x)(-z,y,-x)(z,-y,-x)(z,y,x)'
                         }
        #self.get_sym_ops()
        
        # init goniometer rotation
        self.gon_rot_ax3 = Quaternion.from_axis_rotation(self.par['gon_phi_axs'], np.deg2rad(self.par['gon_phi_ang'])) * \
                           Quaternion.from_axis_rotation(self.par['gon_chi_axs'], np.deg2rad(self.par['gon_chi_ang'])) * \
                           Quaternion.from_axis_rotation(self.par['gon_omg_axs'], np.deg2rad(self.par['gon_omg_ang']))
        self.gon_rot_ax2 = Quaternion.from_axis_rotation(self.par['gon_chi_axs'], np.deg2rad(self.par['gon_chi_ang'])) * \
                           Quaternion.from_axis_rotation(self.par['gon_omg_axs'], np.deg2rad(self.par['gon_omg_ang']))
        self.gon_rot_ax1 = Quaternion.from_axis_rotation(self.par['gon_omg_axs'], np.deg2rad(self.par['gon_omg_ang']))
        self.gon_rot_tth = Quaternion.from_axis_rotation(self.par['gon_tth_axs'], np.deg2rad(self.par['gon_tth_ang']))

        self.scan_timer = QtCore.QTimer()
        self.scan_timer.timeout.connect(self.scan_update)
        
        # run inits
        self.init_gui()
        self.init_gui_parameters()
        self.init_plot()

        # Set minimum width of parameter scroll area to fit all unfolded parameters
        parameter_scroll.setMinimumWidth(self.parameter_box.sizeHint().width() + 20)

        # Initial toggle states
        self.gui_toggle_hkls(self.show_text_hkls)

        # Collapse all widgets
        for widget in self.parameter_box.children():
            if isinstance(widget, QtWidgets.QPushButton):
                widget.click()

    def add_dock_widget(self, title: str):
        btn = QtWidgets.QPushButton(title)
        btn.setCheckable(True)
        btn.clicked.connect(lambda: box.setVisible(not btn.isChecked()))
        box = QtWidgets.QGroupBox()
        box_layout = QtWidgets.QFormLayout()
        box.setLayout(box_layout)
        self.params_layout.addWidget(btn)
        self.params_layout.addWidget(box)
        return box_layout

    def init_gui(self):
        # Plot Options
        self.box_gui_layout = self.add_dock_widget("Plot Options")
        # plot projection
        self.gui_plo_prj = QtWidgets.QCheckBox()
        self.gui_plo_prj.setChecked(self.orth_proj)
        self.gui_plo_prj.stateChanged.connect(self.gui_set_projection)
        self.box_gui_layout.addRow(QtWidgets.QLabel("Orthographic Projection"), self.gui_plo_prj)
        # Ewald sphere
        self.gui_show_ewald = QtWidgets.QCheckBox()
        self.gui_show_ewald.setChecked(self.show_ewald)
        self.gui_show_ewald.stateChanged.connect(self.gui_toggle_ewald)
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Ewald Sphere"), self.gui_show_ewald)
        # show hkls of scan data
        self.gui_show_hkls = QtWidgets.QCheckBox()
        self.gui_show_hkls.setChecked(self.show_text_hkls)
        self.gui_show_hkls.stateChanged.connect(self.gui_toggle_hkls)
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show hkl labels"), self.gui_show_hkls)
        # show reciprocal lattice points
        self.gui_show_latt = QtWidgets.QCheckBox()
        self.gui_show_latt.setChecked(self.show_scat_latt)
        self.gui_show_latt.stateChanged.connect(lambda state: self.gui_toggle_generic(state, 'show_scat_latt', self.scat_latt))
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Reciprocal Lattice"), self.gui_show_latt)
        # show collected data
        self.gui_show_data = QtWidgets.QCheckBox()
        self.gui_show_data.setChecked(self.show_scat_data)
        self.gui_show_data.stateChanged.connect(lambda state: self.gui_toggle_generic(state, 'show_scat_data', self.scat_data))
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Collected Data"), self.gui_show_data)
        # show symmetry equivalents
        self.gui_show_symm = QtWidgets.QCheckBox()
        self.gui_show_symm.setChecked(self.show_scat_symm)
        self.gui_show_symm.stateChanged.connect(lambda state: self.gui_toggle_generic(state, 'show_scat_symm', self.scat_symm))
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Symmetry Equivalents"), self.gui_show_symm)
        # show diffracted beams
        self.gui_show_scat_vecs = QtWidgets.QCheckBox()
        self.gui_show_scat_vecs.setChecked(self.show_scat_vecs)
        self.gui_show_scat_vecs.stateChanged.connect(lambda state: self.gui_toggle_generic(state, 'show_scat_vecs', self.plot_scat_vecs))
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Scattering Vectors"), self.gui_show_scat_vecs)
        # show scattering vectors
        self.gui_show_diff_vecs = QtWidgets.QCheckBox()
        self.gui_show_diff_vecs.setChecked(self.show_diff_vecs)
        self.gui_show_diff_vecs.stateChanged.connect(lambda state: self.gui_toggle_generic(state, 'show_diff_vecs', self.plot_diff_vecs))
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Diffracted Vectors"), self.gui_show_diff_vecs)
        # Unit cell outline
        self.gui_show_cell = QtWidgets.QCheckBox()
        self.gui_show_cell.setChecked(self.show_cell_outline)
        self.gui_show_cell.stateChanged.connect(lambda state: self.gui_toggle_generic(state, 'show_cell_outline', self.plot_cell))
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Unit Cell"), self.gui_show_cell)
        # Cell axes
        self.gui_show_axes = QtWidgets.QCheckBox()
        self.gui_show_axes.setTristate(True)
        self.gui_show_axes.setChecked(self.show_cell_axs)
        self.gui_show_axes.stateChanged.connect(self.gui_toggle_cell_axes)
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Unit Cell Axes"), self.gui_show_axes)
        # Goniometer axes
        self.gui_show_gon_axes = QtWidgets.QCheckBox()
        self.gui_show_gon_axes.setTristate(True)
        self.gui_show_gon_axes.setChecked(self.show_gon_axs)
        self.gui_show_gon_axes.stateChanged.connect(self.gui_toggle_goni_axes)
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Goniometer Axes"), self.gui_show_gon_axes)
        # Detector
        self.gui_show_detector = QtWidgets.QCheckBox()
        self.gui_show_detector.setChecked(self.show_detector)
        self.gui_show_detector.stateChanged.connect(self.gui_toggle_detector)
        self.box_gui_layout.addRow(QtWidgets.QLabel("Show Detector"), self.gui_show_detector)

        # Experimental Information
        self.box_sample_layout = self.add_dock_widget("Experimental")
        # Sample parameters
        self.gui_sample_a = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sample_a.setMinimum(1)
        self.gui_sample_a.setMaximum(100)
        self.gui_sample_a.setDecimals(3)
        self.gui_sample_a.setSingleStep(0.1)
        self.gui_sample_a.setValue(self.par['sample_cell_a'])
        self.gui_sample_a.setSuffix(' Å')
        self.gui_sample_a.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("a"), self.gui_sample_a)
        self.gui_sample_b = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sample_b.setMinimum(1)
        self.gui_sample_b.setMaximum(100)
        self.gui_sample_b.setDecimals(3)
        self.gui_sample_b.setSingleStep(0.1)
        self.gui_sample_b.setValue(self.par['sample_cell_b'])
        self.gui_sample_b.setSuffix(' Å')
        self.gui_sample_b.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("b"), self.gui_sample_b)
        self.gui_sample_c = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sample_c.setMinimum(1)
        self.gui_sample_c.setMaximum(100)
        self.gui_sample_c.setDecimals(3)
        self.gui_sample_c.setSingleStep(0.1)
        self.gui_sample_c.setValue(self.par['sample_cell_c'])
        self.gui_sample_c.setSuffix(' Å')
        self.gui_sample_c.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("c"), self.gui_sample_c)
        self.gui_sample_alpha = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sample_alpha.setMinimum(1)
        self.gui_sample_alpha.setMaximum(180)
        self.gui_sample_alpha.setDecimals(2)
        self.gui_sample_alpha.setSingleStep(0.1)
        self.gui_sample_alpha.setValue(self.par['sample_cell_alpha'])
        self.gui_sample_alpha.setSuffix('°')
        self.gui_sample_alpha.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("Alpha"), self.gui_sample_alpha)
        self.gui_sample_beta = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sample_beta.setMinimum(1)
        self.gui_sample_beta.setMaximum(180)
        self.gui_sample_beta.setDecimals(2)
        self.gui_sample_beta.setSingleStep(0.1)
        self.gui_sample_beta.setValue(self.par['sample_cell_beta'])
        self.gui_sample_beta.setSuffix('°')
        self.gui_sample_beta.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("Beta"), self.gui_sample_beta)
        self.gui_sample_gamma = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sample_gamma.setMinimum(1)
        self.gui_sample_gamma.setMaximum(180)
        self.gui_sample_gamma.setDecimals(2)
        self.gui_sample_gamma.setSingleStep(0.1)
        self.gui_sample_gamma.setValue(self.par['sample_cell_gamma'])
        self.gui_sample_gamma.setSuffix('°')
        self.gui_sample_gamma.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("Gamma"), self.gui_sample_gamma)
        self.gui_sample_point_group = QtWidgets.QComboBox()
        self.gui_sample_point_group.addItems(sorted(self.sym_dict.keys()))
        self.gui_sample_point_group.setCurrentText(self.par['sample_point_group'])
        self.gui_sample_point_group.currentTextChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("Point Group"), self.gui_sample_point_group)
        self.gui_sample_wavelength = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sample_wavelength.setMinimum(0.0001)
        self.gui_sample_wavelength.setMaximum(10)
        self.gui_sample_wavelength.setDecimals(5)
        self.gui_sample_wavelength.setSingleStep(0.00001)
        self.gui_sample_wavelength.setAccelerated(True)
        self.gui_sample_wavelength.setValue(self.par['wavelength']*1E10)
        self.gui_sample_wavelength.setSuffix(' Å')
        self.gui_sample_wavelength.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("Wavelength"), self.gui_sample_wavelength)
        self.gui_prd_res = pg.QtWidgets.QDoubleSpinBox()
        self.gui_prd_res.setMinimum(0.00005)
        self.gui_prd_res.setMaximum(1)
        self.gui_prd_res.setDecimals(5)
        self.gui_prd_res.setSingleStep(0.00001)
        self.gui_prd_res.setValue(self.par['max_resolution']*1E10)
        self.gui_prd_res.setSuffix(' Å')
        self.gui_prd_res.valueChanged.connect(self.restart_check_enable)
        self.box_sample_layout.addRow(QtWidgets.QLabel("Resolution"), self.gui_prd_res)
        self.gui_sample_apply = QtWidgets.QPushButton('Restart')
        self.gui_sample_apply.setStyleSheet(f"background-color: {self.color_gui_buttons.name()}")
        self.gui_sample_apply.clicked.connect(self.restart_new_cell)
        self.box_sample_layout.addRow(self.gui_sample_apply)

        # Detector parameters
        self.box_det_layout = self.add_dock_widget("Detector")
        self.gui_det_distance = pg.QtWidgets.QDoubleSpinBox()
        self.gui_det_distance.setMinimum(0)
        self.gui_det_distance.setMaximum(10000000)
        self.gui_det_distance.setDecimals(2)
        self.gui_det_distance.setSingleStep(1)
        self.gui_det_distance.setValue(self.par['det_distance'] * 1E3)
        self.gui_det_distance.setSuffix(' mm')
        self.gui_det_distance.valueChanged.connect(self.gui_update_detector)
        self.box_det_layout.addRow(QtWidgets.QLabel("Distance"), self.gui_det_distance)
        self.gui_det_dim_1 = pg.QtWidgets.QSpinBox()
        self.gui_det_dim_1.setMinimum(1)
        self.gui_det_dim_1.setMaximum(10000)
        self.gui_det_dim_1.setValue(self.par['det_pix_x'])
        self.gui_det_dim_1.setSuffix(' px')
        self.gui_det_dim_1.valueChanged.connect(self.gui_update_detector)
        self.box_det_layout.addRow(QtWidgets.QLabel("Size x"), self.gui_det_dim_1)
        self.gui_det_dim_2 = pg.QtWidgets.QSpinBox()
        self.gui_det_dim_2.setMinimum(1)
        self.gui_det_dim_2.setMaximum(10000)
        self.gui_det_dim_2.setValue(self.par['det_pix_y'])
        self.gui_det_dim_2.setSuffix(' px')
        self.gui_det_dim_2.valueChanged.connect(self.gui_update_detector)
        self.box_det_layout.addRow(QtWidgets.QLabel("Size y"), self.gui_det_dim_2)
        self.gui_det_pix_size = pg.QtWidgets.QDoubleSpinBox()
        self.gui_det_pix_size.setMinimum(0.01)
        self.gui_det_pix_size.setMaximum(10000)
        self.gui_det_pix_size.setDecimals(2)
        self.gui_det_pix_size.setSingleStep(1)
        self.gui_det_pix_size.setValue(self.par['det_pix_s']*1E6)
        self.gui_det_pix_size.setSuffix(' µm')
        self.gui_det_pix_size.valueChanged.connect(self.gui_update_detector)
        self.box_det_layout.addRow(QtWidgets.QLabel("Pixel Size"), self.gui_det_pix_size)

        # Goniometer
        self.box_gon_layout = self.add_dock_widget("Goniometer")
        self.gui_gon_omg = pg.QtWidgets.QDoubleSpinBox()
        self.gui_gon_omg.setMinimum(-999999999)
        self.gui_gon_omg.setMaximum(999999999)
        self.gui_gon_omg.setDecimals(2)
        self.gui_gon_omg.setSingleStep(1)
        self.gui_gon_omg.setValue(self.par['gon_omg_ang'])
        self.gui_gon_omg.setSuffix('°')
        self.gui_gon_omg.valueChanged.connect(self.rotate_gon)
        self.box_gon_layout.addRow(QtWidgets.QLabel("Omega"), self.gui_gon_omg)
        self.gui_gon_chi = pg.QtWidgets.QDoubleSpinBox()
        self.gui_gon_chi.setMinimum(-999999999)
        self.gui_gon_chi.setMaximum(999999999)
        self.gui_gon_chi.setDecimals(2)
        self.gui_gon_chi.setSingleStep(1)
        self.gui_gon_chi.setValue(self.par['gon_chi_ang'])
        self.gui_gon_chi.setSuffix('°')
        self.gui_gon_chi.valueChanged.connect(self.rotate_gon)
        self.box_gon_layout.addRow(QtWidgets.QLabel("Chi"), self.gui_gon_chi)
        self.gui_gon_phi = pg.QtWidgets.QDoubleSpinBox()
        self.gui_gon_phi.setMinimum(-999999999)
        self.gui_gon_phi.setMaximum(999999999)
        self.gui_gon_phi.setDecimals(2)
        self.gui_gon_phi.setSingleStep(1)
        self.gui_gon_phi.setValue(self.par['gon_phi_ang'])
        self.gui_gon_phi.setSuffix('°')
        self.gui_gon_phi.valueChanged.connect(self.rotate_gon)
        self.box_gon_layout.addRow(QtWidgets.QLabel("Phi"), self.gui_gon_phi)
        self.gui_gon_tth = pg.QtWidgets.QDoubleSpinBox()
        self.gui_gon_tth.setMinimum(-360)
        self.gui_gon_tth.setMaximum(360)
        self.gui_gon_tth.setDecimals(2)
        self.gui_gon_tth.setSingleStep(1)
        self.gui_gon_tth.setValue(self.par['gon_tth_ang'])
        self.gui_gon_tth.setSuffix('°')
        self.gui_gon_tth.valueChanged.connect(self.rotate_gon_tth)
        self.box_gon_layout.addRow(QtWidgets.QLabel("2-Theta"), self.gui_gon_tth)
        self.gui_gon_tth_orientation = QtWidgets.QComboBox()
        self.gui_gon_tth_orientation.addItems(['Vertical', 'Horizontal'])
        self.box_gon_layout.addRow(QtWidgets.QLabel("2-Theta Orientation"), self.gui_gon_tth_orientation)

        # Scan parameters
        self.box_scan_layout = self.add_dock_widget("Scan Parameters")
        self.gui_scan_axis = QtWidgets.QComboBox()
        self.gui_scan_axis.addItems(['Omega', 'Phi', 'SFX'])
        self.box_scan_layout.addRow(QtWidgets.QLabel("Scan Axis"), self.gui_scan_axis)
        self.gui_scan_speed = pg.QtWidgets.QSpinBox()
        self.gui_scan_speed.setMinimum(1)
        self.gui_scan_speed.setMaximum(1000)
        self.gui_scan_speed.setValue(self.par['scan_speed'])
        self.gui_scan_speed.setSuffix(' ms/step')
        self.gui_scan_speed.valueChanged.connect(self.gui_set_scan_speed)
        self.box_scan_layout.addRow(QtWidgets.QLabel("Scan Speed"), self.gui_scan_speed)
        self.gui_scan_range = pg.QtWidgets.QSpinBox()
        self.gui_scan_range.setMinimum(1)
        self.gui_scan_range.setMaximum(999999999)
        self.gui_scan_range.setValue(self.par['scan_range'])
        self.gui_scan_range.setSuffix('°')
        self.box_scan_layout.addRow(QtWidgets.QLabel("Scan Range"), self.gui_scan_range)
        self.gui_scan_step = pg.QtWidgets.QDoubleSpinBox()
        self.gui_scan_step.setMinimum(0.001)
        self.gui_scan_step.setMaximum(1)
        self.gui_scan_step.setDecimals(3)
        self.gui_scan_step.setSingleStep(0.1)
        self.gui_scan_step.setValue(self.par['scan_step'])
        self.gui_scan_step.setSuffix('°')
        self.box_scan_layout.addRow(QtWidgets.QLabel("Scan Step"), self.gui_scan_step)
        self.gui_scan_ewald_offset = pg.QtWidgets.QDoubleSpinBox()
        self.gui_scan_ewald_offset.setMinimum(0.0001)
        self.gui_scan_ewald_offset.setMaximum(0.5)
        self.gui_scan_ewald_offset.setDecimals(4)
        self.gui_scan_ewald_offset.setSingleStep(0.0001)
        self.gui_scan_ewald_offset.setAccelerated(True)
        self.gui_scan_ewald_offset.setValue(self.par['ewald_offset'])
        self.gui_scan_ewald_offset.setSuffix(' Å')
        self.gui_scan_ewald_offset.valueChanged.connect(lambda: self.gui_set_par_generic('ewald_offset', self.gui_scan_ewald_offset.value()))
        self.box_scan_layout.addRow(QtWidgets.QLabel("Ewald Offset"), self.gui_scan_ewald_offset)
        self.gui_scan_total = QtWidgets.QLabel()
        self.gui_scan_total.setEnabled(False)
        self.box_scan_layout.addRow(QtWidgets.QLabel("Reflections"), self.gui_scan_total)
        self.gui_scan_collected = QtWidgets.QLabel()
        self.gui_scan_collected.setEnabled(False)
        self.box_scan_layout.addRow(QtWidgets.QLabel("Collected"), self.gui_scan_collected)
        self.gui_scan_completeness = QtWidgets.QLabel()
        self.gui_scan_completeness.setEnabled(False)
        self.box_scan_layout.addRow(QtWidgets.QLabel("Completeness"), self.gui_scan_completeness)
        self.gui_start_scan = QtWidgets.QPushButton('Start Scan')
        self.gui_start_scan.setStyleSheet(f"background-color: {self.color_gui_buttons.name()}")
        self.gui_start_scan.clicked.connect(self.scan_toggle)
        self.box_scan_layout.addRow(self.gui_start_scan)

        # SFX parameters
        self.box_sfx_layout = self.add_dock_widget("SFX Parameters")
        self.gui_show_sfx_parcor = QtWidgets.QCheckBox()
        self.gui_show_sfx_parcor.setChecked(self.show_sfx_parcor)
        self.box_sfx_layout.addRow(QtWidgets.QLabel("Show ParCor"), self.gui_show_sfx_parcor)
        self.gui_sfx_keep_all_data = QtWidgets.QCheckBox()
        self.gui_sfx_keep_all_data.setChecked(self.sfx_keep_all_data)
        self.box_sfx_layout.addRow(QtWidgets.QLabel("Keep All Data"), self.gui_sfx_keep_all_data)
        self.gui_sfx_spectral_width = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sfx_spectral_width.setMinimum(1E-10)
        self.gui_sfx_spectral_width.setMaximum(1E10)
        self.gui_sfx_spectral_width.setStepType(pg.QtWidgets.QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self.gui_sfx_spectral_width.setSingleStep(0.001)
        self.gui_sfx_spectral_width.setDecimals(4)
        self.gui_sfx_spectral_width.setValue(self.par['sfx_spectral_width'])
        self.gui_sfx_spectral_width.setSuffix('')
        self.gui_sfx_spectral_width.valueChanged.connect(lambda: self.gui_set_par_generic('sfx_spectral_width', self.gui_sfx_spectral_width.value()))
        self.box_sfx_layout.addRow(QtWidgets.QLabel("Spectral width"), self.gui_sfx_spectral_width)
        self.gui_sfx_rlp_size = pg.QtWidgets.QDoubleSpinBox()
        self.gui_sfx_rlp_size.setMinimum(1E-10)
        self.gui_sfx_rlp_size.setMaximum(1E10)
        self.gui_sfx_rlp_size.setStepType(pg.QtWidgets.QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self.gui_sfx_rlp_size.setSingleStep(0.001)
        self.gui_sfx_rlp_size.setDecimals(4)
        self.gui_sfx_rlp_size.setValue(self.par['sfx_rlp_size'])
        self.gui_sfx_rlp_size.setSuffix('')
        self.gui_sfx_rlp_size.valueChanged.connect(lambda: self.gui_set_par_generic('sfx_rlp_size', self.gui_sfx_rlp_size.value()))
        self.box_sfx_layout.addRow(QtWidgets.QLabel("RLP Size"), self.gui_sfx_rlp_size)
        
        # Data collection
        self.box_dat_layout = self.add_dock_widget("Data Collection")
        self.gui_dat_tab = QtWidgets.QTableWidget()
        self.gui_dat_tab.setColumnCount(5)
        self.gui_dat_tab.setHorizontalHeaderLabels(['h', 'k', 'l', 'd', 'pc'])
        # Right align table contents
        class AlignDelegate(QtWidgets.QStyledItemDelegate):
            def initStyleOption(self, option, index):
                super(AlignDelegate, self).initStyleOption(option, index)
                #option.displayAlignment = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                option.displayAlignment = QtCore.Qt.AlignmentFlag.AlignCenter
                #option.font.setFamily('Courier New')
        self.gui_dat_tab.setItemDelegate(AlignDelegate())
        self.gui_dat_tab.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.gui_dat_tab.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.gui_dat_tab.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.gui_dat_tab.setAlternatingRowColors(True)
        self.gui_dat_tab.setSortingEnabled(True)
        self.gui_dat_tab.setFont(self.plot_fnt_tab)
        self.gui_dat_tab.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        #self.gui_dat_tab.horizontalHeader().setVisible(False)
        #self.gui_dat_tab.verticalHeader().setVisible(False)
        self.gui_dat_tab.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.gui_dat_tab.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Set hkl columns to resize to contents
        self.gui_dat_tab.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.gui_dat_tab.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.gui_dat_tab.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.box_dat_layout.addRow(self.gui_dat_tab)

    def init_gui_parameters(self):
        # Update parameters from GUI
        self.par['sample_cell_a'] = self.gui_sample_a.value()
        self.par['sample_cell_b'] = self.gui_sample_b.value()
        self.par['sample_cell_c'] = self.gui_sample_c.value()
        self.par['sample_cell_alpha'] = self.gui_sample_alpha.value()
        self.par['sample_cell_beta'] = self.gui_sample_beta.value()
        self.par['sample_cell_gamma'] = self.gui_sample_gamma.value()
        self.par['wavelength'] = self.gui_sample_wavelength.value() * 1E-10
        self.par['max_resolution'] = self.gui_prd_res.value() * 1E-10
        self.par['sample_point_group'] = self.gui_sample_point_group.currentText()

        # Calculate ewald radius
        self.ewald_rad = 1/(self.par['wavelength']*1E10) # in [A^-1]
        # Calculate incident beam vector
        self.par['exp_incident_beam_vector'] = np.array([0, 0, self.ewald_rad])
        # Calculate reciprocal space
        self.UC = matrix_from_cell([self.par['sample_cell_a'],
                                    self.par['sample_cell_b'],
                                    self.par['sample_cell_c'],
                                    self.par['sample_cell_alpha'],
                                    self.par['sample_cell_beta'],
                                    self.par['sample_cell_gamma']])
        # Apply orientation matrix
        self.OM_UC = self.OM.to_rotation_matrix() @ self.UC
        # get symmetry operations for new point group
        self.get_sym_ops()
        
        self.hkls = calc_hkls(self.OM_UC, self.par['max_resolution'])
        #self.hkls_set = set(map(tuple, self.hkls))
        self.hkls_num = len(self.hkls)
        
        self.hkl_bases = self.hkls @ self.OM_UC.T * 1E-10
        self.last_label_idx = 0
        # Reset scan progress
        self.scan_data = pd.DataFrame(columns=['h', 'k', 'l', 'bx', 'by', 'bz', 'dx', 'dy', 'dz'])
        self.scan_symm = set()
        # Update GUI
        self.gui_scan_total.setText(f'{self.hkls_num}')
        self.gui_scan_completeness.setText(f'{len(self.scan_data)/self.hkls_num*100:.0f} % ({(len(self.scan_symm)+len(self.scan_data))/self.hkls_num*100:.0f} %)')
        self.gui_scan_collected.setText(f'{len(self.scan_data)} ({len(self.scan_symm)+len(self.scan_data)})')
        self.scan_progress = 0
        # Reset goniometer rotation (new hkl are calculated in this orientation)
        self.gui_gon_phi.setValue(0.0)
        self.gui_gon_chi.setValue(0.0)
        self.gui_gon_omg.setValue(0.0)
        self.gui_gon_tth.setValue(0.0)
        self.par['gon_phi_ang'] = 0.0
        self.par['gon_chi_ang'] = 0.0
        self.par['gon_omg_ang'] = 0.0
        self.par['gon_tth_ang'] = 0.0

    def init_plot(self):
        #################
        #    Detector   #
        #################
        self.plot_add_detector()

        # Scattering pattern on detector
        self.scat_pattern = gl.GLScatterPlotItem(size=max(self.plot_det_dist/50, 0.25),
                                                 color=self.color_scat_patt,
                                                 pxMode=False)
        self.gl3d.addItem(self.scat_pattern)

        #################
        # Axes / Labels #
        #################
        cell_axs = self.OM_UC * 1E-10
        axs_a = cell_axs[:,0]
        axs_b = cell_axs[:,1]
        axs_c = cell_axs[:,2]
        # create the axis lines
        self.plot_ax_a = gl.GLLinePlotItem()
        self.plot_ax_a.setData(pos=np.array([[0,0,0], axs_a]),
                                color=QtGui.QColor("#FF0000"), width=self.plot_line_width_thick)
        self.gl3d.addItem(self.plot_ax_a)
        self.plot_ax_b = gl.GLLinePlotItem()
        self.plot_ax_b.setData(pos=np.array([[0,0,0], axs_b]),
                                color=QtGui.QColor("#11FF00"), width=self.plot_line_width_thick)
        self.gl3d.addItem(self.plot_ax_b)
        self.plot_ax_c = gl.GLLinePlotItem()
        self.plot_ax_c.setData(pos=np.array([[0,0,0], axs_c]),
                                color=QtGui.QColor("#0015FF"), width=self.plot_line_width_thick)
        self.gl3d.addItem(self.plot_ax_c)
        # Axis labels
        self.plot_ax_a_lbl = gl.GLTextItem(text='a', color=QtGui.QColor("#FF0000"), font=self.plot_fnt_lbl, parentItem=self.plot_ax_a)
        self.plot_ax_a_lbl.translate(*axs_a/2)
        self.gl3d.addItem(self.plot_ax_a_lbl)
        self.plot_ax_b_lbl = gl.GLTextItem(text='b', color=QtGui.QColor("#11FF00"), font=self.plot_fnt_lbl, parentItem=self.plot_ax_b)
        self.plot_ax_b_lbl.translate(*axs_b/2)
        self.gl3d.addItem(self.plot_ax_b_lbl)
        self.plot_ax_c_lbl = gl.GLTextItem(text='c', color=QtGui.QColor("#0015FF"), font=self.plot_fnt_lbl, parentItem=self.plot_ax_c)
        self.plot_ax_c_lbl.translate(*axs_c/2)
        self.gl3d.addItem(self.plot_ax_c_lbl)
        # create the unit cell
        verts = np.array([[ 0, 0, 0],
                            axs_a, axs_b, axs_c,
                            axs_a + axs_b,
                            axs_a + axs_c,
                            axs_b + axs_c,
                            axs_a + axs_b + axs_c], dtype=float)
        edges = np.array([[0,1],[0,2],[0,3],
                            [1,4],[2,4],
                            [1,5],[3,5],
                            [2,6],[3,6],
                            [7,4],[7,5],[7,6]], dtype=int)
        self.plot_cell = gl.GLGraphItem()
        self.plot_cell.setData(edges=edges, nodePositions=verts,
                                edgeColor=QtGui.QColor("#FFFFFFAA"), edgeWidth=1)
        self.gl3d.addItem(self.plot_cell)
        # create the goniometer axis lines
        self.plot_gon_phi = gl.GLLinePlotItem()
        self.plot_gon_phi.setData(pos=np.array([[0,0,0], self.par['gon_phi_axs']]),
                                    color=QtGui.QColor("#FF00FF"), width=self.plot_line_width_thick)
        self.gl3d.addItem(self.plot_gon_phi)
        self.plot_gon_chi = gl.GLLinePlotItem()
        self.plot_gon_chi.setData(pos=np.array([[0,0,0], self.par['gon_chi_axs']]),
                                    color=QtGui.QColor("#00FFFF"), width=self.plot_line_width_thick)
        self.gl3d.addItem(self.plot_gon_chi)
        self.plot_gon_omg = gl.GLLinePlotItem()
        self.plot_gon_omg.setData(pos=np.array([[0,0,0], self.par['gon_omg_axs']]),
                                    color=QtGui.QColor("#FFFF00"), width=self.plot_line_width_thick)
        self.gl3d.addItem(self.plot_gon_omg)
        # Goniometer axis labels
        self.plot_gon_phi_lbl = gl.GLTextItem(text='\u03D5', color=QtGui.QColor("#FF00FF"), font=self.plot_fnt_lbl, parentItem=self.plot_gon_phi)
        self.plot_gon_phi_lbl.translate(*self.par['gon_phi_axs'])
        self.gl3d.addItem(self.plot_gon_phi_lbl)
        self.plot_gon_chi_lbl = gl.GLTextItem(text='\u03A7', color=QtGui.QColor("#00FFFF"), font=self.plot_fnt_lbl, parentItem=self.plot_gon_chi)
        self.plot_gon_chi_lbl.translate(*self.par['gon_chi_axs'])
        self.gl3d.addItem(self.plot_gon_chi_lbl)
        self.plot_gon_omg_lbl = gl.GLTextItem(text='\u03A9', color=QtGui.QColor("#FFFF00"), font=self.plot_fnt_lbl, parentItem=self.plot_gon_omg)
        self.plot_gon_omg_lbl.translate(*self.par['gon_omg_axs'])
        self.gl3d.addItem(self.plot_gon_omg_lbl)

        #################
        # scatter plots #
        #################
        # Reciprocal lattice points
        self.scat_latt = gl.GLScatterPlotItem(pos=self.hkl_bases,
                                              size=0.05,
                                              color=self.color_scat_latt,
                                              pxMode=False)
        self.gl3d.addItem(self.scat_latt)
        
        # Symmetry equivalents
        self.scat_symm = gl.GLScatterPlotItem(size=0.075,
                                              color=self.color_scat_symm,
                                              pxMode=False)
        self.gl3d.addItem(self.scat_symm)
        
        # Collected data points
        self.scat_data = gl.GLScatterPlotItem(size=0.075,
                                              color=self.color_scat_data,
                                              pxMode=False)
        self.gl3d.addItem(self.scat_data)

        # Scanned scan point
        self.scat_scan = gl.GLScatterPlotItem(size=0.075,
                                              color=self.color_scat_scan,
                                              pxMode=False)
        self.gl3d.addItem(self.scat_scan)

        ################
        # Ewald sphere #
        ################
        """
        # Inner Ewald sphere (for SFX spectral width)
        self.ewald_sphere_inner = gl.GLMeshItem()
        self.ewald_sphere_inner.setMeshData(drawEdges=False, 
                                            meshdata=gl.MeshData.sphere(rows=100,
                                                                        cols=100,
                                                                        radius=self.ewald_rad+self.par['sfx_spectral_width']/2))
        self.ewald_sphere_inner.translate(0, 0, -self.ewald_rad-self.par['sfx_spectral_width']/2)
        self.ewald_sphere_inner.setGLOptions('additive')
        self.ewald_sphere_inner.setColor((0.9, 0.0, 0.0, 0.1))
        self.gl3d.addItem(self.ewald_sphere_inner)
        
        # Outer Ewald sphere
        self.ewald_sphere_outer = gl.GLMeshItem()
        self.ewald_sphere_outer.setMeshData(drawEdges=False, 
                                            meshdata=gl.MeshData.sphere(rows=100,
                                                                        cols=100,
                                                                        radius=self.ewald_rad-self.par['sfx_spectral_width']/2))
        self.ewald_sphere_outer.translate(0, 0, -self.ewald_rad+self.par['sfx_spectral_width']/2)
        self.ewald_sphere_outer.setGLOptions('additive')
        self.ewald_sphere_outer.setColor((0.0, 0.0, 0.9, 0.1))
        self.gl3d.addItem(self.ewald_sphere_outer)
        """

        self.ewald_sphere = gl.GLMeshItem()
        self.ewald_sphere.setMeshData(drawEdges=False, 
                                      meshdata=gl.MeshData.sphere(rows=100,
                                                                  cols=100,
                                                                  radius=self.ewald_rad))
        self.ewald_sphere.translate(0, 0, -self.ewald_rad)
        self.ewald_sphere.setGLOptions('additive')
        self.ewald_sphere.setColor((0.3, 0.3, 0.3, 0.2))
        self.gl3d.addItem(self.ewald_sphere)
        # Ewald sphere outline ab
        self.ewald_outline_ab = gl.GLLinePlotItem()
        theta = np.linspace(0, 2*np.pi, 100)
        x = self.ewald_rad * np.cos(theta)
        y = self.ewald_rad * np.sin(theta)
        z = np.zeros_like(x) - self.ewald_rad
        self.ewald_outline_ab.setData(pos=np.vstack([x,y,z]).T,
                                      color=(0.5,0.5,0.5,0.5),
                                      width=self.plot_line_width_thick)
        self.gl3d.addItem(self.ewald_outline_ab)
        # Ewald sphere outline bc
        self.ewald_outline_bc = gl.GLLinePlotItem()
        self.ewald_outline_bc.setData(pos=np.vstack([x,y,z]).T,
                                      color=(0.5,0.5,0.5,0.5),
                                      width=self.plot_line_width_thick)
        self.ewald_outline_bc.rotate(90, 0, 1, 0)
        self.ewald_outline_bc.translate(self.ewald_rad, 0, -self.ewald_rad)
        self.gl3d.addItem(self.ewald_outline_bc)
        # Ewald sphere outline ac
        self.ewald_outline_ac = gl.GLLinePlotItem()
        self.ewald_outline_ac.setData(pos=np.vstack([x,y,z]).T,
                                      color=(0.5,0.5,0.5,0.5),
                                      width=self.plot_line_width_thick)
        self.ewald_outline_ac.rotate(90, 1, 0, 0)
        self.ewald_outline_ac.translate(0, -self.ewald_rad, -self.ewald_rad)
        self.gl3d.addItem(self.ewald_outline_ac)

        #################
        #    Vectors    #
        #################
        # Incident beam vector
        self.ewald_ki = gl.GLLinePlotItem()
        self.ewald_ki.setData(pos=np.array([[0,0,-self.ewald_rad*2],[0,0,-self.ewald_rad]]),
                        color="#AB1E6E", width=self.plot_line_width)
        self.gl3d.addItem(self.ewald_ki)
        # diffracted beam vector
        self.ewald_ko = gl.GLLinePlotItem()
        self.ewald_ko.setData(pos=np.array([[0,0,-self.ewald_rad],[0,0,0]]),
                        color="#1EAB60", width=self.plot_line_width)
        self.gl3d.addItem(self.ewald_ko)
        # Scattering vectors
        self.plot_scat_vecs = gl.GLGraphItem(edgeColor=self.color_scat_vecs,
                                             nodeColor=self.color_scat_vecs,
                                             edgeWidth=self.plot_line_width)
        self.gl3d.addItem(self.plot_scat_vecs)
        # Diffracted beam vectors
        self.plot_diff_vecs = gl.GLGraphItem(edgeColor=self.color_diff_vecs,
                                             nodeColor=self.color_diff_vecs,
                                             edgeWidth=self.plot_line_width)
        self.gl3d.addItem(self.plot_diff_vecs)

        #################
        #   hkl Labels  #
        #################
        # hkl text labels
        self.text_hkls_list = []
        for _ in range(self.par['plot_max_hkl_labels']):
            _text_hkls = gl.GLTextItem(font=self.plot_fnt_hkl,
                                       color=self.color_labl_hkls)
            self.gl3d.addItem(_text_hkls)
            self.text_hkls_list.append(_text_hkls)

    def plot_add_detector(self):
        if hasattr(self, 'plot_detector') and self.plot_detector in self.gl3d.items:
            self.gl3d.removeItem(self.plot_detector)
        if hasattr(self, 'detector_plane') and self.detector_plane in self.gl3d.items:
            self.gl3d.removeItem(self.detector_plane)
        if hasattr(self, 'detector_frame') and self.detector_frame in self.gl3d.items:
            self.gl3d.removeItem(self.detector_frame)

        _scale = 1# / (self.plot_det_dist / 2) + 1
        det_w = self.par['det_pix_x'] * self.par['det_pix_s'] / self.par['det_distance'] * self.plot_det_dist / 2 * _scale
        det_h = self.par['det_pix_y'] * self.par['det_pix_s'] / self.par['det_distance'] * self.plot_det_dist / 2 * _scale
        det_bch = det_w - self.par['det_poni_x'] / self.par['det_distance'] * self.plot_det_dist * _scale
        det_bcv = det_h - self.par['det_poni_y'] / self.par['det_distance'] * self.plot_det_dist * _scale

        # Create a thin 3D mesh frame (rectangular ring) around the detector face so the edges are clearly visible
        # Extrude the ring by a small depth in z to give it volume
        frame_frac = 0.05
        frame_thickness = max(min(det_w, det_h) * frame_frac, 1e-9)
        inner_half_w = max(det_w + frame_thickness, 1e-9)
        inner_half_h = max(det_h + frame_thickness, 1e-9)
        half_depth = frame_thickness * 0.5

        verts_list = []
        faces_list = []

        def add_box(x1, y1, x2, y2, z0, z1):
            base_idx = len(verts_list)
            # lower z0 (4 verts)
            verts_list.extend([[x1, y1, z0],
                               [x2, y1, z0],
                               [x2, y2, z0],
                               [x1, y2, z0]])
            # upper z1 (4 verts)
            verts_list.extend([[x1, y1, z1],
                               [x2, y1, z1],
                               [x2, y2, z1],
                               [x1, y2, z1]])
            # lower face
            faces_list.append([base_idx,
                               base_idx+1,
                               base_idx+2])
            faces_list.append([base_idx,
                               base_idx+2,
                               base_idx+3])
            # upper face (note winding)
            faces_list.append([base_idx+4,
                               base_idx+6,
                               base_idx+5])
            faces_list.append([base_idx+4,
                               base_idx+7,
                               base_idx+6])
            # side faces (4 sides)
            sides = [(0,1,5,4),
                     (1,2,6,5),
                     (2,3,7,6),
                     (3,0,4,7)]
            for a,b,c,d in sides:
                faces_list.append([base_idx+a,
                                   base_idx+b,
                                   base_idx+c])
                faces_list.append([base_idx+a,
                                   base_idx+c,
                                   base_idx+d])

        # Left box (strip)
        add_box(-det_w, -det_h, -inner_half_w, det_h, -half_depth, half_depth)
        # Right box
        add_box(inner_half_w, -det_h, det_w, det_h, -half_depth, half_depth)
        # Bottom box
        add_box(-inner_half_w, -det_h, inner_half_w, -inner_half_h, -half_depth, half_depth)
        # Top box
        add_box(-inner_half_w, inner_half_h, inner_half_w, det_h, -half_depth, half_depth)

        verts_arr = np.array(verts_list, dtype=np.float32)
        faces_arr = np.array(faces_list, dtype=np.int32)
        face_colors = np.tile(np.array((0.25, 0.25, 0.25, 1.0), dtype=np.float32), (len(faces_arr), 1))
        meshdata_frame = gl.MeshData(vertexes=verts_arr, faces=faces_arr, faceColors=face_colors, vertexColors=None)
        self.detector_frame = gl.GLMeshItem(meshdata=meshdata_frame, smooth=False, drawEdges=False, drawFaces=True)
        self.detector_frame.setGLOptions('translucent')
        self.detector_frame.setDepthValue(-1)
        self.detector_frame.translate(det_bch, det_bcv, self.plot_det_dist)
        self.gl3d.addItem(self.detector_frame)

        # Detector face
        self.plot_detector = gl.GLSurfacePlotItem()
        self.plot_detector.setGLOptions('additive')
        self.plot_detector.setData(x=np.array([-det_w, det_w]),
                                   y=np.array([-det_h, det_h]),
                                   z=np.zeros((2, 2)))
        self.plot_detector.setColor(self.color_detector)
        self.plot_detector.setDepthValue(-1)
        self.plot_detector.translate(det_bch,
                                     det_bcv,
                                     self.plot_det_dist)
        self.gl3d.addItem(self.plot_detector)

        # Create a plane to represent the detector
        verts = np.array([[-det_w, -det_h, 0],
                          [ det_w, -det_h, 0],
                          [ det_w,  det_h, 0],
                          [-det_w,  det_h, 0],
                          [     0,      0, 0]], dtype=np.float32)
        faces = np.array([[1,2,4],
                          [0,1,4],
                          [2,3,4],
                          [3,0,4]], dtype=np.int8)
        colors = np.ones((len(faces), 4), dtype=np.float32) * np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        meshdata = gl.MeshData(vertexes=verts, faces=faces, faceColors=colors)
        self.detector_plane = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, edgeColor=(1,1,1,1), drawFaces=True)
        self.detector_plane.setGLOptions('additive')
        self.detector_plane.setDepthValue(-1)
        self.detector_plane.translate(det_bch, det_bcv, self.plot_det_dist)
        self.gl3d.addItem(self.detector_plane)

    def gui_update_detector(self):
        # Update detector parameters from GUI
        self.par['det_distance'] = self.gui_det_distance.value() * 1E-3
        self.par['det_pix_x'] = self.gui_det_dim_1.value()
        self.par['det_pix_y'] = self.gui_det_dim_2.value()
        self.par['det_pix_s'] = self.gui_det_pix_size.value() * 1E-6

        # Reset poni to detector center
        # This does not work with custom poni positions -> future problem!
        self.par['det_poni_x'] = self.par['det_pix_x'] * self.par['det_pix_s'] / 2
        self.par['det_poni_y'] = self.par['det_pix_y'] * self.par['det_pix_s'] / 2

        self.plot_add_detector()
        self.rotate_gon_tth(relative=False)

    def gui_toggle_generic(self, state, attribute, plot_item):
        setattr(self, attribute, state)
        plot_item.setVisible(state)

    def gui_toggle_hkls(self, state):
        self.show_text_hkls = state
        for txt in self.text_hkls_list:
            txt.setVisible(self.show_text_hkls)
    
    def gui_toggle_ewald(self, state):
        self.show_ewald = state
        self.ewald_sphere.setVisible(self.show_ewald)
        self.ewald_outline_ab.setVisible(self.show_ewald)
        self.ewald_outline_ac.setVisible(self.show_ewald)
        self.ewald_outline_bc.setVisible(self.show_ewald)
        self.ewald_ki.setVisible(self.show_ewald)
        self.ewald_ko.setVisible(self.show_ewald)

    def gui_toggle_detector(self, state):
        self.show_detector = state
        self.plot_detector.setVisible(self.show_detector)
        self.scat_pattern.setVisible(self.show_detector)
        self.detector_plane.setVisible(self.show_detector)
        self.detector_frame.setVisible(self.show_detector)

    def gui_toggle_cell_axes(self, state):
        if state == 0:
            self.show_cell_axs = False
            self.show_cell_axs_lbl = False
        elif state == 1:
            self.show_cell_axs = True
            self.show_cell_axs_lbl = False
        else:
            self.show_cell_axs = True
            self.show_cell_axs_lbl = True
        self.plot_ax_a.setVisible(self.show_cell_axs)
        self.plot_ax_b.setVisible(self.show_cell_axs)
        self.plot_ax_c.setVisible(self.show_cell_axs)
        self.plot_ax_a_lbl.setVisible(self.show_cell_axs_lbl)
        self.plot_ax_b_lbl.setVisible(self.show_cell_axs_lbl)
        self.plot_ax_c_lbl.setVisible(self.show_cell_axs_lbl)
    
    def gui_toggle_goni_axes(self, state):
        if state == 0:
            self.show_gon_axs = False
            self.show_gon_axs_lbl = False
        elif state == 1:
            self.show_gon_axs = True
            self.show_gon_axs_lbl = False
        else:
            self.show_gon_axs = True
            self.show_gon_axs_lbl = True
        self.plot_gon_phi.setVisible(self.show_gon_axs)
        self.plot_gon_chi.setVisible(self.show_gon_axs)
        self.plot_gon_omg.setVisible(self.show_gon_axs)
        self.plot_gon_phi_lbl.setVisible(self.show_gon_axs_lbl)
        self.plot_gon_chi_lbl.setVisible(self.show_gon_axs_lbl)
        self.plot_gon_omg_lbl.setVisible(self.show_gon_axs_lbl)

    def gui_set_projection(self, toggle):
        self.orth_proj = toggle
        if self.orth_proj:
            #params = {'distance': 1000, 'fov': 1, 'elevation': 0, 'azimuth': 0, 'center': pg.Vector(0,0,0)}
            distance = self.gl3d.cameraParams()['distance'] * 6800
            params = {'distance': distance, 'fov': 0.01}
        else:
            #params = {'distance': 25, 'fov': 60, 'elevation': -15, 'azimuth': 0, 'center': pg.Vector(0,0,0)}
            distance = self.gl3d.cameraParams()['distance'] / 6800
            params = {'distance': distance, 'fov': 60}
        self.gl3d.setCameraParams(**params)

    def gui_set_scan_speed(self, value):
        if self.scan_timer.isActive():
            self.scan_timer.setInterval(value)

    def gui_set_par_generic(self, parameter, value):
        self.par[parameter] = value

    def get_sym_ops(self):
        # returns 2 matrices for '1' (?)
        if self.par['sample_point_group'] == '1':
            self.sym_mat = np.zeros((3,3), dtype=np.int8)
        else:
            self.sym_mat = np.vstack(get_sym_mat_from_coords(self.sym_dict[self.par['sample_point_group']])[1:]).reshape(-1, 3, 3).astype(np.int8)
        #self.sym_mat = np.vstack(get_sym_mat_from_coords(self.sym_dict[self.par['sample_point_group']])[1:]).reshape(-1, 3, 3)
        self.sym_mat_bases = self.sym_mat @ self.OM_UC.T * 1E-10

    def get_orientation_matrix(self, OM=None):
        # Make random orientation matrix
        if OM is None:
            ang1, ang2, ang3 = np.random.rand(3)
        else:
            ang1, ang2, ang3 = np.zeros(3)
        self.OM = Quaternion.from_axis_rotation(np.array([1, 0, 0]), ang1) * \
                  Quaternion.from_axis_rotation(np.array([0, 1, 0]), ang2) * \
                  Quaternion.from_axis_rotation(np.array([0, 0, 1]), ang3)

    def restart_new_cell(self):
        # get new parameters from GUI
        self.init_gui_parameters()
        # clear plot
        self.gl3d.clear()
        # redraw plot
        self.init_plot()
        # clear gui data table
        self.gui_dat_tab.setRowCount(0)

    def restart_check_enable(self):
        if (self.par['sample_cell_a'] != self.gui_sample_a.value() or
            self.par['sample_cell_b'] != self.gui_sample_b.value() or
            self.par['sample_cell_c'] != self.gui_sample_c.value() or
            self.par['sample_cell_alpha'] != self.gui_sample_alpha.value() or
            self.par['sample_cell_beta'] != self.gui_sample_beta.value() or
            self.par['sample_cell_gamma'] != self.gui_sample_gamma.value() or
            self.par['wavelength'] != self.gui_sample_wavelength.value() * 1E-10 or
            self.par['max_resolution'] != self.gui_prd_res.value() * 1E-10 or
            self.par['sample_point_group'] != self.gui_sample_point_group.currentText()):
            self.gui_sample_apply.setText('Apply Changes')
        else:
            self.gui_sample_apply.setText('Restart')

    def scan_increment(self):
        # This function calculates the measured hkls for the current goniometer angles
        # First, check the length of the scattering vector -> unity, ewald offset
        # Second, check the intersection point of diffracted beam vector and detector area
        R = self.gon_rot_ax3.to_rotation_matrix()
        
        # This is heavy, we need to speed this up -> future problems
        # - ideas: only calculate for remaining hkls, use KDTree for detector intersection
        # - use some clever pre-calculation of scattering vectors? No idea how, if, yet.
        # - use numba?
        qs = (R @ self.hkl_bases.T).T #* 1E-10
        kds = self.par['exp_incident_beam_vector'] + qs
        
        cond_ewald = abs(np.linalg.norm(kds, axis=1) - self.ewald_rad) <= self.par['ewald_offset']
        c_hkls = self.hkls[cond_ewald]
        c_bases = self.hkl_bases[cond_ewald]
        c_qs = qs[cond_ewald]
        c_kds = kds[cond_ewald]

        TTH = self.gon_rot_tth.inverse().to_rotation_matrix()
        c_kds = (TTH @ c_kds.T).T
        # real space length scale, in [m]
        rsl = self.par['det_distance'] * 1E10 / c_kds[:,2]
        # real space scattering vectors
        rsvs = c_kds * rsl[:,None]
        # x, y projection, relative to beam center, in pixel
        pxs = (rsvs[:,0] * 1E-10 + self.par['det_poni_x']) / self.par['det_pix_s']
        pys = (rsvs[:,1] * 1E-10 + self.par['det_poni_y']) / self.par['det_pix_s']
        c_xys = np.vstack([pxs, pys]).T
        cond_visible = np.where((c_xys[:,0] > 0)&
                                (c_xys[:,1] > 0)&
                                (c_xys[:,0] < self.par['det_pix_x'])&
                                (c_xys[:,1] < self.par['det_pix_y'])&
                                (c_kds[:,2] > 0))

        v_qs = c_qs[cond_visible]
        v_det = np.zeros((0,3), dtype=float)
        v_pc = np.zeros((0,3), dtype=float)
        if len(c_hkls[cond_visible]) > 0:
            v_l0 = np.array([0, 0, -self.ewald_rad])
            v_kds = v_qs - v_l0
            # v_p0 not equal to -v_n after detector correction (roll, pitch, yaw)
            v_p0 = np.array([0, 0, self.plot_det_dist]) @ TTH
            v_n = np.array([0, 0, -self.plot_det_dist]) @ TTH
            v_scale = ((v_p0 - v_l0) @ v_n) / (v_kds @ v_n)
            v_det = v_kds * v_scale[:, None] + v_l0
            v_pc = correction_partiality_geometric(v_kds, self.ewald_rad, self.par['sfx_spectral_width'], self.par['sfx_rlp_size'])

        if len(v_qs) == 0:
            self.plot_diff_vecs.setData(edges=np.zeros((0,2), dtype=int), nodePositions=np.zeros((0,3)))
            self.plot_scat_vecs.setData(edges=np.zeros((0,2), dtype=int), nodePositions=np.zeros((0,3)))
        else:
            edges = np.vstack([np.zeros(len(v_det), dtype=int), np.arange(1, len(v_det)+1, 1)]).T
            nodes = np.vstack([np.array([0, 0, -self.ewald_rad]), v_det])
            self.plot_diff_vecs.setData(edges=edges, nodePositions=nodes)
            if self.show_scat_vecs:
                edges = np.vstack([np.zeros(len(v_qs), dtype=int), np.arange(1, len(v_qs)+1, 1)]).T
                nodes = np.vstack([np.array([0, 0, 0]), v_qs])
                self.plot_scat_vecs.setData(edges=edges, nodePositions=nodes)
        
        return pd.DataFrame(np.hstack([c_hkls[cond_visible], c_bases[cond_visible], v_det, v_pc]), columns=['h', 'k', 'l', 'bx', 'by', 'bz', 'dx', 'dy', 'dz', 'pc'])

    def scan_update(self):
        # Update the gui to rotate the scene objects (goniometer)
        # value update calls the rotate_gon() function
        self.scan_progress += self.gui_scan_step.value()
        self.scan_progress = round(self.scan_progress, 6)
        
        # update goniometer angles
        if self.gui_scan_axis.currentText() == 'Phi':
            self.par['gon_phi_ang'] += self.gui_scan_step.value()
            self.gui_gon_phi.setValue(self.par['gon_phi_ang'])
        elif self.gui_scan_axis.currentText() == 'Omega':
            self.par['gon_omg_ang'] += self.gui_scan_step.value()
            self.gui_gon_omg.setValue(self.par['gon_omg_ang'])
        elif self.gui_scan_axis.currentText() == 'SFX':
            self.par['gon_phi_ang'] = np.rad2deg(np.random.rand() * 2 * np.pi)
            self.gui_gon_phi.setValue(self.par['gon_phi_ang'])
            self.par['gon_chi_ang'] = np.rad2deg(np.random.rand() * 2 * np.pi)
            self.gui_gon_chi.setValue(self.par['gon_chi_ang'])
            self.par['gon_omg_ang'] = np.rad2deg(np.random.rand() * 2 * np.pi)
            self.gui_gon_omg.setValue(self.par['gon_omg_ang'])
        
        # scan and get new data
        _scan_data = self.scan_increment()

        # append/display new data
        if len(_scan_data) > 0:
            self.scan_data = self.scan_data.merge(_scan_data, how='outer')
            if not self.sfx_keep_all_data:
                self.scan_data.drop_duplicates(subset=['h','k','l'], keep='last', inplace=True)
            self.scan_data.reset_index(drop=True, inplace=True)
            # plot the current scan data
            self.scat_scan.setData(pos=_scan_data[['bx', 'by', 'bz']])
            # plot the collected data
            self.scat_data.setData(pos=self.scan_data[['bx', 'by', 'bz']])
            # plot the diffraction pattern on the detector
            self.scat_pattern.setData(pos=self.scan_data[['dx', 'dy', 'dz']])
            # calculate symmetry equivalents
            # set of collected data
            # we need to round the floats to avoid precision issues, 10 digits should be enough
            _set_symm_collected = set(map(tuple, np.round(self.scan_data[['bx', 'by', 'bz']].to_numpy(), 10)))
            # set of all symmetry equivalents of current scan
            _set_symm_equivalent = set(map(tuple, (np.round(_scan_data[['h','k','l']].to_numpy() @ self.sym_mat_bases, 10).reshape(-1,3))))
            # add new symmetry equivalents to collected set
            # subtract already collected data
            self.scan_symm = (self.scan_symm | _set_symm_equivalent) - _set_symm_collected
            self.scat_symm.setData(pos=list(self.scan_symm))

            # hkl text labels
            for i in self.text_hkls_list:
                i.setData(text='')
            for i, row in _scan_data.iterrows():
                if i >= self.par['plot_max_hkl_labels']:
                    break
                # calculate positionfrom hkl
                #base = np.array(dat) @ self.OM_UC.T * 1E-10
                base = row[['bx', 'by', 'bz']].to_numpy()
                h, k, l = row[['h', 'k', 'l']].to_numpy(dtype=int)
                pc = row['pc']
                if self.gui_show_sfx_parcor.isChecked():
                    self.text_hkls_list[i].setData(pos=base, text=f'{h} {k} {l} ({pc:.4f})')
                else:
                    self.text_hkls_list[i].setData(pos=base, text=f'{h} {k} {l}')

        # update progress display
        self.gui_scan_total.setText(f'{self.hkls_num}')
        self.gui_scan_completeness.setText(f'{len(self.scan_data)/self.hkls_num*100:.0f} % ({(len(self.scan_symm)+len(self.scan_data))/self.hkls_num*100:.0f} %)')
        self.gui_scan_collected.setText(f'{len(self.scan_data)} ({len(self.scan_symm)+len(self.scan_data)})')

        # check scan progress
        if self.scan_progress >= self.gui_scan_range.value():
            self.scan_finished()

    def scan_finished(self):
        self.scan_progress = 0
        self.scan_timer.stop()
        self.gui_start_scan.setText('Start Scan')
        self.gui_start_scan.setStyleSheet("background-color: #006A67")

        # sort data by h, k, l
        self.scan_data = self.scan_data.sort_values(['h','k','l'], key=abs)
        # calculate d-spacings
        self.scan_data['d'] = 1 / np.linalg.norm(self.scan_data[['h','k','l']] @ self.OM_UC.T, axis=1) * 1E10
        # Add data to table
        self.gui_dat_tab.setRowCount(len(self.scan_data))
        # reset column count
        self.gui_dat_tab.setColumnCount(5)
        # fill table, all data
        for i, row in self.scan_data.iterrows():
            self.gui_dat_tab.setItem(i, 0, QtWidgets.QTableWidgetItem(str(int(row['h']))))
            self.gui_dat_tab.setItem(i, 1, QtWidgets.QTableWidgetItem(str(int(row['k']))))
            self.gui_dat_tab.setItem(i, 2, QtWidgets.QTableWidgetItem(str(int(row['l']))))
            self.gui_dat_tab.setItem(i, 3, QtWidgets.QTableWidgetItem(str(f'{row['d']:.2f}')))
            self.gui_dat_tab.setItem(i, 4, QtWidgets.QTableWidgetItem(str(f'{row['pc']:.4f}')))
        # remove partiality column if not requested
        if not self.gui_show_sfx_parcor.isChecked():
            self.gui_dat_tab.setColumnCount(4)

    def scan_toggle(self):
        if not self.scan_timer.isActive():
            self.gui_start_scan.setText('Pause Scan')
            self.gui_start_scan.setStyleSheet("background-color: #A02334")
            self.scan_timer.setInterval(self.gui_scan_speed.value())
            self.scan_timer.start()
        else:
            self.gui_start_scan.setText('Continue Scan')
            self.gui_start_scan.setStyleSheet("background-color: #006A67")
            self.scan_timer.stop()

    def look_along(self, vec):
        # Helper to set camera to look along a given axis vector
        v = np.array(vec, dtype=float)
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return
        # elevation: angle above the XY-plane
        elevation = np.rad2deg(np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2)))
        # azimuth: angle in the XY-plane from +X
        azimuth = np.rad2deg(np.arctan2(v[1], v[0]))
        params = self.gl3d.cameraParams()
        distance = params.get('distance', None)
        if distance is None:
            self.gl3d.setCameraPosition(elevation=elevation, azimuth=azimuth)
        else:
            self.gl3d.setCameraPosition(QtGui.QVector3D(0, 0, 0), distance=distance, elevation=elevation, azimuth=azimuth)
            #self.gl3d.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)

    def keyPressEvent(self, ev):
        # Compute the cell axes (in meters) and apply current goniometer rotation (phi, chi, omega)
        if ev.key() == QtCore.Qt.Key.Key_F1:
            # look down rotated a axis
            self.look_along(self.gon_rot_ax3.to_rotation_matrix() @ (self.OM_UC * 1E-10)[:, 0])
        elif ev.key() == QtCore.Qt.Key.Key_F2:
            # look down rotated b axis
            self.look_along(self.gon_rot_ax3.to_rotation_matrix() @ (self.OM_UC * 1E-10)[:, 1])
        elif ev.key() == QtCore.Qt.Key.Key_F3:
            # look down rotated c axis
            self.look_along(self.gon_rot_ax3.to_rotation_matrix() @ (self.OM_UC * 1E-10)[:, 2])
        elif ev.key() == QtCore.Qt.Key.Key_F4:
            # look down current omega goniometer axis (apply combined gon rotation)
            self.look_along(self.gon_rot_ax1.to_rotation_matrix() @ self.par['gon_omg_axs'])
        elif ev.key() == QtCore.Qt.Key.Key_F5:
            # look down current chi axis
            self.look_along(self.gon_rot_ax2.to_rotation_matrix() @ self.par['gon_chi_axs'])
        elif ev.key() == QtCore.Qt.Key.Key_F6:
            # look down current phi axis
            self.look_along(self.gon_rot_ax3.to_rotation_matrix() @ self.par['gon_phi_axs'])
        elif ev.key() == QtCore.Qt.Key.Key_C:
            self.gl3d.cameraParams()
        elif ev.key() == QtCore.Qt.Key.Key_S:
            if self.scan_timer.isActive():
                self.scan_timer.stop()
            else:
                self.scan_toggle()

    def rotate_gon(self):
        # Apply goniometer rotation changes from GUI to all relevant items in the 3D scene
        # calculate new goniometer rotation
        gon_rot_ax3_new = Quaternion.from_axis_rotation(self.par['gon_phi_axs'], np.deg2rad(self.gui_gon_phi.value())) * \
                          Quaternion.from_axis_rotation(self.par['gon_chi_axs'], np.deg2rad(self.gui_gon_chi.value())) * \
                          Quaternion.from_axis_rotation(self.par['gon_omg_axs'], np.deg2rad(self.gui_gon_omg.value()))
        gon_rot_ax2_new = Quaternion.from_axis_rotation(self.par['gon_chi_axs'], np.deg2rad(self.gui_gon_chi.value())) * \
                          Quaternion.from_axis_rotation(self.par['gon_omg_axs'], np.deg2rad(self.gui_gon_omg.value()))
        gon_rot_ax1_new = Quaternion.from_axis_rotation(self.par['gon_omg_axs'], np.deg2rad(self.gui_gon_omg.value()))

        # calculate rotation delta
        gon_rot_ax3_delta = self.gon_rot_ax3.inverse() * gon_rot_ax3_new
        gon_rot_ax2_delta = self.gon_rot_ax2.inverse() * gon_rot_ax2_new
        gon_rot_ax1_delta = self.gon_rot_ax1.inverse() * gon_rot_ax1_new

        # update stored goniometer rotation
        self.gon_rot_ax3 = gon_rot_ax3_new
        self.gon_rot_ax2 = gon_rot_ax2_new
        self.gon_rot_ax1 = gon_rot_ax1_new

        # calculate combined rotation
        ax3, ang3 = gon_rot_ax3_delta.get_axis_angle()
        ax2, ang2 = gon_rot_ax2_delta.get_axis_angle()
        ax1, ang1 = gon_rot_ax1_delta.get_axis_angle()

        # convert to degrees
        ang3 = np.rad2deg(ang3)
        ang2 = np.rad2deg(ang2)
        ang1 = np.rad2deg(ang1)

        # apply combined rotation delta to all relevant items
        self.scat_latt.rotate(ang3, *ax3)
        self.scat_scan.rotate(ang3, *ax3)
        self.scat_data.rotate(ang3, *ax3)
        self.scat_symm.rotate(ang3, *ax3)
        for i in range(self.par['plot_max_hkl_labels']):
            self.text_hkls_list[i].rotate(ang3, *ax3)
        self.plot_ax_a.rotate(ang3, *ax3)
        self.plot_ax_b.rotate(ang3, *ax3)
        self.plot_ax_c.rotate(ang3, *ax3)
        self.plot_cell.rotate(ang3, *ax3)

        # rotate goniometer axes
        self.plot_gon_phi.rotate(ang3, *ax3)
        self.plot_gon_chi.rotate(ang2, *ax2)
        self.plot_gon_omg.rotate(ang1, *ax1)

    def rotate_gon_tth(self, relative=True):
        # Apply 2theta rotation changes from GUI to all relevant items in the 3D scene
        # calculate new 2theta rotation
        if self.gui_gon_tth_orientation.currentText() == 'Vertical':
            gon_rot_new = Quaternion.from_axis_rotation(self.par['gon_ttv_axs'], np.deg2rad(self.gui_gon_tth.value()))
        elif self.gui_gon_tth_orientation.currentText() == 'Horizontal':
            gon_rot_new = Quaternion.from_axis_rotation(self.par['gon_tth_axs'], np.deg2rad(self.gui_gon_tth.value()))
        else:
            pass
        
        if relative:
            gon_rot_delta = self.gon_rot_tth.inverse() * gon_rot_new
            self.gon_rot_tth = gon_rot_new
            axs, ang = gon_rot_delta.get_axis_angle()
            ang = np.rad2deg(ang)
        else:
            axs, ang = gon_rot_new.get_axis_angle()
            ang = np.rad2deg(ang)

        self.plot_detector.rotate(ang, *axs)
        # Rotate the mesh detector and the frame so they stay aligned
        self.detector_plane.rotate(ang, *axs)
        self.detector_frame.rotate(ang, *axs)

if __name__ == '__main__':
    app = pg.mkQApp()
    win = Visualizer()
    win.show()
    app.exec()
