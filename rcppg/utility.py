import re
import sympy
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

class GLWrappedTextItem(gl.GLTextItem):
    def paint(self):
        if len(self.text) < 1:
            return
        self.setupGLState()

        project = self.compute_projection()
        vec3 = QtGui.QVector3D(*self.pos)
        text_rect = self.align_text(project.map(vec3).toPointF())

        painter = QtGui.QPainter(self.view())
        painter.setPen(self.color)
        painter.setBackground(QtGui.QBrush(QtGui.QColor(0,0,0,64)))
        painter.setBackgroundMode(QtCore.Qt.BGMode.OpaqueMode)
        painter.setFont(self.font)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        painter.drawText(text_rect, self.alignment, self.text)
        #painter.drawRect(text_rect)
        painter.end()

    def align_text(self, pos:QtCore.QPointF, margin:float=1.05) -> QtCore.QRectF:
        """
        Aligns the text at the given position according to the given alignment.
        """
        font_metrics = QtGui.QFontMetrics(self.font)
        text_box_size = np.array([(font_metrics.boundingRect(i).width(), font_metrics.boundingRect(i).height()) for i in self.text.splitlines()])
        text_rect = QtCore.QRectF(0, 0, np.max(text_box_size[:, 0]) * margin, np.sum(text_box_size[:, 1]) * margin)
        
        dx = dy = 0.0
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignRight:
            dx = text_rect.width()
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignHCenter:
            dx = text_rect.width() / 2.0
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignBottom:
            dy = text_rect.height()
        if self.alignment & QtCore.Qt.AlignmentFlag.AlignVCenter:
            dy = text_rect.height() / 2.0
        
        pos.setX(pos.x() - dx)
        pos.setY(pos.y() + dy)
        text_rect.moveTopLeft(pos)
        
        return text_rect

class CenterGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scatter_items = []  # Track attached scatter plot objects
        self.centered = False  # Track whether the view is centered on a scatter point

    def addItem(self, item, center=False):
        if isinstance(item, gl.GLScatterPlotItem) and center:
            self.scatter_items.append(item)
        super().addItem(item)

    def removeItem(self, item):
        if item in self.scatter_items:
            self.scatter_items.remove(item)
        super().removeItem(item)

    def resetCenter(self):
        """Resets the rotation center back to the global origin (0, 0, 0)

        without altering active azimuth or elevation angles.
        """
        self.applyNewCenter(np.array([0.0, 0.0, 0.0]))
        self.centered = False

    def mouseMoveEvent(self, ev):
        # Full implementation details and code can be found in the referenced documents.
        lpos = ev.position()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton and ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            speed_scale = 0.1
            dx = diff.x() * speed_scale
            dy = diff.y() * speed_scale
            if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                self.pan(dx, dy, 0, relative='view')
            else:
                self.orbit(-dx, dy)
            ev.accept()
        else:
            self.mousePos = lpos - diff
            super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            # Handle Double Right-Click to Reset
            if ev.type() == ev.Type.MouseButtonDblClick:
                self.resetCenter()
                ev.accept()
                return
            
            # Handle Single Right-Click to Snap onto a Scatter Point
            elif ev.type() == ev.Type.MouseButtonPress:
                if self.snapCameraToClickedPoint(ev.pos(), max_pixel_distance=100):
                    ev.accept()
                    return
                
        super().mousePressEvent(ev)

    def snapCameraToClickedPoint(self, mouse_pos, max_pixel_distance=50):
        """Projects 3D scatter node coordinates into 2D viewport pixels

        using direct MVP matrix multiplication and locks onto the closest match.
        """
        if not self.scatter_items:
            return False

        # 1. Reconstruct combined Model-View-Projection (MVP) matrix
        vm = self.viewMatrix()
        pm = QtGui.QMatrix4x4()
        
        is_ortho = getattr(self, 'opts', {}).get('type', '') == 'ortho' or getattr(self, 'opts', {}).get('fov') == 0
        d = self.opts['distance']
        fov = self.opts.get('fov', 60)

        near_clip = 1.0
        far_clip = max(1000000.0, d * 5.0) 
        
        if is_ortho:
            pm.ortho(-d, d, -d, d, near_clip, far_clip)
        else:
            pm.perspective(fov, float(self.width()) / self.height(), near_clip, far_clip)

        mvp = pm * vm
        
        closest_point_3d = None
        min_pixel_dist = float('inf')

        # 2. Find the mathematically closest point to the cursor in pixel space
        for item in self.scatter_items:
            if item.pos is None:
                continue
            
            points_3d = np.asarray(item.pos)
            for pt in points_3d:
                # Project 3D point into homogeneous Clip Space
                clip_pos = mvp * QtGui.QVector4D(float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
                if clip_pos.w() <= 0:
                    continue
                    
                # Perspective Division -> Normalized Device Coordinates [-1, 1]
                ndc_x = clip_pos.x() / clip_pos.w()
                ndc_y = clip_pos.y() / clip_pos.w()
                if abs(ndc_x) > 1.0 or abs(ndc_y) > 1.0:
                    continue

                # NDC -> 2D Window Screen Pixels
                screen_x = (ndc_x + 1.0) * self.width() / 2.0
                screen_y = (1.0 - ndc_y) * self.height() / 2.0
                
                # Check flat pixel Euclidean distance from cursor position
                pixel_dist = np.hypot(screen_x - mouse_pos.x(), screen_y - mouse_pos.y())
                if pixel_dist < min_pixel_dist and pixel_dist <= max_pixel_distance:
                    min_pixel_dist = pixel_dist
                    closest_point_3d = pt

        # 3. Apply the center update if an eligible node was matched
        if closest_point_3d is not None:
            self.applyNewCenter(closest_point_3d)
            return True

        return False

    def applyNewCenter(self, target_point):
        """Uniformly handles updating focal center coordinates without shifting view perspective."""
        old_center = np.array([self.opts['center'].x(), self.opts['center'].y(), self.opts['center'].z()])
        delta = target_point - old_center

        is_ortho = getattr(self, 'opts', {}).get('type', '') == 'ortho' or getattr(self, 'opts', {}).get('fov') == 0

        if is_ortho:
            self.opts['center'] = pg.Vector(target_point[0], target_point[1], target_point[2])
        else:
            # Shift focal look-at center while locking spatial viewing angles
            self.pan(float(delta[0]), float(delta[1]), float(delta[2]), relative='global')
            
            # Recalculate camera distance metric to avoid sudden zooming/jumping
            cam_pos = np.array([self.cameraPosition().x(), self.cameraPosition().y(), self.cameraPosition().z()])
            self.opts['distance'] = np.linalg.norm(cam_pos - target_point)
        self.centered = True
        self.update()

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
    return pc.reshape(-1,1) # Divide by this.

def calc_angles(v1, v2):
    return np.arccos(np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1))).reshape(-1,1)

def correction_partiality_geometric(kds, ewald_radius, spectral_width=1, rlp_size=1):
    diff_beam_2theta = calc_angles(kds, np.array([[-1,0,0]]))
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
