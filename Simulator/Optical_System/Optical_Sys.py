import optiland
import numpy as np
from optiland import materials  # if using specific materials
import optiland.backend as be  # backend for numerical operations - either numpy or torch
from optiland.optic import Optic
from optiland.physical_apertures import RectangularAperture, DifferenceAperture, RadialAperture
import torch
from ..utils.helper import reflect_with_position_aperture, plane_angles_from_normal, calculate_position
"""
ATTENTION PLEASE:

This Optical System is based on Weiru's RL optical system
    (F) \                            y  
----→----\- (Back of mirror)        |
        | \-                        |  /x
        |  \                        | /
     -\ ↓                           |/_ _ _ _ _ _ z (Global coordinate)
      -\|              
       -\----→--------
        -\

        
Init normal vector of Mirror is [0, 0, 1]        
Cause the mirror has back and front plane, so the mirror 1 angle here is 135°(Rx), Mirror 2 is -45°/315°(Rx).
> Right hand rule applies for rotation direction.

Thie version of Oprical System has the following features:
1. The "dectectors" are placed right at the surface of the mirror 2 and the surfance of the two pinholes.
2. All the units are in mm and radian. And the size is in radius.
3. The aperture should be difined as a list[R_max, R_min], for example [5.0, 0.0] means the the region from 0mm to 5.0mm radius is valid.
4. The detector is defined as a rectangular aperture, and you can set the x_size and y_size.
5. You can change the pinhole 1 size after the optical system is initialized by calling set_pinhole_size(enlarge=True/False) function.

For the pinhole aperture, only R_min<r<R_max can pass ligtht, others will be blocked.

If you want to get a valid sys, you must caculate the correct positions of mirror 1 and mirror 2, you can use the funcs in 'helper.py' to calculate.
"""


class OpticalSystem(Optic):
    def __init__(self,
                 beam_size: float = 0.5,
                 wavelength: float = 0.78,
                 mirror_aperture: list = [12.7, 0.0],
                 pinhole_aperture: list = [0.5, 0.0],
                 detector_aperture: list = [25.0, 25.0],
                 mirror1_position: np.ndarray = np.array([0, 0, 100.0]),
                 mirror2_position: np.ndarray = np.array([0, -100.0, 100.0]),
                 rotation_angles_mirror1: np.ndarray = np.array([np.pi/2+np.pi/4, 0.0, 0.0]),  # [rx1, ry1, rz1]
                 rotation_angles_mirror2: np.ndarray = np.array([np.pi/4, 0.0, 0.0]),  # [rx2, ry2, rz2]
                 pinhole_dist: float = 100.0,
                 m2_a1_dist: float = 20.0,
                 pinhole_max_size: float = 4.0):
        """
        Construct an optical system consisting of two mirrors, two apertures,
        and two detectors, with configurable geometry and beam properties.
        ----------
        mirror_aperture : list [r_max, r_min], default [12.7, 0.0] Radial aperture of the mirrors (outer and inner radius in mm).
        pinhole_aperture : list [r_max, r_min], default [0.5, 0.0] Radial aperture size of the pinhole/aperture (in mm).
        detector_aperture : list [x_size, y_size], default [25.0, 25.0] Rectangular detector size along x and y directions (in mm).
        beam_size : float, default 1.0  Diameter of the incident beam (in mm).
        wavelength : float, default 0.78 Wavelength of the incoming light (in µm).
        mirror1_position : np.ndarray, default [0, 0, 100.0] Position of the first mirror in Cartesian coordinates [x, y, z].
        mirror2_position : np.ndarray, default [0, -100, 100.0] Position of the first mirror in Cartesian coordinates [x, y, z].
        rotation_angles_mirror1 : np.ndarray, default [135.0, 0.0, 0.0] Rotation angles [rx, ry, rz] of the first mirror in degrees.
        rotation_angles_mirror2 : np.ndarray, default [-45.0, 0.0, 0.0] Rotation angles [rx, ry, rz] of the second mirror in degrees.
        pinhole_dist : float, default 100.0 Distance between pinhole 1 and pinhole 2 (in mm).
        m2_a1_dist : float, default 20.0 Distance between mirror 2 and pinhole 1 (in mm).
        """

        super().__init__()

        self.is_valid = True
        # Ensure mirror1_position is a numpy array for easier vector math
        self.mirror1_position = np.array(mirror1_position)
        self.mirror2_position = np.array(mirror2_position)
        self.pinhole_max_size = pinhole_max_size
        self.pinhole_aperture = pinhole_aperture

        # Compute the positions
        p1_hit, d1_out = reflect_with_position_aperture(mirror_pos=mirror1_position,
                                                           aperture=mirror_aperture[0],
                                                            rx=rotation_angles_mirror1[0],
                                                            ry=rotation_angles_mirror1[1],
                                                            rz=rotation_angles_mirror1[2])

        p2_hit, d2_out = reflect_with_position_aperture(r0=p1_hit,
                                                    d_in=d1_out,
                                                    mirror_pos=self.mirror2_position,
                                                    aperture=mirror_aperture[0],
                                                    rx=rotation_angles_mirror2[0],
                                                    ry=rotation_angles_mirror2[1],
                                                    rz=rotation_angles_mirror2[2])
        if p2_hit is False: 
            self.is_valid = False
            return

        self.pinhole1_position = calculate_position(p2_hit, d2_out, m2_a1_dist)
        self.pinhole2_position = calculate_position(p2_hit, d2_out, m2_a1_dist + pinhole_dist)
    

        rx_plane, ry_plane, rz_plane = plane_angles_from_normal(-d2_out)
        rotation_angles = [rx_plane, ry_plane, rz_plane]

        # Define the physical aperture
        mirror_aperture = RadialAperture(r_max=mirror_aperture[0], r_min=mirror_aperture[1])
        ap = RadialAperture(r_max=pinhole_aperture[0], r_min=pinhole_aperture[1])
        ap_2 = RadialAperture(r_max=beam_size, r_min=0)
        ap_detector1 = RectangularAperture(x_max=detector_aperture[0],
                                           x_min=-detector_aperture[0],
                                           y_max=detector_aperture[1],
                                           y_min=-detector_aperture[1])

        

        # Adding surfaces to the system
        self.add_surface(index=0, radius=np.inf, z=-np.inf, comment="Light")  # parallel light
        self.add_surface(index=1, z=0, aperture=ap_2, comment="Aperture")

        # Mirror 1
        self.add_surface(
            index=2,
            z=self.mirror1_position[2],  # Take the z position of mirror1_position
            radius=np.inf,
            material="mirror",
            rx=rotation_angles_mirror1[0],
            ry=rotation_angles_mirror1[1],
            surface_type="standard",
            aperture=mirror_aperture,
            comment="Mirror1"
        )

        # Mirror 2 detector
        self.add_surface(
            index=3,
            z=self.mirror2_position[2],
            x=self.mirror2_position[0],
            y=self.mirror2_position[1],
            radius=np.inf,
            rx=rotation_angles_mirror2[0],
            ry=rotation_angles_mirror2[1],
            aperture=ap_detector1,
            comment="detector"
        )

        # Mirror 2
        self.add_surface(
            index=4,
            z=self.mirror2_position[2],
            x=self.mirror2_position[0],
            y=self.mirror2_position[1],
            radius=np.inf,
            material="mirror",
            rx=rotation_angles_mirror2[0],
            ry=rotation_angles_mirror2[1],
            surface_type="standard",
            aperture=mirror_aperture,
            comment="Mirror2"
        )

        # Detector 1
        self.add_surface(
            index=5,
            x=self.pinhole1_position[0],
            z=self.pinhole1_position[2],
            y=self.pinhole1_position[1],
            rx=rotation_angles[0],
            ry=rotation_angles[1],
            rz=rotation_angles[2],
            aperture=ap_detector1,
            comment="Detector"
        )

        # Aperture 1
        self.add_surface(
            index=6,
            x=self.pinhole1_position[0],
            z=self.pinhole1_position[2],
            y=self.pinhole1_position[1],
            rx=rotation_angles[0],
            ry=rotation_angles[1],
            rz=rotation_angles[2],
            material="glass",
            aperture=ap,
            is_stop=True,
            comment="Aperture1"
        )

        self.add_surface(
            index=7,
            x=self.pinhole1_position[0],
            z=self.pinhole1_position[2],
            y=self.pinhole1_position[1],
            rx=rotation_angles[0],
            ry=rotation_angles[1],
            rz=rotation_angles[2],
            aperture=ap,
            is_stop=True,
            comment="Aperture1"
        )

        # Detector 2
        self.add_surface(
            index=8,
            x=self.pinhole2_position[0],
            z=self.pinhole2_position[2],
            y=self.pinhole2_position[1],
            rx=rotation_angles[0],
            ry=rotation_angles[1],
            rz=rotation_angles[2],
            aperture=ap_detector1,
            comment="Detector"
        )

        # Aperture 2
        self.add_surface(
            index=9,
            x=self.pinhole2_position[0],
            z=self.pinhole2_position[2],
            y=self.pinhole2_position[1],
            rx=rotation_angles[0],
            ry=rotation_angles[1],
            rz=rotation_angles[2],
            material="glass",
            surface_type="standard",
            aperture=ap,
            is_stop=True,
            comment="Aperture2"
        )
        self.add_surface(
            index=10,
            x=self.pinhole2_position[0],
            z=self.pinhole2_position[2],
            y=self.pinhole2_position[1],
            rx=rotation_angles[0],
            ry=rotation_angles[1],
            rz=rotation_angles[2],
            material="air",
            surface_type="standard",
            aperture=ap,
            is_stop=True,
            comment="Aperture2"
        )        

        # Basic settings
        self.set_aperture(aperture_type="EPD", value=beam_size*2)
        self.set_field_type(field_type="angle")
        self.add_field(y=0)
        self.add_wavelength(value=wavelength, is_primary=True)

    def set_mirror_angle(self, rx1=None, ry1=None, rx2=None, ry2=None, gradient: bool = False):

        """
        Set the rotation angles of mirror 1 and mirror 2.

        Parameters
        ----------
        rx1 : float or torch.Tensor, optional
            Rotation angle of mirror 1 around the x-axis (in radians).
        ry1 : float or torch.Tensor, optional
            Rotation angle of mirror 1 around the y-axis (in radians).
        rx2 : float or torch.Tensor, optional
            Rotation angle of mirror 2 around the x-axis (in radians).
        ry2 : float or torch.Tensor, optional
            Rotation angle of mirror 2 around the y-axis (in radians).
        """


        cs2 = self.surface_group.surfaces[2].geometry.cs
        cs3 = self.surface_group.surfaces[3].geometry.cs
        cs4 = self.surface_group.surfaces[4].geometry.cs

        def _assign_scalar(tgt: torch.Tensor, val):
            if isinstance(val, torch.Tensor):
                v = val.to(dtype=tgt.dtype, device=tgt.device)
                if v.numel() == 1:
                    tgt.fill_(v.item())
                else:
                    assert v.shape == tgt.shape, f"shape mismatch: {v.shape} vs {tgt.shape}"
                    tgt.copy_(v)
            else:
                tgt.fill_(float(val))
        if gradient == True:
            with torch.no_grad():
                if rx1 is not None:
                    _assign_scalar(cs2.rx, rx1)
                if ry1 is not None:
                    _assign_scalar(cs2.ry, ry1)

                if rx2 is not None:
                    _assign_scalar(cs4.rx, rx2)
                if ry2 is not None:
                    _assign_scalar(cs4.ry, ry2)
        else:
            if rx1 is not None:
                cs2.rx = rx1
            if ry1 is not None:
                cs2.ry = ry1
            if rx2 is not None:
                cs4.rx = rx2
                cs3.rx = rx2   # detector follows mirror2
            if ry2 is not None:
                cs4.ry = ry2
                cs3.ry = ry2


    def get_mirror_angle(self):
        cs2 = self.surface_group.surfaces[2].geometry.cs
        cs4 = self.surface_group.surfaces[4].geometry.cs
        return [cs2.rx, cs2.ry, cs4.rx, cs4.ry]

    def set_pinhole_size(self,enlarge=False):
        if enlarge is False:
            self.surface_group.surfaces[6].aperture.r_max=self.pinhole_aperture[0]
            self.surface_group.surfaces[7].aperture.r_max=self.pinhole_aperture[0]
        else:
            self.surface_group.surfaces[6].aperture.r_max=self.pinhole_max_size

            self.surface_group.surfaces[7].aperture.r_max=self.pinhole_max_size

