from abc import ABC, abstractmethod
import numpy as np
import meep as mp
from meep.materials import Au, Ti, SiO2
from utils.geometry_utils import *
xm = 1000

# =========================================================
# Base class
# =========================================================

class AntennaBase(ABC):

    @abstractmethod
    def build_geometry(self):
        pass

    @abstractmethod
    def bounding_box(self):
        pass


# =========================================================
# BowTie
# =========================================================

class BowTieEquilateral(AntennaBase):

    def __init__(self,
                 gap,
                 amp,
                 thickness,
                 radius,
                 material,
                 z_offset=0.0,
                 center=(0.0, 0.0)):

        self.gap = gap
        self.corected_gap = gap
        self.amp = amp
        self.thickness = thickness
        self.radius = radius
        self.material = material
        self.z_offset = z_offset
        self.center = np.array(center)

    # -----------------------------------------------------

    def build_geometry(self):

        if self.radius > 1e-12:
            self.corected_gap = corrected_gap(
                self.gap,
                self.radius,
                np.deg2rad(60))
            # print(f"Corrected gap for radius {self.radius*1e3:.1f} nm: {self.gap*1e3:.2f} nm")

        # Right triangle tip
        P1 = np.array([
            self.center[0] + self.corected_gap/2.0,
            self.center[1]
        ])

        # Equilateral triangle assumption
        P2 = P1 + self.amp * np.array([
            1.0,
            np.tan(np.deg2rad(30))
        ])

        P3 = P2 * np.array([1.0, -1.0])

        tip_right = [
            mp.Vector3(*P1),
            mp.Vector3(*P2),
            mp.Vector3(*P3)
        ]

        # Mirror on Y axis
        mirror = np.array([-1.0, 1.0])

        tip_left = [
            mp.Vector3(*(P1 * mirror)),
            mp.Vector3(*(P2 * mirror)),
            mp.Vector3(*(P3 * mirror))
        ]

        x_centroid = self.amp * 2/3 + self.corected_gap/2.0
        bow_tie = [
            mp.Prism(
                tip_right,
                height=self.thickness,
                material=self.material,
                center=mp.Vector3(x_centroid, 0, self.z_offset)
            ),
            mp.Prism(
                tip_left,
                height=self.thickness,
                material=self.material,
                center=mp.Vector3(-x_centroid, 0, self.z_offset)
            )
        ]

        # Fillets / rounding
        if self.radius > 1e-12:

            bow_tie += clear_edges(
                points=[P1, P2, P3],
                radius=self.radius,
                height=self.thickness,
                z_offset=self.z_offset
            )

            bow_tie += clear_edges(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                radius=self.radius,
                height=self.thickness,
                z_offset=self.z_offset
            )

            bow_tie += fillet_polygon(
                points=[P1, P2, P3],
                radius=self.radius,
                height=self.thickness,
                material=self.material,
                z_offset=self.z_offset
            )

            bow_tie += fillet_polygon(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                radius=self.radius,
                height=self.thickness,
                material=self.material,
                z_offset=self.z_offset
            )

        return bow_tie

    # -----------------------------------------------------

    def bounding_box(self):

        return [
            2*self.amp + self.gap,
            2*self.amp*np.tan(np.deg2rad(30)),
            self.thickness
        ]