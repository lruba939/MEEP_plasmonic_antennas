from abc import ABC, abstractmethod
import numpy as np
import meep as mp
from meep.materials import Au, Ti, SiO2
from src.geometry import *
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
                 center=(0.0, 0.0)):

        self.gap = gap
        self.amp = amp
        self.thickness = thickness
        self.radius = radius
        self.material = material
        self.center = np.array(center)

    # -----------------------------------------------------

    def build_geometry(self):

        if self.radius > 1e-12:
            self.gap = corrected_gap(
                self.gap,
                self.radius,
                np.deg2rad(60))

        # Right triangle tip
        P1 = np.array([
            self.center[0] + self.gap/2.0,
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

        bow_tie = [
            mp.Prism(
                tip_right,
                height=self.thickness,
                material=self.material
            ),
            mp.Prism(
                tip_left,
                height=self.thickness,
                material=self.material
            )
        ]

        # Fillets / rounding
        if self.radius > 1e-12:

            bow_tie += clear_edges(
                points=[P1, P2, P3],
                radius=self.radius,
                height=self.thickness
            )

            bow_tie += clear_edges(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                radius=self.radius,
                height=self.thickness
            )

            bow_tie += fillet_polygon(
                points=[P1, P2, P3],
                radius=self.radius,
                height=self.thickness
            )

            bow_tie += fillet_polygon(
                points=[P1 * mirror, P2 * mirror, P3 * mirror],
                radius=self.radius,
                height=self.thickness
            )

        return bow_tie

    # -----------------------------------------------------

    def bounding_box(self):

        return [
            2*self.amp + self.gap,
            2*self.amp*np.tan(np.deg2rad(30)),
            self.thickness
        ]