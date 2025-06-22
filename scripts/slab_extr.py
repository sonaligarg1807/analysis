import numpy as np
import os

class SlabExtractor:
    def __init__(self, gro_path, natoms, vel=True):
        self.gro_path = gro_path
        self.natoms = natoms
        self.vel = vel
        self.lines = self.read_gro()

    @staticmethod
    def atomic_mass(lbl):
        mass = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "S": 32.065,
        }
        return mass[lbl]

    @staticmethod
    def get_com(mol, vel=True):
        mr_tot = np.zeros(3)
        m_tot = 0.0
        for line in mol:
            l = line.split()
            m = SlabExtractor.atomic_mass(l[1][0])
            pos = l[-6:-3] if vel else l[-3:]
            mr_tot += np.array([float(i) * m for i in pos])
            m_tot += m
        return mr_tot / m_tot

    def read_gro(self):
        with open(self.gro_path, 'r') as f:
            lines = f.readlines()
        return lines[2:-1]

    def select_slab_single_direction(self, lim):  # lim = [a, b, "z"]
        mol = []
        slab = []

        axis_index = {"x": 0, "y": 1, "z": 2}[lim[2]]

        for line in self.lines:
            mol.append(line)
            if len(mol) == self.natoms:
                com = self.get_com(mol, self.vel)
                if lim[0] <= com[axis_index] <= lim[1]:
                    slab.append(mol)
                mol = []

        return slab

    def select_slab_multi_direction(self, lim):  # lim = {"x": (a,b), "y": (a,b), "z": (a,b)}
        mol = []
        slab = []

        for line in self.lines:
            mol.append(line)
            if len(mol) == self.natoms:
                com = self.get_com(mol, self.vel)
                within_limits = True
                for axis, (low, high) in lim.items():
                    idx = {"x": 0, "y": 1, "z": 2}[axis]
                    if not (low <= com[idx] <= high):
                        within_limits = False
                        break
                if within_limits:
                    slab.append(mol)
                mol = []

        return slab

    def write_xyz(self, slab, outname):
        with open(outname, 'w') as f:
            f.write(f"{len(slab) * self.natoms}\n\n")
            for mol in slab:
                for line in mol:
                    l = line.split()
                    if self.vel:
                        coords = [float(l[-6]), float(l[-5]), float(l[-4])]
                    else:
                        coords = [float(l[-3]), float(l[-2]), float(l[-1])]
                    f.write(f"{l[1][0]}\t {coords[0]*10.:10.4f}\t {coords[1]*10.:10.4f}\t {coords[2]*10.:10.4f}\n")
