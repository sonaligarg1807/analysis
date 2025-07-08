#functionality of reading both .gro and xyz files

import numpy as np

class COMDistanceCalculator:
    atom_masses = {
        'C': 12.011,
        'H': 1.008
    }

    nm_to_angstrom = 10.0

    def __init__(self, input_file, mode, mol1, mol2):
        self.input_file = input_file
        self.mode = mode
        self.mol1 = mol1
        self.mol2 = mol2

        if mode == 'gro':
            self.atoms = self.parse_gro_file()
        elif mode == 'xyz':
            self.atoms = self.parse_xyz_file()
        else:
            raise ValueError("Mode must be 'gro' or 'xyz'")

    def parse_gro_file(self):
        atoms = []
        with open(self.input_file, 'r') as file:
            lines = file.readlines()[2:-1]
            for line in lines:
                resid = int(line[0:5].strip())
                atomname = line[10:15].strip()
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())
                atoms.append((resid, atomname, np.array([x, y, z])))  # in nm
        return atoms

    def parse_xyz_file(self):
        atoms = []
        with open(self.input_file, 'r') as file:
            lines = file.readlines()[2:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                atomname = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append((atomname, np.array([x, y, z])))  # in Å
        return atoms

    def get_molecule_atoms(self, identifier):
        if self.mode == 'gro':
            resid = identifier
            return [atom for atom in self.atoms if atom[0] == resid]
        elif self.mode == 'xyz':
            start, end = identifier
            return self.atoms[start - 3 : end - 2]  # adjust for header
        else:
            raise ValueError("Invalid mode")

    def calculate_com(self, atoms):
        total_mass = 0.0
        weighted_positions = np.zeros(3)

        for atom in atoms:
            if self.mode == 'gro':
                atomname = atom[1][0]
                pos = atom[2]
            else:
                atomname = atom[0][0]
                pos = atom[1]

            mass = self.atom_masses.get(atomname, 12.011)
            weighted_positions += mass * pos
            total_mass += mass

        return weighted_positions / total_mass

    def calculate_distance(self, com1, com2):
        return np.linalg.norm(com1 - com2)

    def run(self):
        mol1_atoms = self.get_molecule_atoms(self.mol1)
        mol2_atoms = self.get_molecule_atoms(self.mol2)

        com1 = self.calculate_com(mol1_atoms)
        com2 = self.calculate_com(mol2_atoms)
        dist_angstrom = self.calculate_distance(com1, com2)

        if self.mode == 'gro':
            dist_nm = dist_angstrom
            dist_angstrom *= self.nm_to_angstrom
        else:
            dist_nm = dist_angstrom / self.nm_to_angstrom

        print(f"COM Molecule 1: {com1}")
        print(f"COM Molecule 2: {com2}")
        print(f"Distance between COMs: {dist_nm:.3f} nm | {dist_angstrom:.3f} Å")
