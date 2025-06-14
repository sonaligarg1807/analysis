import numpy as np

class COMDistanceCalculator:
    # Atomic masses of pentacene atoms (C = 12.011, H = 1.008)
    atom_masses = {
        'C': 12.011,
        'H': 1.008
    }

    nm_to_angstrom = 10.0

    def __init__(self, gro_file, resid1, resid2, output_filename):
        self.gro_file = gro_file
        self.resid1 = resid1
        self.resid2 = resid2
        self.output_filename = output_filename
        self.atoms = self.parse_gro_file()

    def parse_gro_file(self):
        atoms = []
        with open(self.gro_file, 'r') as file:
            lines = file.readlines()[2:-1]  # Skip header and box line
            for line in lines:
                resid = int(line[0:5].strip())
                atomname = line[10:15].strip()
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())
                atoms.append((resid, atomname, np.array([x, y, z])))
        return atoms

    def get_molecule_atoms(self, resid):
        return [atom for atom in self.atoms if atom[0] == resid]

    def calculate_com(self, atoms):
        total_mass = 0.0
        weighted_positions = np.zeros(3)

        for atom in atoms:
            atomname = atom[1][0]  # First character: 'C' or 'H'
            mass = self.atom_masses.get(atomname, 0)
            weighted_positions += mass * atom[2]
            total_mass += mass

        return weighted_positions / total_mass

    def calculate_distance(self, com1, com2):
        return np.linalg.norm(com1 - com2)

    def save_to_file(self, com1, com2, dist_nm, dist_angstrom):
        with open(self.output_filename, 'w') as f:
            f.write(f"Center of Mass of Molecule {self.resid1}: {com1}\n")
            f.write(f"Center of Mass of Molecule {self.resid2}: {com2}\n")
            f.write(f"Distance between COMs (nm): {dist_nm:.3f} nm\n")
            f.write(f"Distance between COMs (angstroms): {dist_angstrom:.3f} Å\n")
        print(f"Results saved to {self.output_filename}")

    def run(self):
        mol1_atoms = self.get_molecule_atoms(self.resid1)
        mol2_atoms = self.get_molecule_atoms(self.resid2)

        com1 = self.calculate_com(mol1_atoms)
        com2 = self.calculate_com(mol2_atoms)

        dist_nm = self.calculate_distance(com1, com2)
        dist_angstrom = dist_nm * self.nm_to_angstrom

        print(f"Center of Mass of Molecule {self.resid1}: {com1}")
        print(f"Center of Mass of Molecule {self.resid2}: {com2}")
        print(f"Distance between COMs: {dist_nm:.3f} nm")
        print(f"Distance between COMs: {dist_angstrom:.3f} Å")

        self.save_to_file(com1, com2, dist_nm, dist_angstrom)
