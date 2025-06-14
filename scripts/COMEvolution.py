import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class COMEvolution:
    AU_TO_NM = 0.0529177
    ATOMIC_MASSES = {"C": 12.011, "H": 1.008}

    def __init__(self, gro_file, qm_resids, xyz_ref, com_file, save_path=None, annotate_resids=None, annotate_label="GB Site", qm_color='red', charge_color='blue', title="Charge Evolution Over Time"):
        self.gro_file = gro_file
        self.qm_resids = self._normalize_resids(qm_resids)
        self.xyz_ref = xyz_ref
        self.com_file = com_file
        self.save_path = save_path
        self.annotate_resids = annotate_resids or []
        self.annotate_label = annotate_label
        self.qm_color = qm_color
        self.charge_color = charge_color
        self.title = title

        self.atom_data = self.read_gro_file()
        self.qm_atom_data, self.qm_positions = self.extract_qm_positions()
        self.atom_types = np.loadtxt(self.xyz_ref, dtype=str, usecols=0, skiprows=2)
        self.qm_com_positions = self.compute_qm_com()
        self.com_data = self.read_xvg()
        self.time_steps, self.com_x, self.com_y, self.com_z = self.extract_filtered_charge_positions()
        
    def _normalize_resids(self, resids):
        """
        Normalize input: accepts a list of integers or a string like '[1 34 100]'.
        Returns a list of integers.
        """
        if isinstance(resids, str):
            resids = resids.strip("[]")  # Remove brackets if present
            return [int(r) for r in resids.split()]
        elif isinstance(resids, (list, np.ndarray)):
            return list(map(int, resids))
        else:
            raise ValueError("qm_resids must be a list of integers or a string like '[1 2 3]'")


    def read_gro_file(self):
        with open(self.gro_file, "r") as f:
            lines = f.readlines()
        atom_data = []
        for line in lines[2:-1]:
            resid = int(line[:5].strip())
            atom_name = line[10:15].strip()
            x, y, z = map(float, [line[20:28], line[28:36], line[36:44]])
            atom_data.append((resid, atom_name, x, y, z))
        return atom_data

    def extract_qm_positions(self):
        qm_atom_data = [entry for entry in self.atom_data if entry[0] in self.qm_resids]
        qm_positions = np.array([[entry[2], entry[3], entry[4]] for entry in qm_atom_data])
        return qm_atom_data, qm_positions

    def compute_qm_com(self):
        num_qm_sites = len(self.qm_resids)
        qm_com_positions = np.zeros((num_qm_sites, 3))
        for i, resid in enumerate(self.qm_resids):
            atom_subset = [entry for entry in self.qm_atom_data if entry[0] == resid]
            atom_names = [entry[1] for entry in atom_subset]
            positions = np.array([[entry[2], entry[3], entry[4]] for entry in atom_subset])
            masses = np.array([self.ATOMIC_MASSES.get(atom, 12.011) for atom in atom_names])
            com = np.sum(positions * masses[:, np.newaxis], axis=0) / np.sum(masses)
            qm_com_positions[i] = com
        return qm_com_positions

    def read_xvg(self):
        data = []
        with open(self.com_file, "r") as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                cols = line.split()
                time = float(cols[0])
                charge_x = float(cols[1]) * self.AU_TO_NM
                charge_y = float(cols[2]) * self.AU_TO_NM
                charge_z = float(cols[3]) * self.AU_TO_NM
                data.append([time, charge_x, charge_y, charge_z])
        return np.array(data, dtype=float)

    def extract_filtered_charge_positions(self):
        time_steps = self.com_data[:, 0]
        com_x, com_y, com_z = self.com_data[:, 1], self.com_data[:, 2], self.com_data[:, 3]
        mask = (time_steps % 5 == 0)
        return time_steps[mask], com_x[mask], com_y[mask], com_z[mask]

    def animate(self):
        x, y = self.qm_com_positions[:, 0], self.qm_com_positions[:, 1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, c=self.qm_color, marker='o', s=30, label="QM sites")
        ax.tick_params(axis='both', which='major', labelsize=14)

        gb_indices = [self.qm_resids.index(resid) for resid in self.annotate_resids if resid in self.qm_resids]
        for idx in gb_indices:
            ax.annotate(self.annotate_label, (x[idx], y[idx]), textcoords="offset points",
                        xytext=(5, 5), ha='center', fontsize=10, color=self.charge_color, fontweight='bold')

        charge_marker, = ax.plot([], [], 'x', color=self.charge_color, markersize=8, label='Charge')
        charge_path, = ax.plot([], [], '--', color=self.charge_color, alpha=0.7)

        ax.set_xlabel("X (nm)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Y (nm)", fontsize=14, fontweight='bold')
        ax.set_title(self.title)
        ax.legend(fontsize=10, frameon=False)

        def update(frame):
            charge_marker.set_data([self.com_x[frame]], [self.com_y[frame]])
            charge_path.set_data(self.com_x[:frame + 1], self.com_y[:frame + 1])
            return charge_marker, charge_path

        ani = animation.FuncAnimation(fig, update, frames=range(len(self.com_x)), interval=50, blit=True)

        if self.save_path:
            ani.save(self.save_path, writer="ffmpeg", fps=20)

        plt.show()
