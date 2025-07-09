from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Cl": 35.453, "Br": 79.904, "I": 126.904
}


class CouplingVisualizer:
    def __init__(self, cpls_path: str, avg_gro_path: str):
        self.cpls_path = cpls_path
        self.avg_gro_path = avg_gro_path
        self.cpls_data = self.read_cpls()
        self.molecules, self.data, self.resids_plot1 = self.prepare_molecular_data()

    def read_cpls(self):
        cpls = {}
        with open(self.cpls_path, 'r') as f:
            next(f)
            for line in f:
                parts = line.split()
                resid = int(parts[0])
                coupling = float(parts[1])
                cpls[resid] = coupling
        return cpls

    def get_resid_atoms_from_avg_gro(self, lines, res_id):
        crds, lbls = [], []
        for line in lines[2:]:
            if len(line) < 44:
                continue
            try:
                current_resid = int(''.join(c for c in line[:5].rstrip() if c.isdigit()))
            except ValueError:
                continue
            if current_resid == res_id:
                atom_name = line[10:15].strip()
                element = atom_name[0].upper()
                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])
                lbls.append(element)
                crds.append([x, y, z])
        return np.array(crds, dtype=float), lbls

    def get_center_of_mass(self, pos, lbls):
        mass_arr = np.array([MASSES.get(lbl, 1.0) for lbl in lbls])[:, np.newaxis]
        return np.sum(pos * mass_arr, axis=0) / np.sum(mass_arr)

    def prepare_molecular_data(self):
        molecules = {}
        with open(self.avg_gro_path, 'r') as f:
            lines = f.readlines()
        N = len(self.cpls_data)
        data = np.zeros((N, 4))
        resids_plot1 = []
        for i, (res, coupling) in enumerate(self.cpls_data.items()):
            coords, lbls = self.get_resid_atoms_from_avg_gro(lines, res)
            com_arr = self.get_center_of_mass(coords, lbls)
            data[i, :3] = com_arr
            data[i, 3] = coupling
            resids_plot1.append(res)
            molecules[res] = {'avg_cpl': coupling, 'crds': coords, 'lbls': lbls, 'com': com_arr}
        return molecules, data, resids_plot1

    def Gaussian(self, r, A, mu, sigma):
        return A * np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))

    def plot_histogram_fit(self):
        avg_cpls = self.data[:, 3]
        plt.figure(figsize=(8, 6))
        counts, bins, _ = plt.hist(avg_cpls, bins=40, density=True, alpha=0.6, color='g', label='Histogram of Avg CPLs')
        bin_centers = (bins[:-1] + bins[1:]) / 2
        params, _ = curve_fit(self.Gaussian, bin_centers, counts, p0=[max(counts), np.mean(avg_cpls), np.std(avg_cpls)])
        x_fit = np.linspace(min(avg_cpls), max(avg_cpls), 1000)
        plt.plot(x_fit, self.Gaussian(x_fit, *params), color='red', label='Gaussian Fit')
        plt.legend()
        plt.xlabel("Coupling (meV)")
        plt.ylabel("Density")
        plt.title("Gaussian Fit of Average Coupling Values")
        plt.tight_layout()
        plt.show()

    def plot_interpolated_gradient_field(self, plane="xy", bins=(30, 30), cmap="inferno",
                                        annotate_all=False,
                                        annotate_resids_grain: Optional[List[int]] = None,
                                        annotate_resids_gb: Optional[List[int]] = None):
        x, y, z, energy = self.data[:, 0], self.data[:, 1], self.data[:, 2], self.data[:, 3]
        if plane == "xy":
            X, Y = x, y
            xlabel, ylabel = "X (nm)", "Y (nm)"
        elif plane == "xz":
            X, Y = x, z
            xlabel, ylabel = "X (nm)", "Z (nm)"
        elif plane == "yz":
            X, Y = y, z
            xlabel, ylabel = "Y (nm)", "Z (nm)"
        else:
            raise ValueError("Invalid plane")
        grid_x, grid_y = np.mgrid[
            X.min():X.max():complex(bins[0]),
            Y.min():Y.max():complex(bins[1])
        ]
        grid_energy = griddata((X, Y), energy, (grid_x, grid_y), method="linear")
        if np.isnan(grid_energy).any():
            fallback = griddata((X, Y), energy, (grid_x, grid_y), method="nearest")
            grid_energy[np.isnan(grid_energy)] = fallback[np.isnan(grid_energy)]
        dEdX, dEdY = np.gradient(grid_energy)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(grid_energy.T, extent=(X.min(), X.max(), Y.min(), Y.max()),
                        origin="lower", cmap=cmap, aspect="auto")
        plt.quiver(grid_x, grid_y, dEdX, dEdY, color='white', width=0.004)
        for i, resid in enumerate(self.resids_plot1):
            show = False
            color = 'black'
            if annotate_all or \
            (annotate_resids_grain and resid in annotate_resids_grain) or \
            (annotate_resids_gb and resid in annotate_resids_gb):
                show = True
                if annotate_resids_gb and resid in annotate_resids_gb:
                    color = 'yellow'
            if show:
                plt.annotate(str(resid), (X[i], Y[i]), xytext=(2, 2), textcoords="offset points",
                            fontsize=8, ha='center', color=color, fontweight='bold')
        cbar = plt.colorbar(im, label="Transfer Integrals")
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Transfer Integrals (meV)', fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.title(f"Gradient Field ({plane.upper()} Plane)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_local_gradient_vectors(self, plane="xy", bins=(35, 35), cmap="inferno", cutoff=0.6, scale=0.05,
                                    annotate_all=False,
                                    annotate_resids_grain: Optional[List[int]] = None,
                                    annotate_resids_gb: Optional[List[int]] = None):
        x, y, z, energy = self.data[:, 0], self.data[:, 1], self.data[:, 2], self.data[:, 3]
        positions = np.stack([x, y, z], axis=1)
        gradients = self.compute_site_gradients(positions, energy, cutoff)
        if plane == "xy":
            X, Y = x, y
            U, V = gradients[:, 0], gradients[:, 1]
            xlabel, ylabel = "X (nm)", "Y (nm)"
        elif plane == "xz":
            X, Y = x, z
            U, V = gradients[:, 0], gradients[:, 2]
            xlabel, ylabel = "X (nm)", "Z (nm)"
        elif plane == "yz":
            X, Y = y, z
            U, V = gradients[:, 1], gradients[:, 2]
            xlabel, ylabel = "Y (nm)", "Z (nm)"
        else:
            raise ValueError("Invalid plane.")
        plt.figure(figsize=(12, 10))
        sc = plt.scatter(X, Y, c=energy, cmap=cmap, s=400, edgecolors='k', linewidths=0.2)
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1/scale, color='black', width=0.003)
        for i, resid in enumerate(self.resids_plot1):
            show = False
            color = 'black'
            if annotate_all or \
            (annotate_resids_grain and resid in annotate_resids_grain) or \
            (annotate_resids_gb and resid in annotate_resids_gb):
                show = True
                if annotate_resids_gb and resid in annotate_resids_gb:
                    color = 'yellow'
            if show:
                plt.annotate(str(resid), (X[i], Y[i]), xytext=(2, 2), textcoords="offset points",
                            fontsize=8, ha='center', color=color, fontweight='bold')
        cbar = plt.colorbar(sc, label="Transfer Integrals")
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Transfer Integrals (meV)', fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.title(f"Transfer Integrals and Per-Site Gradient Vectors ({plane.upper()} Plane)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def compute_site_gradients(self, positions, energies, cutoff):
        tree = cKDTree(positions)
        gradients = np.zeros_like(positions)
        for i, pos in enumerate(positions):
            neighbors = tree.query_ball_point(pos, cutoff)
            neighbors = [j for j in neighbors if j != i]
            if not neighbors:
                continue
            grad = np.zeros(3)
            for j in neighbors:
                rij = positions[j] - pos
                dist = np.linalg.norm(rij)
                if dist > 1e-6:
                    direction = rij / dist
                    dE = energies[j] - energies[i]
                    grad += dE * direction
            gradients[i] = grad / len(neighbors)
        return gradients
