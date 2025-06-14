import numpy as np
import matplotlib.pyplot as plt

class Mobility:
    def __init__(self, temperature=300):
        self.temperature = temperature
        self.e_charge = 1.602176634e-19  # C
        self.kb = 1.380649e-23           # J/K
        self.amu2_to_cm2 = 0.279841e-16  # amu^2 to cm^2
        self.fs_to_sec = 1e-15

    def calculate_slope(self, t_range, msd_range):
        slope, intercept = np.polyfit(t_range, msd_range, 1)
        return slope, intercept

    def convert_slope_units(self, slope):
        return slope * self.amu2_to_cm2 / self.fs_to_sec

    def calculate_charge_mobility(self, slope_cm2_per_sec, n):
        return (self.e_charge * slope_cm2_per_sec) / (2 * n * self.kb * self.temperature)

    def plot_multi_file_msd(self, fnames, labels, colors, ns, start_t, end_t, save_filename=None):
        plt.figure(figsize=(10, 8))
        for fname, label, color, n in zip(fnames, labels, colors, ns):
            t, msd_x, msd_y, msd_z = np.loadtxt(fname, usecols=(0, 7, 8, 9), unpack=True)
            msd = np.sqrt(msd_x**2 + msd_y**2 + msd_z**2)
            mask = (t >= start_t) & (t <= end_t)
            t_range, msd_range = t[mask], msd[mask]

            slope, intercept = self.calculate_slope(t_range, msd_range)
            slope_cm2_per_sec = self.convert_slope_units(slope)
            mobility = self.calculate_charge_mobility(slope_cm2_per_sec, n)

            plt.plot(t_range, msd_range, marker='o', linestyle='-', color=color,
                    label=f'{label}', markersize=4, markevery=10, linewidth=2)

            print(f"{label} (n={n}):")
            print(f"  Slope (amu^2/fs): {slope}")
            print(f"  Slope (cm^2/sec): {slope_cm2_per_sec}")
            print(f"  Charge Mobility: {mobility} cm^2/(V·s)")

        plt.xlabel("Time (fs)", fontsize=12)
        plt.ylabel("MSD (amu^2)", fontsize=12)
        plt.legend(loc='upper left', fontsize=10, frameon=False)

        if save_filename:
            plt.savefig(save_filename, bbox_inches='tight', dpi=300)
            print(f"Image saved as {save_filename}")
        plt.show()

    def plot_single_file_msd(self, fname, start_t, end_t, n, save_filename=None):
        t, msd_x, msd_y, msd_z = np.loadtxt(fname, usecols=(0, 7, 8, 9), unpack=True)
        msd = np.sqrt(msd_x**2 + msd_y**2 + msd_z**2)
        mask = (t >= start_t) & (t <= end_t)
        t_range, msd_range = t[mask], msd[mask]

        slope, intercept = self.calculate_slope(t_range, msd_range)
        slope_cm2_per_sec = self.convert_slope_units(slope)
        mobility = self.calculate_charge_mobility(slope_cm2_per_sec, n)

        plt.figure(figsize=(6, 4))
        plt.plot(t_range, msd_range, 'o', label='Data Points')
        plt.plot(t_range, slope * t_range + intercept, '-', label='Fitted Line', color='black')
        plt.xlabel("Time (fs)", fontdict={'fontname': 'Comic Sans MS'}, fontsize=15)
        plt.ylabel("MSD (amu^2)", fontdict={'fontname': 'Comic Sans MS'}, fontsize=15)

        if save_filename:
            plt.savefig(save_filename, bbox_inches='tight', dpi=300)
            print(f"Image saved as {save_filename}")

        plt.grid(True)
        plt.legend()
        plt.show()

        print(f"Slope (amu^2/fs): {slope}")
        print(f"Slope (cm^2/sec): {slope_cm2_per_sec}")
        print(f"Charge Mobility: {mobility} cm^2/(V·s)")
