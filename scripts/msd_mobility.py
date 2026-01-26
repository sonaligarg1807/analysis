import numpy as np
import matplotlib.pyplot as plt


class Mobility:
    """
    Mobility from TB_WFPROPS.xvg using Einstein relation.

    TB_WFPROPS.xvg (as you described):
      col 1: time (fs)
      col 5-7 : MSD per dimension from [R_COC(t)-R_COC(0)]^2
      col 8-10: population-weighted MSD per dimension sum_i rho_i(t)[R_i(t)-R_COC(0)]^2   (recommended)

    Correct multi-dim MSD is the SUM of chosen components, NOT sqrt(sum(component^2)).

    Flexible options:
      - msd_block: "alt" (cols 8-10) or "coc" (cols 5-7)
      - dims: "x", "y", "z", "xy", "xz", "yz", "xyz"
      - dims: "custom"  -> MSD = MSDx + MSDy + MSDz, but forces n=1 (your requested legacy/custom mode)
      - n: if None, inferred from dims (x->1, xy->2, xyz->3). For "custom", n is forced to 1.
    """

    def __init__(self, temperature=300.0, length_unit="bohr"):
        self.temperature = float(temperature)

        # constants for Einstein relation
        self.e_charge = 1.602176634e-19  # C
        self.kb = 1.380649e-23           # J/K
        self.fs_to_sec = 1e-15

        # TB_WFPROPS positions are Bohr (per your doc), so MSD is Bohr^2.
        bohr_to_cm = 0.529177210903e-8
        self.bohr2_to_cm2 = bohr_to_cm ** 2

        # Set MSD unit conversion
        if length_unit.lower() in ["bohr", "au", "a.u.", "atomic_units"]:
            self.msd2_to_cm2 = self.bohr2_to_cm2
            self.length_unit_label = "Bohr"
        elif length_unit.lower() in ["cm"]:
            self.msd2_to_cm2 = 1.0
            self.length_unit_label = "cm"
        else:
            raise ValueError(
                "Unsupported length_unit. Use 'bohr' (recommended) or 'cm'. "
                "If your MSD columns are in another unit, add its conversion here."
            )

    # ---------- Core helpers ----------
    @staticmethod
    def _read_xvg_cols(fname, usecols):
        """Robust XVG reader: ignores lines starting with @ or #."""
        data = []
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("@") or line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    row = [float(parts[i]) for i in usecols]
                except Exception:
                    continue
                data.append(row)
        if len(data) == 0:
            raise ValueError(f"No numeric data read from: {fname}")
        arr = np.array(data, dtype=float)
        return [arr[:, j] for j in range(arr.shape[1])]

    @staticmethod
    def calculate_slope(t_range, y_range):
        slope, intercept = np.polyfit(t_range, y_range, 1)
        return slope, intercept

    def convert_slope_units_to_cm2_per_s(self, slope_length2_per_fs):
        """Convert slope from (length^2/fs) to (cm^2/s)."""
        return slope_length2_per_fs * self.msd2_to_cm2 / self.fs_to_sec

    def calculate_charge_mobility(self, D_cm2_per_s):
        """μ = e D / (kB T). Returns μ in cm^2/(V s) if D in cm^2/s."""
        return (self.e_charge * D_cm2_per_s) / (self.kb * self.temperature)

    # ---------- MSD construction ----------
    @staticmethod
    def infer_n_from_dims(dims):
        dims = dims.lower()
        if dims == "custom":
            return 1
        if dims in ["x", "y", "z"]:
            return 1
        if dims in ["xy", "xz", "yz"]:
            return 2
        if dims == "xyz":
            return 3
        raise ValueError(f"Unsupported dims='{dims}'. Use x,y,z,xy,xz,yz,xyz,custom.")

    @staticmethod
    def dims_to_indices(dims):
        """Map dims to indices in [msd_x, msd_y, msd_z]."""
        dims = dims.lower()
        if dims == "custom":
            return [0, 1, 2]
        mapping = {"x": 0, "y": 1, "z": 2}
        if dims in mapping:
            return [mapping[dims]]
        if dims == "xy":
            return [0, 1]
        if dims == "xz":
            return [0, 2]
        if dims == "yz":
            return [1, 2]
        if dims == "xyz":
            return [0, 1, 2]
        raise ValueError(f"Unsupported dims='{dims}'.")

    def load_time_and_msd_components(self, fname, msd_block="alt"):
        """
        Returns:
          t (fs), msd_xyz (Nt,3) where each column is MSD per dimension (length^2)

        msd_block:
          - "alt": columns 8-10 in 1-indexed -> usecols (0,7,8,9) in 0-indexed
          - "coc": columns 5-7 in 1-indexed -> usecols (0,4,5,6) in 0-indexed
        """
        msd_block = msd_block.lower()
        if msd_block == "alt":
            t, msd_x, msd_y, msd_z = self._read_xvg_cols(fname, usecols=[0, 7, 8, 9])
        elif msd_block == "coc":
            t, msd_x, msd_y, msd_z = self._read_xvg_cols(fname, usecols=[0, 4, 5, 6])
        else:
            raise ValueError("msd_block must be 'alt' or 'coc'.")

        msd_xyz = np.column_stack([msd_x, msd_y, msd_z])
        return t, msd_xyz

    def build_msd(self, msd_xyz, dims="xyz"):
        """
        Correct MSD in chosen dims: SUM of per-dimension MSD components.
        For dims='custom': MSD = MSDx+MSDy+MSDz (same as xyz), but n will be forced to 1 elsewhere.
        """
        idx = self.dims_to_indices(dims)
        return msd_xyz[:, idx].sum(axis=1)

    # ---------- High-level API ----------
    def fit_D_and_mu_from_file(
        self,
        fname,
        start_t,
        end_t,
        dims="x",
        n=None,
        msd_block="alt",
        return_fit=False
    ):
        """
        Load file -> build MSD -> fit slope in [start_t,end_t] -> compute D and μ.

        For dims='custom':
          - MSD is MSDx+MSDy+MSDz
          - n is forced to 1 (overrides provided n)
        """
        t, msd_xyz = self.load_time_and_msd_components(fname, msd_block=msd_block)
        msd = self.build_msd(msd_xyz, dims=dims)

        mask = (t >= start_t) & (t <= end_t)
        t_range = t[mask]
        msd_range = msd[mask]
        if t_range.size < 5:
            raise ValueError("Fit window too small or no points in range. Check start_t/end_t.")

        slope, intercept = self.calculate_slope(t_range, msd_range)

        # Determine n
        dims_l = dims.lower()
        if dims_l == "custom":
            n_eff = 1
        else:
            n_eff = self.infer_n_from_dims(dims) if n is None else int(n)

        # Einstein: MSD = 2*n*D*t  => D = slope / (2*n)
        D_length2_per_fs = slope / (2.0 * n_eff)

        # Convert to cm^2/s
        D_cm2_per_s = self.convert_slope_units_to_cm2_per_s(D_length2_per_fs)

        # Mobility
        mu = self.calculate_charge_mobility(D_cm2_per_s)

        out = {
            "fname": str(fname),
            "dims": dims,
            "n": n_eff,
            "msd_block": msd_block,
            "slope_MSD_length2_per_fs": slope,
            "intercept_length2": intercept,
            "D_length2_per_fs": D_length2_per_fs,
            "D_cm2_per_s": D_cm2_per_s,
            "mu_cm2_per_Vs": mu,
        }

        if return_fit:
            out.update({"t": t, "msd": msd, "t_range": t_range, "msd_range": msd_range})
        return out

    # ---------- Plotters ----------
    def plot_single_file_msd(
        self,
        fname,
        start_t,
        end_t,
        dims="x",
        n=None,
        msd_block="alt",
        save_filename=None,
        markevery=10
    ):
        res = self.fit_D_and_mu_from_file(
            fname=fname, start_t=start_t, end_t=end_t,
            dims=dims, n=n, msd_block=msd_block, return_fit=True
        )

        t_range = res["t_range"]
        msd_range = res["msd_range"]
        slope = res["slope_MSD_length2_per_fs"]
        intercept = res["intercept_length2"]

        plt.figure(figsize=(6, 4))
        plt.plot(t_range, msd_range, "o", label="Data", markersize=4, markevery=markevery)
        plt.plot(t_range, slope * t_range + intercept, "-", color="black", label="Linear fit")

        plt.xlabel("Time (fs)", fontsize=12)
        plt.ylabel(f"MSD({dims}) [{self.length_unit_label}^2]", fontsize=12)
        plt.grid(True)
        plt.legend()

        if save_filename:
            plt.savefig(save_filename, bbox_inches="tight", dpi=300)
            print(f"Saved: {save_filename}")
        plt.show()

        print(f"File: {fname}")
        print(f"MSD block: {msd_block}  (alt=cols 8-10, coc=cols 5-7)")
        print(f"Dims: {dims}  -> n={res['n']}")
        print(f"Slope d(MSD)/dt [{self.length_unit_label}^2/fs]: {res['slope_MSD_length2_per_fs']:.6g}")
        print(f"D [{self.length_unit_label}^2/fs]: {res['D_length2_per_fs']:.6g}")
        print(f"D [cm^2/s]: {res['D_cm2_per_s']:.6g}")
        print(f"Mobility μ [cm^2/(V·s)]: {res['mu_cm2_per_Vs']:.6g}")

        return res

    def plot_multi_file_msd(
        self,
        fnames,
        labels,
        colors,
        start_t,
        end_t,
        dims="x",
        ns=None,
        msd_block="alt",
        save_filename=None,
        markevery=10
    ):
        """
        Plot multiple MSD curves and print mobility for each.

        If ns is None, n is inferred from dims for all (except custom forces n=1).
        If ns is provided, it must be a list same length as fnames (allows overriding per curve),
        but note: dims='custom' will still force n=1.
        """
        if ns is not None and len(ns) != len(fnames):
            raise ValueError("ns must be None or the same length as fnames.")

        plt.figure(figsize=(10, 6))

        results = []
        for i, (fname, label, color) in enumerate(zip(fnames, labels, colors)):
            n_i = None if ns is None else ns[i]
            res = self.fit_D_and_mu_from_file(
                fname=fname, start_t=start_t, end_t=end_t,
                dims=dims, n=n_i, msd_block=msd_block, return_fit=True
            )
            results.append(res)

            t_range = res["t_range"]
            msd_range = res["msd_range"]

            plt.plot(
                t_range, msd_range,
                marker="o", linestyle="-", color=color,
                label=label, markersize=4, markevery=markevery, linewidth=2
            )

            print(f"{label}:")
            print(f"  dims={dims}, n={res['n']}, msd_block={msd_block}")
            print(f"  slope d(MSD)/dt [{self.length_unit_label}^2/fs]: {res['slope_MSD_length2_per_fs']:.6g}")
            print(f"  D [cm^2/s]: {res['D_cm2_per_s']:.6g}")
            print(f"  μ [cm^2/(V·s)]: {res['mu_cm2_per_Vs']:.6g}")

        plt.xlabel("Time (fs)", fontsize=12)
        plt.ylabel(f"MSD({dims}) [{self.length_unit_label}^2]", fontsize=12)
        plt.legend(loc="upper left", fontsize=10, frameon=False)
        plt.grid(True)

        if save_filename:
            plt.savefig(save_filename, bbox_inches="tight", dpi=300)
            print(f"Saved: {save_filename}")
        plt.show()

        return results
