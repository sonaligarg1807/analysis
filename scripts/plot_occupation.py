import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

class PlotOccupation:
    def __init__(self, significance_threshold=0.05, edge_site_count=3):
        self.significance_threshold = significance_threshold
        self.edge_site_count = edge_site_count

    def averaged_occupation(self, file_path, selected_sites=None, time_skip=20):
        """
        Plots the averaged occupation over time for selected sites.
        """
        plt.figure(figsize=(8, 8))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        try:
            data = np.loadtxt(file_path)
            time_steps = data[:, 0]
            site_occupations = data[:, 1:]

            time_steps = time_steps[::time_skip]
            site_occupations = site_occupations[::time_skip, :]

            if selected_sites is None:
                selected_sites = list(range(site_occupations.shape[1]))

            for site_idx in selected_sites:
                if site_idx < site_occupations.shape[1]:
                    plt.plot(time_steps, site_occupations[:, site_idx], 
                            label=f'Site {site_idx + 1}', linewidth=1.5, alpha=0.8)

        except FileNotFoundError:
            print(f"File '{file_path}' not found. Please check the file path and try again.")
            return
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            return

        plt.xlabel('Time (fs)', fontsize=14, fontweight='bold')
        plt.ylabel('Occupation', fontsize=14, fontweight='bold')
        plt.title('Occupation vs. Time')
        plt.legend(fontsize=10, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def individual_occupation(self, parent_dir, output_file="tb_occupation_stats.dat",
                            sufficiency_file="qm_zone_check.dat", plot_last_site=True):
        """
        Analyzes occupation files in trajectory directories and evaluates sufficiency.
        Optionally plots the last site occupation across trajectories.
        """
        def process_xvg_file(filepath):
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith(('#', '@')):
                        continue
                    try:
                        values = list(map(float, line.split()))
                        data.append(values)
                    except ValueError:
                        continue
            if not data:
                return None
            data_array = np.array(data, dtype=np.float64)
            return np.mean(data_array[:, 1:], axis=0)

        def worker_task(args):
            subdir_name, xvg_path = args
            if os.path.exists(xvg_path):
                averages = process_xvg_file(xvg_path)
                if averages is not None:
                    return (subdir_name, averages)
            return None

        tasks = []
        for entry in sorted(os.scandir(parent_dir), key=lambda e: e.name):
            if entry.is_dir() and entry.name.startswith('subdir_'):
                xvg_path = os.path.join(entry.path, 'TRAJ1', 'TB_OCCUPATION.xvg')
                tasks.append((entry.name, xvg_path))

        print(f"Found {len(tasks)} subdirectories with potential data.")

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(worker_task, tasks)

        results = [res for res in results if res is not None]
        results.sort(key=lambda x: x[0])

        last_site_values = []

        with open(output_file, 'w') as out_f, open(sufficiency_file, 'w') as suff_f:
            if results:
                max_cols = max(len(avg) for _, avg in results)
                header = f"{'Subdirectory':<20}" + "".join([f"{'Col'+str(i):<10}" for i in range(2, max_cols + 2)])
                out_f.write(header + "\n")

                suff_f.write(f"{'Subdirectory':<20} {'TotalOcc':<10} {'SigSites':<10} {'EdgeSigSites':<15} {'Sufficient':<10}\n")

                for subdir, averages in results:
                    site_occupations = averages[:-1]
                    num_sites = len(site_occupations)
                    total_occupation = np.sum(site_occupations)
                    threshold = self.significance_threshold * total_occupation

                    significant_sites = [i for i, val in enumerate(site_occupations) if val > threshold]
                    edge_sites = list(range(self.edge_site_count)) + list(range(num_sites - self.edge_site_count, num_sites))
                    edge_significant = [i for i in significant_sites if i in edge_sites]
                    is_sufficient = "YES" if len(edge_significant) == 0 else "NO"

                    col_format = "{:<20}" + "{:<10.4f}" * len(averages)
                    out_f.write(col_format.format(subdir, *averages) + "\n")

                    suff_f.write(f"{subdir:<20} {total_occupation:<10.4f} {len(significant_sites):<10} {len(edge_significant):<15} {is_sufficient:<10}\n")

                    if len(site_occupations) >= 2:
                        last_site_values.append(site_occupations[-2])

                print(f"Analysis complete. Results saved to:\n  - {output_file}\n  - {sufficiency_file}")

                if plot_last_site:
                    plt.figure(figsize=(12, 6))
                    plt.scatter(range(len(last_site_values)), last_site_values, alpha=0.7, s=30, c='purple')
                    plt.xlabel("Trajectory Index")
                    plt.ylabel("Occupation of Last Site")
                    plt.title("Occupation of Last Site Across Trajectories")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
            else:
                print("No valid results found to write.")
