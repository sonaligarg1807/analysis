import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class TBAverager:
    def __init__(self, base_dir, output_path, input_filename, traj_dirs=None, max_workers=16):
        """
        Parameters:
        - base_dir (str): Path containing trajectory directories (e.g., TRAJ0, TRAJ1, ...)
        - output_path (str): Path to save the averaged data
        - input_filename (str): Name of the input file inside each trajectory folder
        - traj_dirs (list, optional): List of trajectory directory names. If None, autodetects TRAJ*.
        - max_workers (int): Number of parallel threads to use
        """
        self.base_dir = base_dir
        self.output_path = output_path
        self.input_filename = input_filename
        self.max_workers = max_workers

        if traj_dirs is None:
            self.traj_dirs = sorted([
                f for f in os.listdir(base_dir)
                if f.startswith('TRAJ') and os.path.isdir(os.path.join(base_dir, f))
            ])
        else:
            self.traj_dirs = traj_dirs

        if not self.traj_dirs:
            raise ValueError(f"No valid trajectory directories found in: {base_dir}")

    def _load_data(self, traj_dir):
        file_path = os.path.join(self.base_dir, traj_dir, self.input_filename)
        try:
            data = np.loadtxt(file_path, comments="#")
            return data
        except Exception as e:
            print(f"[WARN] Skipped {file_path}: {e}")
            return None

    def average(self):
        print(f"[INFO] Starting averaging over {len(self.traj_dirs)} trajectories...")

        all_data = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._load_data, td): td for td in self.traj_dirs}
            for future in as_completed(futures):
                data = future.result()
                if data is not None:
                    all_data.append(data)

        if not all_data:
            raise RuntimeError("No valid data loaded from any trajectory directories.")

        num_columns = all_data[0].shape[1]
        max_rows = max(data.shape[0] for data in all_data)

        sum_data = np.zeros((max_rows, num_columns))
        count_data = np.zeros((max_rows, 1))

        for data in all_data:
            rows = data.shape[0]
            if data.shape[1] != num_columns:
                raise ValueError(f"Inconsistent column count detected in one of the files.")
            sum_data[:rows] += data
            count_data[:rows] += 1

        mean_data = sum_data / count_data

        np.savetxt(self.output_path, mean_data, fmt='%.5f', delimiter='      ', comments=' ')
        print(f"[SUCCESS] Averaged data saved to {self.output_path}")
