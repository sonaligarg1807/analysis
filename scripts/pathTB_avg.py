import os
import numpy as np

class PathTBAverager:
    def __init__(self, base_dir, output_path, input_filename):
        """
        Parameters:
        - base_dir (str): Root directory containing 'subdir_*' folders, each with 'TRAJ*' folders.
        - output_path (str): File path to save the averaged data.
        - input_filename (str): Name of the file to be averaged within each TRAJ directory.
        """
        self.base_dir = base_dir
        self.output_path = output_path
        self.input_filename = input_filename

    def average(self):
        sum_data = None
        traj_count = 0

        result_dirs = [
            d for d in os.listdir(self.base_dir)
            if d.startswith("subdir_") and os.path.isdir(os.path.join(self.base_dir, d))
        ]

        print(f"[INFO] Found {len(result_dirs)} subdirectories to process.")

        for result_dir in result_dirs:
            result_path = os.path.join(self.base_dir, result_dir)

            traj_dirs = [
                d for d in os.listdir(result_path)
                if d.startswith("TRAJ") and os.path.isdir(os.path.join(result_path, d))
            ]

            for traj_dir in traj_dirs:
                file_path = os.path.join(result_path, traj_dir, self.input_filename)

                if not os.path.isfile(file_path):
                    print(f"[WARN] Missing file: {file_path}")
                    continue

                try:
                    data = np.loadtxt(file_path, comments="#")

                    if sum_data is None:
                        sum_data = np.zeros_like(data)

                    if data.shape != sum_data.shape:
                        raise ValueError(f"Data shape mismatch in {file_path}")

                    sum_data += data
                    traj_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")

        if traj_count == 0:
            raise RuntimeError("No valid TRAJ directories processed. Check your inputs.")

        mean_data = sum_data / traj_count

        np.savetxt(self.output_path, mean_data, fmt="%.5f", delimiter="      ", comments=" ")
        print(f"[SUCCESS] Averaged data saved to {self.output_path}")
