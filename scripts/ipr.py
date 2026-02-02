import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class IPR:
    def __init__(self, filename):
        self.filename = filename
        self.data = np.loadtxt(filename)
        self.time = self.data[:, 0]
        self.ipr_values = self.data[:, 1]

    def averaged_ipr(
        self,
        output_filename=None,
        legend_name=None,
        time_range=None,
        plot_color='blue',
        title='Time vs. IPR',
        xlabel='Time (fs)',
        ylabel='IPR',
        font='DejaVu Sans',
        fontsize=15
    ):
        """
        Plot IPR vs. time and calculate average IPR.
        If time_range=(start,end) is provided, BOTH the plot and the average
        are restricted to that range.
        """

        # --- Select data to plot/average ---
        if time_range is not None:
            start_time, end_time = time_range
            mask = (self.time >= start_time) & (self.time <= end_time)

            time_plot = self.time[mask]
            ipr_plot = self.ipr_values[mask]

            if time_plot.size == 0:
                raise ValueError(
                    f"No data points found in time_range={time_range}. "
                    f"Available time range: {self.time.min()} to {self.time.max()} fs"
                )
        else:
            time_plot = self.time
            ipr_plot = self.ipr_values

        average_ipr = np.mean(ipr_plot)

        plt.figure(figsize=(5, 3))
        label = f"{legend_name or 'IPR'} (Avg: {average_ipr:.4f})"
        plt.plot(time_plot, ipr_plot, color=plot_color, linestyle='-', label=label)

        plt.title(title, fontdict={'fontname': font}, fontsize=fontsize)
        plt.xlabel(xlabel, fontdict={'fontname': font}, fontsize=fontsize)
        plt.ylabel(ylabel, fontdict={'fontname': font}, fontsize=fontsize)
        plt.grid(True)
        plt.legend(fontsize=fontsize - 5, frameon=False)

        if output_filename:
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.show()

        if time_range is not None:
            print(f"Average IPR in range {start_time}-{end_time} fs: {average_ipr:.4f}")
        else:
            print(f"Average IPR (entire range): {average_ipr:.4f}")

        return average_ipr

    def animation_ipr(
        self,
        output_filename,
        fps=30,
        step_fs=None,
        time_range=None,
        plot_color='blue',
        title='Time vs. Charge Delocalisation',
        xlabel='Time (fs)',
        ylabel='Delocalisation',
        fontweight='bold',
        fontsize=14
    ):
        """
        Create an animated line plot of IPR vs. time.

        - step_fs: sample every step_fs femtoseconds (optional).
        - time_range: (start,end) in fs to restrict BOTH animation and average.
        """

        # ---- Apply time range first (so step_fs acts inside the chosen window) ----
        if time_range is not None:
            start_time, end_time = time_range
            mask = (self.time >= start_time) & (self.time <= end_time)
            time_base = self.time[mask]
            ipr_base = self.ipr_values[mask]

            if time_base.size == 0:
                raise ValueError(
                    f"No data points found in time_range={time_range}. "
                    f"Available time range: {self.time.min()} to {self.time.max()} fs"
                )
        else:
            time_base = self.time
            ipr_base = self.ipr_values

        # ---- Determine sampling step in indices ----
        if step_fs is None:
            step_size = 1
        else:
            if time_base.size < 2:
                raise ValueError("Not enough points in selected range to determine timestep for step_fs.")
            dt = time_base[1] - time_base[0]
            if dt <= 0:
                raise ValueError("Time column must be strictly increasing for step_fs sampling.")
            step_size = int(np.round(step_fs / dt))
            step_size = max(step_size, 1)

        time_sel = time_base[::step_size]
        ipr_sel = ipr_base[::step_size]

        if time_sel.size == 0:
            raise ValueError("After applying step_fs/time_range, no points remain to animate.")

        avg_ipr = np.mean(ipr_sel)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title, fontsize=fontsize, fontweight=fontweight)
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        line, = ax.plot([], [], color=plot_color, linestyle='-', label=f"IPR (Avg: {avg_ipr:.2f})")
        ax.legend(fontsize=fontsize - 2, frameon=False)

        ax.set_xlim(time_sel.min(), time_sel.max())

        # Add a small padding so the line isn't glued to the borders
        y_min, y_max = float(np.min(ipr_sel)), float(np.max(ipr_sel))
        if np.isclose(y_min, y_max):
            pad = 0.5 if y_min == 0 else 0.05 * abs(y_min)
        else:
            pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

        def update(frame):
            line.set_data(time_sel[:frame + 1], ipr_sel[:frame + 1])
            return (line,)

        interval_ms = 1000 / fps
        ani = animation.FuncAnimation(
            fig, update,
            frames=len(time_sel),
            interval=interval_ms,
            blit=True
        )

        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.show()

        if time_range is not None:
            print(f"Animated IPR range {start_time}-{end_time} fs | Avg over animated points: {avg_ipr:.4f}")
        else:
            print(f"Animated full range | Avg over animated points: {avg_ipr:.4f}")
