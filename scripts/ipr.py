import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class IPR:
    def __init__(self, filename):
        self.filename = filename
        self.data = np.loadtxt(filename)
        self.time = self.data[:, 0]
        self.ipr_values = self.data[:, 1]

    def averaged_ipr(self, output_filename=None, legend_name=None, time_range=None, plot_color='blue', title='Time vs. IPR', 
                xlabel='Time (fs)', ylabel='IPR', font='Comic Sans MS', fontsize=15):
        """
        Plot IPR vs. time and calculate average IPR.
        Allows time_range filtering and custom styling.
        """
        # Calculate average IPR first
        if time_range:
            start_time, end_time = time_range
            indices = np.where((self.time >= start_time) & (self.time <= end_time))
            ipr_range = self.ipr_values[indices]
            average_ipr = np.mean(ipr_range)
        else:
            average_ipr = np.mean(self.ipr_values)
        
        plt.figure(figsize=(5, 3))
        # Add average to legend
        label = f"{legend_name or 'IPR'} (Avg: {average_ipr:.4f})"
        plt.plot(self.time, self.ipr_values, color=plot_color, linestyle='-', label=label)
        plt.title(title, fontdict={'fontname': font}, fontsize=fontsize)
        plt.xlabel(xlabel, fontdict={'fontname': font}, fontsize=fontsize)
        plt.ylabel(ylabel, fontdict={'fontname': font}, fontsize=fontsize)
        plt.grid(True)
        plt.legend(fontsize=fontsize-5, frameon=False)
    
        if output_filename:
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.show()
    
        if time_range:
            print(f"Average IPR in range {start_time}-{end_time} fs: {average_ipr:.4f}")
        else:
            print(f"Average IPR (entire range): {average_ipr:.4f}")
    
        return average_ipr

    def animation_ipr(self, output_filename, fps=None, step_fs=None, plot_color='blue', title='Time vs. Charge Delocalisation', xlabel='Time (fs)', 
                    ylabel='Delocalisation', fontweight='bold', fontsize=14):
        """
        Create an animated line plot of IPR vs. time.
        Allows frame rate, sampling step, and visual customization.
        """
        # Calculate step size in indices
        step_size = int(np.round(step_fs / (self.time[1] - self.time[0])))

        time_sel = self.time[::step_size]
        ipr_sel = self.ipr_values[::step_size]
        avg_ipr = np.mean(ipr_sel)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title, fontsize=fontsize, fontweight=fontweight)
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        line, = ax.plot([], [], color=plot_color, linestyle='-', label=f"IPR (Avg: {avg_ipr:.2f})")
        legend = ax.legend(fontsize=fontsize-2, frameon=False)

        ax.set_xlim(time_sel.min(), time_sel.max())
        ax.set_ylim(ipr_sel.min(), ipr_sel.max())

        def update(frame):
            line.set_data(time_sel[:frame], ipr_sel[:frame])
            return line,

        ani = animation.FuncAnimation(fig, update, frames=len(time_sel), interval=1000/fps, blit=True)
        ani.save(output_filename, writer='ffmpeg', fps=fps)
        plt.show()
