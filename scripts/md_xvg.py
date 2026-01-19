#!/usr/bin/env python3
"""
General purpose script for plotting .xvg files from MD simulations.
Plots column 1 (usually time) vs column 2 (parameter of interest).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys


class XVGPlotter:
    """
    A class for reading and plotting .xvg files from MD simulations.
    
    Attributes:
    -----------
    filename : str
        Path to the .xvg file
    time : np.ndarray
        Data from column 1 (usually time)
    parameter : np.ndarray
        Data from column 2 (parameter of interest)
    """
    
    def __init__(self, filename):
        """
        Initialize XVGPlotter with a file.
        
        Parameters:
        -----------
        filename : str
            Path to the .xvg file
        """
        self.filename = filename
        self.time = None
        self.parameter = None
    
    def read_xvg(self):
        """
        Read .xvg file and load data from columns 1 and 2.
        Skips comment lines starting with # or @.
        
        Returns:
        --------
        self : XVGPlotter
            Returns self for method chaining
        """
        time = []
        parameter = []
        
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    # Skip comment lines
                    if line.startswith('#') or line.startswith('@'):
                        continue
                    # Skip empty lines
                    if line.strip() == '':
                        continue
                    
                    # Parse data
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            time.append(float(parts[0]))
                            parameter.append(float(parts[1]))
                    except (ValueError, IndexError):
                        continue
            
            self.time = np.array(time)
            self.parameter = np.array(parameter)
            return self
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.filename}' not found.")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
    
    def filter_time_range(self, start_time=None, end_time=None):
        """
        Filter data to only include specified time range.
        
        Parameters:
        -----------
        start_time : float, optional
            Start time for filtering
        end_time : float, optional
            End time for filtering
        
        Returns:
        --------
        self : XVGPlotter
            Returns self for method chaining
        """
        if self.time is None or self.parameter is None:
            raise ValueError("Data not loaded. Call read_xvg() first.")
        
        mask = np.ones(len(self.time), dtype=bool)
        
        if start_time is not None:
            mask &= (self.time >= start_time)
        
        if end_time is not None:
            mask &= (self.time <= end_time)
        
        self.time = self.time[mask]
        self.parameter = self.parameter[mask]
        return self
    
    def plot(self, xlabel, ylabel, title=None, output=None, figsize=(10, 6), 
             linewidth=1.5, dpi=300):
        """
        Create and save/show plot of time series data.
        
        Parameters:
        -----------
        xlabel : str
            Label for x-axis
        ylabel : str
            Label for y-axis
        title : str, optional
            Plot title
        output : str, optional
            Output filename to save plot. If None, display plot.
        figsize : tuple, optional
            Figure size (width, height) in inches
        linewidth : float, optional
            Line width for plot
        dpi : int, optional
            DPI for saved figure
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object
        """
        if self.time is None or self.parameter is None:
            raise ValueError("Data not loaded. Call read_xvg() first.")
        
        fig = plt.figure(figsize=figsize)
        plt.plot(self.time, self.parameter, linewidth=linewidth)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        if title:
            plt.title(title, fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output:
            plt.savefig(output, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {output}")
        else:
            plt.show()
        
        return fig
    
    def get_data(self):
        """
        Get the loaded time and parameter data.
        
        Returns:
        --------
        time : np.ndarray
            Time data
        parameter : np.ndarray
            Parameter data
        """
        if self.time is None or self.parameter is None:
            raise ValueError("Data not loaded. Call read_xvg() first.")
        return self.time, self.parameter
    
    def get_statistics(self):
        """
        Calculate basic statistics of the parameter data.
        
        Returns:
        --------
        stats : dict
            Dictionary containing mean, std, min, max of the parameter
        """
        if self.parameter is None:
            raise ValueError("Data not loaded. Call read_xvg() first.")
        
        return {
            'mean': np.mean(self.parameter),
            'std': np.std(self.parameter),
            'min': np.min(self.parameter),
            'max': np.max(self.parameter),
            'n_points': len(self.parameter)
        }




def main():
    """Command-line interface for XVGPlotter."""
    parser = argparse.ArgumentParser(
        description='Plot .xvg files from MD simulations (column 1 vs column 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.xvg -x "Time (ps)" -y "RMSD (nm)"
  %(prog)s input.xvg -x "Time (ps)" -y "Energy (kJ/mol)" --start 1000 --end 5000
  %(prog)s input.xvg -x "Time (ps)" -y "Temperature (K)" -o output.png --title "Temperature vs Time"
        """
    )
    
    parser.add_argument('input', help='Input .xvg file')
    parser.add_argument('-x', '--xlabel', required=True, help='Label for x-axis (e.g., "Time (ps)")')
    parser.add_argument('-y', '--ylabel', required=True, help='Label for y-axis (e.g., "RMSD (nm)")')
    parser.add_argument('-t', '--title', help='Plot title (optional)')
    parser.add_argument('-o', '--output', help='Output filename to save plot (e.g., plot.png). If not specified, plot is displayed.')
    parser.add_argument('--start', type=float, help='Start time for filtering data')
    parser.add_argument('--end', type=float, help='End time for filtering data')
    
    args = parser.parse_args()
    
    try:
        # Create plotter instance and read data
        plotter = XVGPlotter(args.input)
        print(f"Reading data from: {args.input}")
        plotter.read_xvg()
        
        time, parameter = plotter.get_data()
        print(f"Loaded {len(time)} data points")
        print(f"Time range: {time.min():.2f} to {time.max():.2f}")
        
        # Filter time range if specified
        if args.start is not None or args.end is not None:
            plotter.filter_time_range(args.start, args.end)
            time, parameter = plotter.get_data()
            print(f"Filtered to {len(time)} data points")
            if args.start is not None:
                print(f"Start time: {args.start}")
            if args.end is not None:
                print(f"End time: {args.end}")
        
        # Create plot
        plotter.plot(args.xlabel, args.ylabel, args.title, args.output)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

