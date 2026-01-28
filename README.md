# Analysis Scripts

Python package for molecular dynamics and quantum mechanics analysis.

## Modules

### com_dist.py
Calculates distances between molecules and atoms. Supports COM-to-COM, plane-to-plane, and atom-to-atom distances. Works with .gro (nm) and .xyz (Å) files with PBC support.

**Initialization:**
```python
from scripts.com_dist import COMDistanceCalculator

# For .gro files (molecules selected by residue ID)
calc = COMDistanceCalculator('input.gro', mode='gro', mol1=1, mol2=2)

# For .xyz files (molecules selected by line ranges, 1-based after header)
calc = COMDistanceCalculator('input.xyz', mode='xyz', mol1=(1, 50), mol2=(51, 100))
```

**Methods:**
```python
# COM-to-COM distance
dist = calc.com_com_distance(mol1=1, mol2=2, print_info=True, use_pbc=False)

# Plane-to-plane distance (using 3 atom indices each)
dist = calc.plane_centroid_distance(plane1_indices=[1, 2, 3], plane2_indices=[4, 5, 6], print_info=True)

# Atom-to-atom distance
dist = calc.atom_atom_distance(index1=1, index2=2, print_info=True, use_pbc=False, output_file='dist.txt')

# COM distance by explicit atom ranges
dist = calc.com_distance_by_atom_ranges(
    mol1_ranges=[(1, 50), (100, 150)],
    mol2_ranges=[(200, 250)],
    one_based=True,
    print_info=True,
    use_pbc=False
)
```

**Parameters:**
- `input_file`: Path to .gro or .xyz file
- `mode`: 'gro' or 'xyz'
- `mol1, mol2`: Molecule selectors (int for .gro, tuple for .xyz)
- `use_pbc`: Enable periodic boundary conditions (only for .gro)
- `print_info`: Print distance information
- `output_file`: Save results to file

---

### combine_video.py
Combines two videos by stacking them horizontally or vertically using FFmpeg.

**Usage:**
```python
from scripts.combine_video import CombineVideo

# Horizontal stacking
combiner = CombineVideo('video1.mp4', 'video2.mp4', 'output.mp4', stack_direction='horizontal')
combiner.stack_videos()

# Vertical stacking
combiner = CombineVideo('video1.mp4', 'video2.mp4', 'output.mp4', stack_direction='vertical')
combiner.stack_videos()
```

**Parameters:**
- `video1_path`: Path to first video
- `video2_path`: Path to second video
- `output_path`: Output path for combined video
- `stack_direction`: 'horizontal' or 'vertical' (default: 'horizontal')

---

### COMEvolution.py
Visualizes charge evolution over time by tracking QM region center of mass positions.

**Usage:**
```python
from scripts.COMEvolution import COMEvolution

com = COMEvolution(
    gro_file='input.gro',
    qm_resids=[1, 2, 3],  # or "[1 2 3]" as string
    xyz_ref='ref.xyz',
    com_file='com.xvg',
    save_path='animation.mp4',  # or 'animation.gif'
    annotate_resids=[1],
    annotate_label='GB Site',
    qm_color='red',
    charge_color='blue',
    title='Charge Evolution Over Time'
)
com.animate()
```

**Parameters:**
- `gro_file`: Path to .gro file
- `qm_resids`: List of QM residue IDs or string "[1 2 3]"
- `xyz_ref`: Reference .xyz file
- `com_file`: Path to .xvg file with COM data
- `save_path`: Save animation as .mp4 (ffmpeg) or .gif (pillow), None to display
- `annotate_resids`: List of residues to annotate
- `annotate_label`: Label for annotated sites
- `qm_color`: Color for QM sites (default: 'red')
- `charge_color`: Color for charge marker (default: 'blue')
- `title`: Plot title

---

### cpl_map.py
Visualizes molecular coupling data on 2D/3D scatter plots or heatmaps from .gro files.

**Usage:**
```python
from scripts.cpl_map import CouplingVisualizer

viz = CouplingVisualizer('couplings.dat', 'avg.gro')

# Histogram with Gaussian fit
viz.plot_histogram_fit()

# Interpolated gradient field
viz.plot_interpolated_gradient_field(
    plane='xy',  # or 'xz', 'yz'
    bins=(30, 30),
    cmap='inferno',
    annotate_all=False,
    annotate_resids_grain=[1, 2, 3],
    annotate_resids_gb=[10, 11]
)

# Local gradient vectors
viz.plot_local_gradient_vectors(
    plane='xy',
    bins=(35, 35),
    cmap='inferno',
    cutoff=0.6,
    scale=0.05,
    annotate_all=False,
    annotate_resids_grain=[1, 2],
    annotate_resids_gb=[5, 6]
)
```

**Parameters:**
- `cpls_path`: Path to coupling data file (format: resid coupling)
- `avg_gro_path`: Path to averaged .gro file
- `plane`: Projection plane ('xy', 'xz', 'yz')
- `bins`: Grid resolution for interpolation
- `cmap`: Matplotlib colormap
- `cutoff`: Neighbor cutoff distance (nm) for gradient calculation
- `scale`: Arrow scale for gradient vectors
- `annotate_all`: Annotate all residues
- `annotate_resids_grain`: List of grain residues to annotate (black)
- `annotate_resids_gb`: List of grain boundary residues to annotate (yellow)

---

### ipr.py
Plots and calculates Inverse Participation Ratio (IPR) from time-series data. Creates plots and animations.

**Usage:**
```python
from scripts.ipr import IPR

ipr = IPR('ipr_data.dat')

# Plot averaged IPR
avg_ipr = ipr.averaged_ipr(
    output_filename='ipr_plot.png',
    legend_name='IPR',
    time_range=(0, 1000),  # Optional: (start_time, end_time)
    plot_color='blue',
    title='Time vs. IPR',
    xlabel='Time (fs)',
    ylabel='IPR',
    font='Comic Sans MS',
    fontsize=15
)

# Create animation
ipr.animation_ipr(
    output_filename='ipr_animation.mp4',
    fps=20,
    step_fs=10,
    plot_color='blue',
    title='Time vs. Charge Delocalisation',
    xlabel='Time (fs)',
    ylabel='Delocalisation',
    fontweight='bold',
    fontsize=14
)
```

**Parameters:**
- `filename`: Path to IPR data file (2 columns: time, ipr_value)
- `output_filename`: Save plot/animation to file
- `legend_name`: Custom legend name
- `time_range`: Tuple (start, end) to filter time range
- `plot_color`: Line color
- `title`, `xlabel`, `ylabel`: Plot labels
- `font`: Font name for labels
- `fontsize`: Font size
- `fps`: Frames per second for animation
- `step_fs`: Time step between frames (fs)

---

### md_xvg.py
Plots .xvg files from MD simulations (column 1 vs column 2).

**Usage:**
```python
from scripts.md_xvg import XVGPlotter

plotter = XVGPlotter('data.xvg')
plotter.read_xvg()

# Filter time range (optional)
plotter.filter_time_range(start_time=100, end_time=500)

# Create plot
plotter.plot(
    xlabel='Time (ps)',
    ylabel='RMSD (nm)',
    title='RMSD over Time',
    output='plot.png',  # None to display
    figsize=(10, 6),
    linewidth=1.5,
    dpi=300
)

# Get data
time, param = plotter.get_data()

# Get statistics
stats = plotter.get_statistics()
# Returns: {'mean', 'std', 'min', 'max', 'n_points'}
```

**Command-line usage:**
```bash
python md_xvg.py input.xvg -x "Time (ps)" -y "RMSD (nm)" -o output.png --start 100 --end 500 -t "Title"
```

**Parameters:**
- `filename`: Path to .xvg file
- `start_time`, `end_time`: Time range for filtering
- `xlabel`, `ylabel`: Axis labels
- `title`: Plot title
- `output`: Output filename (None to display)
- `figsize`: Figure size tuple (width, height)
- `linewidth`: Line width
- `dpi`: Resolution for saved figure

---

### mobility.py
Calculates charge mobility from mean square displacement (MSD) data.

**Usage:**
```python
from scripts.mobility import Mobility

mob = Mobility(temperature=300)  # Temperature in Kelvin

# Plot multiple files
mob.plot_multi_file_msd(
    fnames=['file1.dat', 'file2.dat'],
    labels=['System 1', 'System 2'],
    colors=['blue', 'red'],
    ns=[1, 2],  # Number of charge carriers
    start_t=0,
    end_t=1000,
    save_filename='mobility.png'
)

# Plot single file with fit line
mob.plot_single_file_msd(
    fname='msd.dat',
    start_t=0,
    end_t=1000,
    n=1,
    save_filename='single_msd.png'
)
```

**Parameters:**
- `temperature`: Temperature in Kelvin (default: 300)
- `fnames`: List of MSD data files
- `labels`: List of labels for each file
- `colors`: List of colors for each plot
- `ns`: List of charge carrier numbers for each system
- `start_t`, `end_t`: Time range for linear fit (fs)
- `save_filename`: Output filename (None to display)

**Note:** MSD files should have columns: time, ..., msd_x, msd_y, msd_z (columns 0, 7, 8, 9)

---

### pathTB_avg.py
Averages data files across multiple trajectory subdirectories (subdirs containing TRAJ folders).

**Usage:**
```python
from scripts.pathTB_avg import PathTBAverager

averager = PathTBAverager(
    base_dir='/path/to/base',
    output_path='averaged_output.dat',
    input_filename='TB_COUPLING.dat'
)
averager.average()
```

**Parameters:**
- `base_dir`: Root directory containing 'subdir_*' folders
- `output_path`: File path to save averaged data
- `input_filename`: Name of file to average within each TRAJ directory

**Directory structure expected:**
```
base_dir/
  subdir_1/
    TRAJ0/input_filename
    TRAJ1/input_filename
  subdir_2/
    TRAJ0/input_filename
    TRAJ1/input_filename
```

---

### plot_occupation.py
Plots and analyzes site occupation over time from trajectory data.

**Usage:**
```python
from scripts.plot_occupation import PlotOccupation

plotter = PlotOccupation(
    significance_threshold=0.05,
    edge_site_count=3
)

# Plot averaged occupation
plotter.averaged_occupation(
    file_path='occupation.dat',
    selected_sites=[0, 1, 2],  # None for all sites
    time_skip=20
)

# Analyze individual trajectories
plotter.individual_occupation(
    parent_dir='trajectories/',
    output_file='tb_occupation_stats.dat',
    sufficiency_file='qm_zone_check.dat',
    plot_last_site=True
)
```

**Parameters:**
- `significance_threshold`: Threshold for significant site occupation (default: 0.05)
- `edge_site_count`: Number of edge sites to check (default: 3)
- `file_path`: Path to occupation data file
- `selected_sites`: List of site indices to plot (None = all)
- `time_skip`: Plot every Nth time step
- `parent_dir`: Directory containing subdir_* folders with trajectory data
- `output_file`: Output file for occupation statistics
- `sufficiency_file`: Output file for QM zone sufficiency check
- `plot_last_site`: Whether to plot last site occupation across trajectories

---

### slab_extr.py
Extracts molecular slabs from .gro files based on center of mass position.

**Usage:**
```python
from scripts.slab_extr import SlabExtractor

extractor = SlabExtractor(
    gro_path='input.gro',
    natoms=10,  # Number of atoms per molecule
    vel=True    # True if .gro contains velocities
)

# Extract slab along single direction
slab = extractor.select_slab_single_direction(
    lim=[lower_bound, upper_bound, 'z']  # axis: 'x', 'y', or 'z'
)

# Extract slab with multiple direction constraints
slab = extractor.select_slab_multi_direction(
    lim={'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
)

# Write extracted slab to .xyz file
extractor.write_xyz(slab, outname='slab.xyz')
```

**Parameters:**
- `gro_path`: Path to .gro file
- `natoms`: Number of atoms per molecule
- `vel`: Whether .gro file contains velocities (default: True)
- `lim`: Boundary limits for slab extraction
  - Single direction: [lower, upper, 'axis']
  - Multi direction: {'axis': (lower, upper), ...}
- `outname`: Output .xyz filename

---

### TB_avg.py
Averages data files from multiple trajectory directories in parallel.

**Usage:**
```python
from scripts.TB_avg import TBAverager

averager = TBAverager(
    base_dir='/path/to/trajectories',
    output_path='averaged_output.dat',
    input_filename='TB_COUPLING.dat',
    traj_dirs=None,  # Auto-detect TRAJ* dirs, or provide list
    max_workers=16
)
averager.average()
```

**Parameters:**
- `base_dir`: Path containing trajectory directories (TRAJ0, TRAJ1, ...)
- `output_path`: Path to save averaged data
- `input_filename`: Name of input file inside each trajectory folder
- `traj_dirs`: List of trajectory directory names (None = auto-detect TRAJ*)
- `max_workers`: Number of parallel threads (default: 16)

**Directory structure expected:**
```
base_dir/
  TRAJ0/input_filename
  TRAJ1/input_filename
  TRAJ2/input_filename
```