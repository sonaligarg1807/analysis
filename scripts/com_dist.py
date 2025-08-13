#class containing all functions for calculating distances
import numpy as np
from typing import List, Tuple, Dict, Union, Optional

class COMDistanceCalculator:
    """
    Compute:
      • COM→COM distance between two molecules
      • Plane–plane distance using centroids of two planes (3 atoms each)
      • Atom–atom distance by 1-based global indices
      • COM→COM by explicit atom index ranges

    Modes:
    • mode='gro' : positions read in nm; molecules selected by resid (int)
    • mode='xyz' : positions read in Å; molecules selected by (start,end)
    AFTER the 2-line header, 1-based, inclusive (keeps your slice rule)

    Units:
    • Prints distances in Å and nm
    • Returns distances in nm

    PBC:
    • For .gro files, the last line is parsed as the box (orthorhombic or triclinic).
    • Use `use_pbc=True` (where available) to compute minimum-image distances.
    • For .xyz, no box is assumed (PBC disabled).
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Constants & mass table
    # ──────────────────────────────────────────────────────────────────────────
    atom_masses: Dict[str, float] = {
        'H': 1.008,
        'C': 12.011,
        'N': 14.007,
        'O': 15.999,
        'P': 30.974,
        'S': 32.06,
    }
    nm_to_angstrom: float = 10.0

    # ──────────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        input_file: str,
        mode: str,
        mol1: Optional[Union[int, Tuple[int, int]]] = None,
        mol2: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        if mode not in ('gro', 'xyz'):
            raise ValueError("Mode must be 'gro' or 'xyz'")

        self.input_file = input_file
        self.mode = mode
        self.mol1 = mol1
        self.mol2 = mol2

        # Parse molecules/atoms
        self.atoms = self._parse_gro_file() if mode == 'gro' else self._parse_xyz_file()

        # Build 1-based global index → (atomname, pos_Å)
        self._build_global_index_map()

        # Parse box for PBC if .gro
        self.box_A = self._parse_gro_box_to_matrix_A() if mode == 'gro' else None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers: parsing (.gro / .xyz)
    # ──────────────────────────────────────────────────────────────────────────
    def _parse_gro_file(self):
        """
        Return list of tuples: (resid:int, atomname:str, pos_nm:np.ndarray[3])
        """
        atoms = []
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        core = lines[2:-1]  # skip title, natoms; drop box
        for line in core:
            resid = int(line[0:5].strip())
            atomname = line[10:15].strip()
            x = float(line[20:28].strip())
            y = float(line[28:36].strip())
            z = float(line[36:44].strip())
            atoms.append((resid, atomname, np.array([x, y, z], dtype=float)))  # nm
        self._gro_box_line = lines[-1].strip()  # keep for box parsing
        return atoms

    def _parse_xyz_file(self):
        """
        Return list of tuples: (atomname:str, pos_A:np.ndarray[3])
        (after skipping the 2-line header)
        """
        atoms = []
        with open(self.input_file, 'r') as f:
            lines = f.readlines()[2:]  # skip 2-line header
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            atomname = parts[0]
            x, y, z = map(float, parts[1:4])
            atoms.append((atomname, np.array([x, y, z], dtype=float)))  # Å
        return atoms

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers: global index map for plane/atom ops (1-based → Å)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_global_index_map(self):
        """
        Build a dict mapping 1-based global atom index to (atomname, pos in Å).
        For .gro we convert nm→Å. For .xyz the indices count from the first
        atom line after the 2-line header (1-based).
        """
        index_map = {}
        if self.mode == 'gro':
            with open(self.input_file, 'r') as f:
                lines = f.readlines()[2:-1]
            for line in lines:
                atom_index = int(line[15:20].strip())  # .gro atom serial (1-based)
                atomname = line[10:15].strip()
                x = float(line[20:28].strip()) * self.nm_to_angstrom
                y = float(line[28:36].strip()) * self.nm_to_angstrom
                z = float(line[36:44].strip()) * self.nm_to_angstrom
                index_map[atom_index] = (atomname, np.array([x, y, z], dtype=float))
        else:  # xyz
            with open(self.input_file, 'r') as f:
                lines = f.readlines()[2:]  # after header
            for idx, line in enumerate(lines, start=1):  # 1-based
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                atomname = parts[0]
                x, y, z = map(float, parts[1:4])
                index_map[idx] = (atomname, np.array([x, y, z], dtype=float))
        self._index_map = index_map

    def _positions_by_indices_A(self, indices: List[int]) -> np.ndarray:
        """
        Return Nx3 array of positions in Å for the given 1-based global atom indices.
        """
        try:
            return np.vstack([self._index_map[i][1] for i in indices])
        except KeyError as e:
            raise ValueError(f"Atom index {e.args[0]} not found in file.") from None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers: box parsing (.gro) and minimum-image metric
    # ──────────────────────────────────────────────────────────────────────────
    def _parse_gro_box_to_matrix_A(self) -> Optional[np.ndarray]:
        """
        Parse the .gro box line into a 3x3 box matrix (Å). Supports:
        - 3 values: orthorhombic (a, b, c)
        - >=9 values: triclinic-like using the first 9 floats as row-major entries:
            [[vxx, vxy, vxz],
            [vyx, vyy, vyz],
            [vzx, vzy, vzz]]
        If parsing fails, returns None (PBC off).
        """
        try:
            toks = [float(x) for x in self._gro_box_line.split()]
            if len(toks) >= 9:
                mat_nm = np.array([
                    [toks[0], toks[3], toks[4]],
                    [toks[5], toks[1], toks[6]],
                    [toks[7], toks[8], toks[2]],
                ], dtype=float)
            elif len(toks) == 3:
                mat_nm = np.diag(toks)
            else:
                return None
            return mat_nm * self.nm_to_angstrom
        except Exception:
            return None

    @staticmethod
    def _min_image_distance_A(p1_A: np.ndarray, p2_A: np.ndarray, box_A: Optional[np.ndarray]) -> float:
        """
        Minimum-image distance between two points in Å.
        If box_A is None, returns simple Euclidean.
        For triclinic boxes, uses fractional wrapping: Δf = frac(Δr @ inv(box)) ∈ [-0.5,0.5),
        then Δr_min = box @ Δf.
        """
        if box_A is None:
            return float(np.linalg.norm(p2_A - p1_A))
        # invert box (3x3)
        try:
            inv_box = np.linalg.inv(box_A)
        except np.linalg.LinAlgError:
            # fallback to Euclidean if box is singular
            return float(np.linalg.norm(p2_A - p1_A))

        dr = (p2_A - p1_A).astype(float)
        df = inv_box @ dr
        df -= np.round(df)  # wrap to [-0.5, 0.5)
        dr_min = box_A @ df
        return float(np.linalg.norm(dr_min))

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers: selections & chemistry
    # ──────────────────────────────────────────────────────────────────────────
    def _element_symbol(self, atomname: str) -> str:
        """
        Extract a plausible element symbol (simple heuristic): first uppercase letter.
        Falls back to first character.
        """
        for ch in atomname:
            if ch.isalpha() and ch.upper() == ch and ch != ' ':
                return ch
        return atomname[0]

    def _get_molecule_atoms(self, identifier: Union[int, Tuple[int, int]]):
        """
        For .gro: identifier is resid (int), returns [(resid, name, pos_nm), ...]
        For .xyz: identifier is (start, end) 1-based AFTER header, inclusive.
                Returns [(name, pos_A), ...]
        NOTE: Keeps your original slice offset for compatibility.
        """
        if self.mode == 'gro':
            resid = int(identifier)
            return [atom for atom in self.atoms if atom[0] == resid]
        else:
            start, end = identifier
            return self.atoms[start - 3 : end - 2]  # original adjustment

    def _calculate_com(self, atoms) -> np.ndarray:
        """
        Mass-weighted center of mass.
        Returns:
        • nm for .gro inputs
        • Å  for .xyz inputs
        """
        total_mass = 0.0
        weighted = np.zeros(3, dtype=float)

        for atom in atoms:
            if self.mode == 'gro':
                atomname = atom[1]
                element = self._element_symbol(atomname)
                pos = atom[2]                 # nm
            else:
                atomname = atom[0]
                element = self._element_symbol(atomname)
                pos = atom[1]                 # Å

            mass = self.atom_masses.get(element, 12.011)  # default Carbon
            weighted += mass * pos
            total_mass += mass

        return weighted / total_mass

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _to_A(self, v: np.ndarray) -> np.ndarray:
        """Convert vector from native units to Å."""
        return v * self.nm_to_angstrom if self.mode == 'gro' else v

    @staticmethod
    def _centroid(P: np.ndarray) -> np.ndarray:
        """Centroid of Nx3 points (here N=3)."""
        P = np.asarray(P, dtype=float)
        if P.shape != (3, 3):
            raise ValueError("Provide exactly 3 atom indices per plane.")
        return P.mean(axis=0)

    def _mass_weighted_com_from_indices_A(self, indices: List[int]) -> np.ndarray:
        """
        Mass-weighted COM from 1-based atom indices (positions in Å).
        Uses element guess from stored atom names.
        """
        total = 0.0
        wsum = np.zeros(3, dtype=float)
        for i in indices:
            try:
                atomname, posA = self._index_map[i]
            except KeyError:
                raise ValueError(f"Atom index {i} not found in file.")
            elem = self._element_symbol(atomname)
            m = self.atom_masses.get(elem, 12.011)
            wsum += m * posA
            total += m
        if total == 0.0:
            raise ValueError("Total mass is zero; check input indices.")
        return wsum / total

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN #1: COM → COM distance (resid or (start,end))
    # ──────────────────────────────────────────────────────────────────────────
    def com_com_distance(
        self,
        mol1: Union[int, Tuple[int, int]],
        mol2: Union[int, Tuple[int, int]],
        print_info: bool = True,
        use_pbc: bool = False,
    ) -> float:
        """
        Compute COM→COM distance between two molecules.

        Args:
        mol1, mol2:
            • .gro: resid (int)
            • .xyz: (start, end) AFTER header, 1-based inclusive
        use_pbc: Use minimum-image metric (.gro only, requires box)
        Returns:
        Distance in nm (also prints Å)
        """
        atoms1 = self._get_molecule_atoms(mol1)
        atoms2 = self._get_molecule_atoms(mol2)

        com1_native = self._calculate_com(atoms1)  # nm (gro) / Å (xyz)
        com2_native = self._calculate_com(atoms2)

        if self.mode == 'gro':
            com1_A = self._to_A(com1_native)
            com2_A = self._to_A(com2_native)
            d_A = self._min_image_distance_A(com1_A, com2_A, self.box_A if use_pbc else None)
            d_nm = d_A / self.nm_to_angstrom
        else:
            d_A = self._euclidean(com1_native, com2_native)
            d_nm = d_A / self.nm_to_angstrom

        if print_info:
            print(f"COM 1: {com1_native}")
            print(f"COM 2: {com2_native}")
            print(f"COM–COM distance: {d_nm:.3f} nm  |  {d_A:.3f} Å"
                + ("  (PBC)" if (use_pbc and self.mode == 'gro' and self.box_A is not None) else ""))

        return d_nm

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN #2: Plane (centroid) → Plane (centroid) distance
    # ──────────────────────────────────────────────────────────────────────────
    def plane_centroid_distance(
        self,
        plane1_indices: List[int],
        plane2_indices: List[int],
        print_info: bool = True,
    ) -> float:
        """
        Distance between centroids of two planes (each defined by exactly 3 atoms).
        Atom indices are 1-based global indices:
        • .gro: atom serials
        • .xyz: 1-based after header
        Returns:
        Distance in nm (also prints Å)
        """
        P1_A = self._positions_by_indices_A(plane1_indices)  # Å
        P2_A = self._positions_by_indices_A(plane2_indices)  # Å

        C1_A = self._centroid(P1_A)
        C2_A = self._centroid(P2_A)

        d_A = self._euclidean(C1_A, C2_A)
        d_nm = d_A / self.nm_to_angstrom

        if print_info:
            print(f"Plane 1 atoms: {plane1_indices}, centroid (Å) = {C1_A}")
            print(f"Plane 2 atoms: {plane2_indices}, centroid (Å) = {C2_A}")
            print(f"Centroid–centroid distance: {d_A:.3f} Å  |  {d_nm:.3f} nm")

        return d_nm

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN #3: Atom → Atom distance (1-based indices)
    # ──────────────────────────────────────────────────────────────────────────
    def atom_atom_distance(
        self,
        index1: int,
        index2: int,
        print_info: bool = True,
        use_pbc: bool = False,
        output_file: Optional[str] = None,
    ) -> float:
        """
        Distance between two atoms by 1-based global indices.

        Args:
        index1, index2 : 1-based atom indices (for .gro: atom serials; for .xyz: 1-based after header)
        use_pbc        : Use minimum-image (.gro only, requires box)
        output_file    : If given, write a small report there.
        Returns:
        Distance in nm
        """
        p1_A = self._positions_by_indices_A([index1])[0]
        p2_A = self._positions_by_indices_A([index2])[0]

        if self.mode == 'gro' and not use_pbc:
            # .gro positions in Å were built from nm; also show nm positions like your snippet
            # Re-parse the nm for printing symmetry with your example:
            with open(self.input_file, 'r') as f:
                core = f.readlines()[2:-1]
            pos1_nm = np.array([float(core[index1 - 1][20:28]),
                                float(core[index1 - 1][28:36]),
                                float(core[index1 - 1][36:44])], dtype=float)
            pos2_nm = np.array([float(core[index2 - 1][20:28]),
                                float(core[index2 - 1][28:36]),
                                float(core[index2 - 1][36:44])], dtype=float)
        else:
            pos1_nm = p1_A / self.nm_to_angstrom
            pos2_nm = p2_A / self.nm_to_angstrom

        d_A = self._min_image_distance_A(p1_A, p2_A, self.box_A if use_pbc else None)
        d_nm = d_A / self.nm_to_angstrom

        if print_info:
            print(f"Atom {index1} position (nm): {pos1_nm}")
            print(f"Atom {index2} position (nm): {pos2_nm}")
            print(f"Distance: {d_nm:.3f} nm / {d_A:.3f} Å"
                + ("  (PBC)" if (use_pbc and self.mode == 'gro' and self.box_A is not None) else ""))

        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"Atom {index1} position (nm): {pos1_nm}\n")
                f.write(f"Atom {index2} position (nm): {pos2_nm}\n")
                f.write(f"Distance: {d_nm:.3f} nm / {d_A:.3f} Å\n")

        return d_nm

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN #4: COM distance by explicit atom ranges (inclusive)
    # ──────────────────────────────────────────────────────────────────────────
    def com_distance_by_atom_ranges(
        self,
        mol1_ranges: List[Tuple[int, int]],
        mol2_ranges: List[Tuple[int, int]],
        one_based: bool = True,
        print_info: bool = True,
        use_pbc: bool = False,
    ) -> float:
        """
        Compute COM→COM distance where each molecule is defined by inclusive atom
        index ranges (like your MDAnalysis version). Works for both .gro and .xyz.

        Args:
        mol1_ranges, mol2_ranges : list of (start, end), inclusive, typically 1-based
        one_based                : keep True for .gro-style indices
        use_pbc                  : Use minimum-image metric (.gro only, requires box)
        Returns:
        Distance in nm
        """
        def build_indices(ranges):
            idx = []
            for s, e in ranges:
                if not one_based:
                    s, e = s + 1, e + 1  # convert to 1-based inclusive
                if s > e:
                    raise ValueError(f"Invalid range {(s, e)} (start>end)")
                idx.extend(range(s, e + 1))
            return idx

        idx1 = build_indices(mol1_ranges)
        idx2 = build_indices(mol2_ranges)

        com1_A = self._mass_weighted_com_from_indices_A(idx1)  # Å
        com2_A = self._mass_weighted_com_from_indices_A(idx2)  # Å

        d_A = self._min_image_distance_A(com1_A, com2_A, self.box_A if use_pbc else None)
        d_nm = d_A / self.nm_to_angstrom

        if print_info:
            print(f"Mol1 ranges (inclusive): {mol1_ranges}  -> {len(idx1)} atoms")
            print(f"Mol2 ranges (inclusive): {mol2_ranges}  -> {len(idx2)} atoms")
            print(f"Mol1 COM (Å): {com1_A}")
            print(f"Mol2 COM (Å): {com2_A}")
            print(f"COM–COM distance (ranges): {d_nm:.3f} nm  |  {d_A:.3f} Å"
                + ("  (PBC)" if (use_pbc and self.mode == 'gro' and self.box_A is not None) else ""))

        return d_nm

    # ──────────────────────────────────────────────────────────────────────────
    # Back-compat shim
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> float:
        """
        Backward-compatible wrapper:
        Calls com_com_distance(self.mol1, self.mol2).
        """
        if self.mol1 is None or self.mol2 is None:
            raise ValueError("run() needs mol1 and mol2 set on the instance.")
        return self.com_com_distance(self.mol1, self.mol2, print_info=True)