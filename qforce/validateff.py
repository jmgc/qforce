import numpy as np
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from .elements import ATOM_SYM
#
from .frequencies import calc_vibrational_frequencies


def compute_forces(coords, mol):
    """
    Scope:
    ------
    For each displacement, calculate the forces.
    """
    energy = 0.0
    force = np.zeros((mol.topo.n_atoms, 3), dtype=float)

    for term in mol.terms:
        energy += term.do_force(coords, force)

    return energy, force


def compute_hessian(coords, mol, displ=1e-5):
    """
    Scope:
    -----
    Perform displacements to calculate the MD hessian numerically.
    """
    full_hessian = np.zeros((3*mol.topo.n_atoms, 3*mol.topo.n_atoms), dtype=float)
    displ_twice = 2.0*displ

    with mol.terms.add_ignore('charge_flux'):
        for a in range(mol.topo.n_atoms):
            for xyz in range(3):
                coords[a][xyz] += displ
                _, f_plus = compute_forces(coords, mol)
                coords[a][xyz] -= displ_twice
                _, f_minus = compute_forces(coords, mol)
                coords[a][xyz] += displ
                diff = - (f_plus - f_minus) / displ_twice
                full_hessian[a*3+xyz] = diff.flatten()

    return full_hessian


def _compute_mean_percent_error(qm_freq, md_freq):
    #
    if qm_freq[5] > 300:
        transrot = 5
    else:
        transrot = 6
    mask = np.arange(qm_freq.size) >= transrot
    #
    qm_freq = qm_freq[mask]
    md_freq = md_freq[mask]
    #
    errors = []
    for i, (q, m) in enumerate(zip(qm_freq, md_freq)):
        diff = q - m
        err = diff / q * 100
        if q > 100:
            errors.append(err)
    #
    return np.abs(np.array(errors)).mean()


def _plot_frequencies(folder, qm_freq, md_freq):
    matplotlib.use('Agg')
    mean_percent_error = _compute_mean_percent_error(qm_freq, md_freq)
    n_freqs = np.arange(len(qm_freq))+1
    width, height = plt.figaspect(0.6)
    f = plt.figure(figsize=(width, height), dpi=300)
    sns.set(font_scale=1.3)
    plt.title(f'Mean Percent Error = {round(mean_percent_error, 2)}%', loc='left')
    plt.xlabel('Vibrational Mode #')
    plt.ylabel(r'Frequencies (cm$^{-1}$)')
    plt.plot(n_freqs, qm_freq, linewidth=3, label='QM')
    plt.plot(n_freqs, md_freq, linewidth=3, label='Q-Force')
    plt.tight_layout()
    plt.legend(ncol=2, bbox_to_anchor=(1.03, 1.12), frameon=False)
    f.savefig(folder / "frequencies.pdf", bbox_inches='tight')
    plt.close()


def _write_vibrational_frequencies(folder, name, qm_freq, qm_vec, md_freq, md_vec, qm):
    """
    Scope:
    ------
    Create the following files for comparing QM reference to the generated
    MD frequencies/eigenvalues.

    Output:
    ------
    JOBNAME_qforce.freq : QM vs MD vibrational frequencies and eigenvectors
    JOBNAME_qforce.nmd : MD eigenvectors that can be played in VMD with:
                                vmd -e filename
    """
    freq_file = folder / "frequencies.txt"
    nmd_file = folder / "frequencies.nmd"
    errors = []

    if qm_freq[5] > 300:
        transrot = 5
    else:
        transrot = 6
    mask = np.arange(qm_freq.size) >= transrot

    qm_freq = qm_freq[mask]
    md_freq = md_freq[mask]
    qm_vec = qm_vec[mask]
    md_vec = md_vec[mask]

    with open(freq_file, "w") as f:
        f.write(" mode    QM-Freq     MD-Freq       Diff.  %Error\n")
        for i, (q, m) in enumerate(zip(qm_freq, md_freq)):
            diff = q - m
            err = diff / q * 100
            if q > 100:
                errors.append(err)
            f.write(f"{i+transrot+1:>4}{q:>12.3f}{m:>12.3f}{diff:>12.3f}{err:>8.2f}\n")
        f.write("\n\n         QM vectors              MD Vectors\n")
        f.write(62*"=")
        for i, (qm1, md1) in enumerate(zip(qm_vec, md_vec)):
            f.write(f"\nMode {i+transrot+1}\n")
            for qm2, md2 in zip(qm1, md1):
                f.write("{:>10.5f}{:>10.5f}{:>10.5f}  {:>10.5f}{:>10.5f}{:>10.5f}\n".format(*qm2,
                                                                                            *md2))

    with open(nmd_file, "w") as nmd:
        nmd.write(f"nmwiz_load {name}_qforce.nmd\n")
        nmd.write(f"title {name}\n")
        nmd.write("names")
        for ids in qm.atomids:
            nmd.write(f" {ATOM_SYM[ids]}")
        nmd.write("\nresnames")
        for i in range(qm.n_atoms):
            nmd.write(" RES")
        nmd.write("\nresnums")
        for i in range(qm.n_atoms):
            nmd.write(" 1")
        nmd.write("\ncoordinates")
        for c in qm.coords:
            nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
        for i, m in enumerate(md_vec):
            nmd.write(f"\nmode {i+7}")
            for c in m:
                nmd.write(f" {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")


class ValidateFF:

    def __init__(self, mol, jobname):
        self.mol = mol
        self.name = jobname

    def hessian(self, folder, qmout):
        """Create Hessian analysis inside a folder"""
        os.makedirs(folder, exist_ok=True)
        full_hessian = compute_hessian(qmout.coords, self.mol)
        # compute mm hessian
        mm_hessian = []
        count = 0
        for i in range(self.mol.topo.n_atoms*3):
            for j in range(i+1):
                hes = (full_hessian[i, j] + full_hessian[j, i]) / 2.0
                if round(hes, 4) == 0.0 or np.abs(qmout.hessian[count]) < 0.0001:
                    mm_hessian.append(0.0)
                else:
                    mm_hessian.append(hes)
                count += 1
        #
        qm_freq, qm_vec = calc_vibrational_frequencies(qmout.hessian, qmout)
        mm_freq, mm_vec = calc_vibrational_frequencies(mm_hessian, qmout)
        #
        _plot_frequencies(folder, qm_freq, mm_freq)
        _write_vibrational_frequencies(folder, self.name, qm_freq, qm_vec, mm_freq, mm_vec, qmout)

    def hessians(self, folder, structs):
        for i, (_, qmout) in enumerate(structs.hessitr()):
            path = folder / f"hessian_{i:02d}"
            self.hessian(path, qmout)

    def energy_errors(self, folder, structs):
        errors = []
        for i, (_, qmout) in enumerate(structs.enitr()):
            e, _ = compute_forces(qmout.coords, self.mol)
            errors.append(qmout.energy - e)
        return errors
