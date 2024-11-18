import numpy as np
#
from .baseterms import TermBase
from ..forces import get_dist, get_angle
from ..forces import calc_bonds, calc_morse_bonds, calc_angles, calc_cosine_angles


class HarmonicBondTerm(TermBase):
    name = 'HarmonicBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_bonds(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.bondname(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            bond_terms.append(cls([a1, a2], dist, bond['vers']))

        return bond_terms

    def write_forcefield(self, software, writer):
        software.write_harmonic_bond_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_harmonic_bond_header(writer)


class MorseBondTerm(TermBase):
    name = 'MorseBondTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_morse_bonds(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.bondname(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ[0] = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
        bond_terms = cls.get_terms_container()

        for a1, a2 in topo.bonds:
            bond = topo.edge(a1, a2)
            dist = bond['length']
            bond_terms.append(cls([a1, a2], [dist, 2.2], bond['vers']))

        return bond_terms

    def write_forcefield(self, software, writer):
        software.write_morse_bond_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_morse_bond_header(writer)


class HarmonicAngleTerm(TermBase):
    name = 'HarmonicAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_angles(crd, self.atomids, self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.anglename(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
        angle_terms = cls.get_terms_container()

        for a1, a2, a3 in topo.angles:

            if not topo.edge(a2, a1)['in_ring3'] or not topo.edge(a2, a3)['in_ring3']:
                theta = get_angle(topo.coords[[a1, a2, a3]])[0]
                if theta > 2.9671:  # if angle is larger than 170 degree, make it 180
                    theta = np.pi

                b21 = topo.edge(a2, a1)['vers']
                b23 = topo.edge(a2, a3)['vers']
                a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                                 f"{topo.types[a2]}({b23}){topo.types[a3]}"])
                a_type = f"{a_type[0]}_{a_type[1]}"
                angle_terms.append(cls([a1, a2, a3], theta, a_type))

        return angle_terms

    def write_forcefield(self, software, writer):
        software.write_harmonic_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_harmonic_angle_header(writer)


class CosineAngleTerm(HarmonicAngleTerm):
    name = 'CosineAngleTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_cosine_angles(crd, self.atomids, self.equ, fconst, force)

    def write_forcefield(self, software, writer):
        software.write_cosine_angle_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_cosine_angle_header(writer)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.anglename(*self.atomids)]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value


class UreyBradleyTerm(TermBase):
    name = 'UreyBradleyTerm'

    def _calc_forces(self, crd, force, fconst):
        return calc_bonds(crd, self.atomids[::2], self.equ, fconst, force)

    def constants(self):
        """return constants for the class should return a list of names of the constants used in the class"""
        return [self.bondname(*self.atomids[::2])]

    def update_constants(self, dct):
        """update constants for the class"""
        name = self.constants()[0]
        value = dct.get(name, None)
        if value is not None:
            self.equ = value

    @classmethod
    def _get_terms(cls, topo, non_bonded):
        urey_terms = cls.get_terms_container()

        for a1, a2, a3 in topo.angles:
            dist = get_dist(topo.coords[a1], topo.coords[a3])[1]
            theta = get_angle(topo.coords[[a1, a2, a3]])[0]
            #  No Urey term  if linear angle (>170) or if in 3-member ring
            if theta < 2.9671 and (not topo.edge(a2, a1)['in_ring3'] or
                                   not topo.edge(a2, a3)['in_ring3']):
                b21, b23 = topo.edge(a2, a1)['vers'], topo.edge(a2, a3)['vers']

                a_type = sorted([f"{topo.types[a2]}({b21}){topo.types[a1]}",
                                 f"{topo.types[a2]}({b23}){topo.types[a3]}"])
                a_type = f"{a_type[0]}_{a_type[1]}"

                urey_terms.append(cls([a1, a2, a3], dist, a_type))

        return urey_terms

    def write_forcefield(self, software, writer):
        software.write_urey_bradley_term(self, writer)

    def write_ff_header(self, software, writer):
        return software.write_urey_bradley_header(writer)


def get_bond_dissociation_energies(md_data, mol):
    dissociation_energies = {}

    bde_dict = read_bond_dissociation_energy_csv(md_data)

    for a1, a2, props in mol.topo.graph.edges(data=True):
        a1, a2 = sorted((a1, a2))
        print(a1, a2, )
        if props['type'] not in bde_dict:
            raise ValueError('Morse potential chosen, but dissociation energy not known for this atom number pair '
                             f'with the bond order in parenthesis: {props["type"]}. '
                             'You can add this too csv file in the data directory')

        neighs1 = [edge[2]['type'] for edge in mol.topo.graph.edges(a1, data=True)]
        neighs2 = [edge[2]['type'] for edge in mol.topo.graph.edges(a2, data=True)]
        e_dis, n_matches = choose_bond_type(bde_dict[props['type']], neighs1, neighs2)

        if mol.atomids[a1] == mol.atomids[a2]:
            e_dis2, n_matches2 = choose_bond_type(bde_dict[props['type']], neighs1, neighs2, 1, 0)
            if n_matches2 > n_matches:
                e_dis = e_dis2

        dissociation_energies[(a1, a2)] = float(e_dis)
    return dissociation_energies

def read_bond_dissociation_energy_csv(md_data):
    bde_dict = {}

    with open(f'{md_data}/bond_dissociation_energy.csv', 'r') as file:
        file.readline()
        for line in file:
            a1, a2, b_order, a1_neighs, a2_neighs, de = line.split(',')[:6]
            a1_neighs = read_bond_dissociation_neighbors(a1, a1_neighs)
            a2_neighs = read_bond_dissociation_neighbors(a2, a2_neighs)
            name = f'{a1}({float(b_order):.1f}){a2}'

            if name in bde_dict:
                bde_dict[name].append((a1_neighs, a2_neighs, de))
            else:
                bde_dict[name] = [(a1_neighs, a2_neighs, de)]

    return bde_dict


def choose_bond_type(bde_list, neighs1, neighs2, ndx1=0, ndx2=1):
    highest_match = 0
    match_type = None

    for bde_type in bde_list:
        current_sum1 = sum(typ in neighs1 for typ in bde_type[ndx1])
        current_sum2 = sum(typ in neighs2 for typ in bde_type[ndx2])
        current_sum = current_sum1 + current_sum2

        if current_sum > highest_match:
            highest_match = current_sum
            match_type = bde_type
        elif highest_match == 0 and bde_type[0] == [] and bde_type[1] == []:
            match_type = bde_type

    return match_type[2], highest_match


def read_bond_dissociation_neighbors(a, neighs):
    neighs_formatted = []
    if neighs != '*' and neighs != '':
        neighs = neighs.split()
        for neigh in neighs:
            a_neigh, bo_neigh = neigh.split('_')
            neighs_formatted.append(f'{min(a, a_neigh)}({float(bo_neigh):.1f}){max(a, a_neigh)}')
    return neighs_formatted
