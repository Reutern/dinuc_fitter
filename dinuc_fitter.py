#!/usr/bin/env python
#
#Created 10.8.2017
#
#@author: Marc von Reutern
#
#
#Programm to fit a PWM and compare the result with the dinucleotide mutations

import pylab
import matplotlib.pyplot as plt
import numpy as np
import sys, getopt
import os.path
plt.style.use('ggplot')
plt.rcParams['axes.facecolor']='w'

baseDic = {'A':0, 'a':0, 'C':1, 'c':1, 'G':2, 'g':2, 'T':3, 't':3, 'N':100, 'n':100, '-':1000}
bases = ['A', 'C', 'G', 'T']


class Motif:
    n_pos = 0
    PWM = np.zeros([0,4])
    DPWM = np.zeros([0,4])
    consensus = ''
    
    def __init__(self, n_pos_tmp):
        self.n_pos = n_pos_tmp
        self.PWM = np.ones([n_pos_tmp, 4])
        self.DPWM = np.ones([n_pos_tmp-1, 16])
        self.consensus = 'N' * n_pos_tmp

    def score_site(self, seq, first_order=True):    # calculate the binding weight
        if len(seq) != self.n_pos:
            print("Error: sequence length {} does not fit PWM length {}.".format(len(seq), n_pos))
            return RuntimeError
        weight = 1            
        # score zero-order motif
        for pos in range(self.n_pos):
            base = baseDic[seq[pos]]
            if base < 4:
                weight *= self.PWM[pos, base]
            elif base == 100:
                weight *= min(self.PWM[pos,:])        # if base is a N, score worst possible option
            elif base == 1000:
                weight = 0                        # if position is missing (base = '-'), score no binding
        # score first-order motif
        if first_order == True:
            for pos in range(self.n_pos-1):
                base_1 = baseDic[seq[pos]]
                base_2 = baseDic[seq[pos+1]]
                dinuc = 4 * base_1 + base_2        
                if dinuc < 16:
                    weight *= self.DPWM[pos, dinuc]
                elif dinuc < 1000:
                    weight *= min(self.DPWM[pos,:])        # if base is a N, score worst possible option
                elif dinuc > 1000:
                    weight = 0                        # if position is missing (base = '-'), score no binding        
        return weight

    def score_reverse_site(self, seq, first_order=True):    # calculate the binding weight of the reverse complement
        seq_reverse = reverse_complement(seq)
        weight = self.score_site(seq_reverse, first_order)
        return weight
    
    def set_consensus(self, sites):
        n = len(sites[0][0])
        consensus_tmp = ''
        for pos in range(n):
            bases_tmp = []
            for site in sites:
                seq = site[0]
                bases_tmp.append(seq[pos])
            consensus_tmp += max(set(bases_tmp), key=bases_tmp.count)
        self.consensus = consensus_tmp
        return consensus_tmp    
    
    # generate_motif_from_sites builds a binding model based on a set of binding sites
    def generate_motif_from_sites(self, sites, first_order=False):
        if len(self.consensus) != len(sites[0][0]):
            print("Error: consensus length {0} does not match sites length {1}!\nUse set_consensus() first!".format(len(self.consensus),len(sites[0][0])))
            return RuntimeError

        sites_filtered = average_sites(sites)
        # Find the consensus affinity
        affinity_consensus = 0
        for site in sites_filtered:
            seq = site[0]
            if seq == self.consensus:
                affinity_consensus = site[1]
                
        self.n_pos = len(sites[0][0])
        self.PWM = np.ones([self.n_pos, 4])
        self.DPWM = np.ones([self.n_pos-1, 16])
        for site in sites_filtered:
            seq = site[0]
            affinity = site[1]
            if count_mutations(seq, self.consensus) == 1:
                # find mutation in seq
                pos_tmp = 0
                base_tmp = 0
                for pos in range(self.n_pos):
                    if seq[pos] != self.consensus[pos]:
                        pos_tmp = pos
                        base = baseDic[seq[pos]]
                        self.PWM[pos_tmp, base] = affinity_consensus / affinity
                        break   
    
                
        if first_order:
            for site in sites_filtered:
                seq = site[0]
                affinity = site[1]
                if count_mutations(seq, self.consensus) == 2:
                    # find mutation in seq
                    pos_tmp = 0
                    base_tmp = 0
                    for pos in range(self.n_pos-1):
                        if seq[pos] != self.consensus[pos] and seq[pos+1] != self.consensus[pos+1]:
                            pos_tmp = pos
                            base_1 = baseDic[seq[pos]]
                            base_2 = baseDic[seq[pos+1]]
                            kd_predicted = self.score_site(seq, first_order=False)
                            dinuc = 4 * base_1 + base_2
                            self.DPWM[pos_tmp, dinuc] = affinity_consensus / affinity / kd_predicted
                            break        

    # normalize motif so that all entries of PWM sum to 4 and of DPWM to 16
    def normalize_motif(self):
        for row in range(self.n_pos):
            self.PWM[row, :] *= 4 / sum(self.PWM[row, :])
            capped_row = min(row, self.n_pos-2)                    # DPWM is one position shorter
            #self.DPWM[capped_row, :] *= 16 / sum(self.DPWM[capped_row, :])        
        
    # crop motif
    def crop_motif(self):
        self.normalize_motif()
        PWM_cropped = self.PWM
        DPWM_cropped = self.DPWM
        consensus_cropped = self.consensus
        n_pos_cropped = self.n_pos
        for row in range(self.n_pos-1, -1, -1):
            if( np.all(self.PWM[row, :] == 1) and np.all(self.DPWM[max(row-1,0), :] == 1)):
                PWM_cropped = np.delete(PWM_cropped, row, 0)
                DPWM_cropped = np.delete(DPWM_cropped, max(row-1,0), 0)                
                n_pos_cropped -= 1
                consensus_cropped = consensus_cropped[0:row] + consensus_cropped[row+1:]
        self.PWM = PWM_cropped
        self.DPWM = DPWM_cropped
        self.consensus = consensus_cropped
        self.n_pos = n_pos_cropped
    
    def update_motif(self, oligomers, mode, first_order=False):
        sites = []
        for oligomer in oligomers:
            seq_tmp = oligomer[0]
            affinity_tmp = oligomer[1]
            sites_tmp = annotate_sites([seq_tmp], self)
            # Filter the sites for the palindrome mode (only + strand)
            if mode == 1:
                sites_tmp = filter(lambda s: s[2]=='+', sites_tmp)
            weight_sum = sum([s[1] for s in sites_tmp])
            for site in sites_tmp:
                seq_site_tmp = site[0]
                weight_fraction = site[1] / weight_sum
                affinity_site_tmp = 1 / (weight_fraction / affinity_tmp)
                sites.append((seq_site_tmp, affinity_site_tmp))

        self.generate_motif_from_sites(sites, first_order)
    
    def print_PWM(self):
        for pos, row in enumerate(self.PWM):
            row = map(lambda x: round(x/4,5), row)
            string_tmp = ''#str(pos+1)
            for elem in row:
                string_tmp += str(elem) + '\t'
            print string_tmp
        
    def print_DPWM(self):
        for pos, row in enumerate(self.DPWM):
            row = map(lambda x: round(x,5), row)
            string_tmp = '' #str(pos+1)
            for elem in row:
                string_tmp += str(elem)  + '\t'
            print string_tmp
    
    def print_probability_file(self, file_name):
        f_out = open(file_name + '.ihbp', 'w')
        # write first position
        string_0 = ''
        string_1 = ''
        for idx in range(4):
            string_0 +=  str(self.PWM[0, idx]/4) + ' '
            string_1 += (str(self.PWM[0, idx]/4) + ' ') * 4
        f_out.write(string_0 + '\n' + string_1 + '\n\n')
        # write consecutive positions
        for pos in range(1, self.n_pos):
            string_0 = ''
            string_1 = ''
            weights = []            
            for idx in range(4):
                string_0 += str(self.PWM[pos, idx]/4) + ' '
                for idx_2 in range(4):
                    weights.append(self.DPWM[pos-1, 4 * idx + idx_2] * self.PWM[pos-1, idx]/4*self.PWM[pos, idx_2]/4)
            weights /= sum(weights)            
            for elem in weights:
                string_1 += str(elem) + ' '
            f_out.write(string_0 + '\n' + string_1 + '\n\n')
        f_out.close()
        f_out = open(file_name + '.ihbcp', 'w')
        # write first position
        string_0 = ''
        string_1 = ''
        for idx in range(4):
            string_0 +=  str(self.PWM[0, idx]/4) + ' '
            string_1 += (str(self.PWM[0, idx]/4) + ' ') * 4
        f_out.write(string_0 + '\n' + string_1 + '\n\n')
        # write consecutive positions
        for pos in range(1, self.n_pos):
            string_0 = ''
            string_1 = ''
            for idx in range(4):
                string_0 += str(self.PWM[pos, idx]/4) + ' '
                weights = []            
                for idx_2 in range(4):
                    weights.append(self.DPWM[pos-1, 4 * idx + idx_2] * self.PWM[pos, idx_2]/4)
                weights /= sum(weights)            
                for elem in weights:
                    string_1 += str(elem) + ' '
            f_out.write(string_0 + '\n' + string_1 + '\n\n')
        f_out.close()                    

    def calc_information_content(self):
        ic_0 = 0
        
        for pos in range(self.n_pos):
            row = self.PWM[pos, :] / sum(self.PWM[pos, :])
            for base in range(4):
                ic_0 += row[base]*np.log2(row[base]/0.25)

        ic_1 = 0
        for pos in range(self.n_pos-1):
            prob = np.zeros([4, 4])
            cond = np.zeros([4, 4])
            sum_prob = 0
            for base_1 in range(4):
                for base_2 in range(4):
                    prob[base_1, base_2] = self.DPWM[pos, 4*base_1+base_2] * self.PWM[pos,base_1] * self.PWM[pos+1,base_2]    # probability base_1, base_2
                    cond[base_1, base_2] = self.DPWM[pos, 4*base_1+base_2] * self.PWM[pos,base_1]                         # cond. prob. base_2 | base_1
                    sum_prob += prob[base_1, base_2]
            prob /= sum_prob
            for base_1 in range(4):
                cond[base_1, :] /= sum(cond[base_1, :])
                for base_2 in range(4):
                    p_tmp = self.PWM[pos+1,base_2] / sum(self.PWM[pos+1, :])
                    ic_1 += prob[base_1, base_2] * np.log2(cond[base_1, base_2] / p_tmp)
            
        ic = 0
        row0 = self.PWM[0, :] / sum(self.PWM[0, :])
        for base in range(4):
            ic += row0[base] * np.log2(row0[base] / 0.25)
        for pos in range(1, self.n_pos):
            row0 = self.PWM[pos-1, :] / sum(self.PWM[pos-1, :])
            row1 = self.PWM[pos, :] / sum(self.PWM[pos, :])
            prob_list = []
            for base1 in range(4):
                for base2 in range(4):
                    prob_list.append(row0[base1] * row1[base2] * self.DPWM[base1,base2])
            prob_list = prob_list / sum(prob_list)
            for prob in prob_list:
                ic += prob * np.log2(prob * 16)

        return (ic_0, ic-ic_0)
        
                    
# A function that reads the input sequences in the FASTA format and the measured affinities
def read_data_file(data_file):
    data = []
    f_in = open(data_file, 'r')

    for line_tmp in f_in.readlines():
        line_tmp = line_tmp.split(',')
        # Read the affinity
        try:
            affinity = float(line_tmp[-1])        
        except ValueError:
            continue    # affinity is not readable => skip header element 
        # Read sequence 
        line_seq = line_tmp[-2]
        # Check sequence 
        for base in line_seq:
            if(base not in baseDic.keys()):
                print("Error in file {} line {}: {} is not a legal base!".format(data_file, line_tmp, base))
                raise ValueError
        data.append((line_seq, affinity))

    f_in.close()
    return data


# A function that measures the number of mutations between two sequences
def count_mutations(seq_1, seq_2):
    assert(len(seq_1) == len(seq_2))
    mutation_counter = 0
    for pos in range(len(seq_1)):
        if seq_1[pos] != seq_2[pos]:
            mutation_counter += 1
    
    return mutation_counter


# A function to build the reverse complement of a single sequence
def reverse_complement(seq):
    seq_reverse = ''
    for pos in range(len(seq)):
        base = seq[-pos-1]
        base_reverse = ''
        if base in ['A', 'a']:
            base_reverse = 'T'
        elif base in ['C', 'c']:
            base_reverse = 'G'
        elif base in ['G', 'g']:
            base_reverse = 'C'
        elif base in ['T', 't']:
            base_reverse = 'A'
        elif base in ['N', 'n']:
            base_reverse = 'N'
        elif base == '-':
            base_reverse = '-'
        else:
            print("Error in function: reverse_complement! {} is not a legal base".format(base))
            return ValueError
        seq_reverse += base_reverse
    assert(len(seq) == len(seq_reverse))
    return seq_reverse


# A function for reading the sites of a single motif in a sequnce
def predict_affinity(seq, motif, target=False, padding=False):

    sites = annotate_sites([seq], motif, padding)
    if target == False:
        weight_out = sum([s[1] for s in sites])
    else: 
        weight_out = max([s[1] for s in sites])
    return weight_out


def annotate_sites(seqs, motif, padding=False):
    # Check the motif input. If motif is a list, check if the entry looks like a motif
    if type(motif) is list:
        if len(motif) == 1 and isinstance(motif[0], Motif):        
            motif = motif[0]
        else:
            sys.exit("read_sites() takes class Motif as input, instead of a list with {} entries!".format(len(motif)))            

    site_list = []
    for seq in seqs:
        # go through all positions of the plus and reverse strand
        seqLength = len(seq)
        motifLength = motif.n_pos
        Npos = seqLength + motifLength + 1

        # Extend the sequence on both sites with 'N' or '-' if padding is disabled
        if padding:
            seq_padded = 'N' * motifLength + seq + 'N' * motifLength
        else:
            seq_padded = '-' * motifLength + seq + '-' * motifLength
        sites = []
        for pos in range(Npos):
            weight_plus = 1
            weight_reverse = 1
            seq_tmp = seq_padded[pos:pos+motifLength]
            sites.append([seq_tmp, motif.score_site(seq_tmp), '+'])
            sites.append([reverse_complement(seq_tmp), motif.score_reverse_site(seq_tmp), '-'])
    sites_filtered = filter(lambda s: s[1] > 0, sites)    # ignore sites with zero binding weight
    return sites_filtered


# find sites with the same sequence and average their binding strength
def average_sites(sites):
    sites_dic = {}
    # go through all sites
    for site in sites:
        seq = site[0]
        affinity = site[1]
        # is the sequence already known
        if seq not in sites_dic.keys():
            sites_dic.update({seq : [1/affinity]})
        else:
            sites_dic[seq].append(1/affinity)
        
    sites_out = []        
    for site in sites_dic:
        sites_out.append((site, 1/np.mean(sites_dic[site])))

    return sites_out


# The objective function sum of squared errors SSE
def calc_SSE(oligomers, motif, print_results=False):
    n_oligomers = len(oligomers)
    affinity_measured = np.zeros(n_oligomers)
    affinity_predicted = np.zeros(n_oligomers)    
    affinity_predicted_target = np.zeros(n_oligomers)        
    for idx, oligomer in enumerate(oligomers):
        affinity_measured[idx] = 1 / oligomer[1]
        affinity_predicted[idx] = predict_affinity(oligomer[0], motif)
        affinity_predicted_target[idx] = predict_affinity(oligomer[0], motif, target=True)
            
    # calculate the scaling factor for the scale-free SSE
    beta = sum(affinity_measured * affinity_predicted) / sum(affinity_predicted * affinity_predicted)
    SSE = sum((affinity_measured - beta * affinity_predicted)**2) / n_oligomers

    if print_results:
        list_results = []
        scale = max(affinity_measured)
        for idx, oligomer in enumerate(oligomers):
            meas = round(affinity_measured[idx] / scale, 3)
            pred = round(beta*affinity_predicted[idx] / scale,3)
            pred_target = round(beta*affinity_predicted_target[idx] / scale,3)
            list_results.append((oligomer[0], meas, pred, pred_target, meas-pred))
    
        list_sorted =  sorted(list_results, key=lambda el:el[-1])

        print "oligo meas pred pred_max_site diff"
        for elem in list_sorted:
            print elem[0], elem[1], elem[2], elem[3], elem[4]    
            
    return SSE
        

def main(argv=None):
    data_file = ''
    output_file = '.'
    iterations = 4
    mode = 0
    print_results = False
    
    doc_string = 'dinuc_fitter.py -i <input_data> -o <output_file> '
    doc_string_optional = '-s <iterations> -m <mode> -p <print results>'
    
    try:
        opts, args = getopt.getopt(argv,"hi:o:s:m:p",[])
    except getopt.GetoptError:
        print doc_string + doc_string_optional
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print doc_string + doc_string_optional
            sys.exit()
        elif opt in ("-i"):
            data_file = arg
        elif opt in ("-o"):
            output_file = arg
        elif opt in ("-s"):
            iterations = int(arg)
        elif opt in ("-m"):
            mode = int(arg)
        elif opt in ("-p"):
            print_results = True
                        
            
    if mode not in [0,1,2]:
        print("Error: mode {} is not a valid input!\nChoose between 0 (standard mode, default), 1 (palindrome mode), 2 (strict palindrome mode)".format(mode) )        
        sys.exit(1)        

    oligomers_all = read_data_file(data_file)
    oligomers = average_sites(oligomers_all)

    # Initialize the Motif class
    motif = Motif(1) 

    # Determine the consensus
    motif.set_consensus(oligomers)
    # Determine initial motif and crop it to a reasonable range
    motif.generate_motif_from_sites(oligomers)
    motif.crop_motif()

    print 'Iteration 0:', np.sqrt(calc_SSE(oligomers, motif))
    for idx in range(iterations):
        motif.update_motif(oligomers, mode, first_order=False)
        motif.normalize_motif()
        SSE = calc_SSE(oligomers, motif)
        print 'Iteration {}: {}'.format(idx+1, np.sqrt(SSE))
        
    motif.print_probability_file(output_file)

    if print_results:
        # Predict the binding energies
        calc_SSE(oligomers, motif, print_results=True)
        print("Motif")    
        motif.print_PWM()
        motif.print_DPWM() 
        print("Information Content")
        print motif.calc_information_content()

if __name__ == '__main__':
    main(sys.argv[1:])  





