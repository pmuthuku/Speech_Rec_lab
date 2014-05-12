import numpy as np

__author__ = 'anoop'

mfcc_dir = 'cont_rec'
generated_models_dir = 'generated_models/'
hmm_list = [0,1,2,3,4,5,6,7,8,9]
state_nos = [0,1,2,3,4]

trans_list = [np.zeros((len(state_nos)*2, 39)) for _ in range(len(hmm_list))]

def main():

    for hmm_no in hmm_list:
        for state_no in state_nos:
            file_name = "{0}/{1}_{2}.mfcc".format(mfcc_dir, hmm_no, state_no)
            data = np.loadtxt(file_name)
            trans_list[hmm_no][state_no*2, :] = np.mean(data,0)
            trans_list[hmm_no][state_no*2 + 1, :] = np.diag(np.cov(data, rowvar=0))
        np.savetxt(generated_models_dir + str(hmm_no)+'.hmm', trans_list[hmm_no])

if __name__ == '__main__':
    main()