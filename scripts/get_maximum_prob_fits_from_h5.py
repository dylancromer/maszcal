import argparse
import numpy as np
import emcee


def get_max_prob_param(file):
    reader = emcee.backends.HDFBackend(file)
    tau = reader.get_autocorr_time()
    burnin = int(6 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    max_prob_index = reader.get_log_prob(discard=burnin, flat=True, thin=thin).argmax()
    return reader.get_chain(discard=burnin, flat=True, thin=thin)[max_prob_index, :]

def main(files, outfile):
    max_prob_params = np.stack([get_max_prob_param(file) for file in files])
    np.savetxt(outfile, max_prob_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'files',
        type=str,
        nargs='+',
        help='h5 files containing MCMC chains to be summarized',
    )
    parser.add_argument(
        '-o',
        '--output-file',
        type=str,
        help='name of file to output maximum probability params to',
    )
    args = parser.parse_args()
    main(args.files, args.output_file)
