import argparse
import os
import numpy as np
import utils.rfm as rfm

parser = argparse.ArgumentParser()
parser.add_argument('-datadir', default="data", type=str, help="data directory")
parser.add_argument('-outdir', default="outputs", type=str, help="data directory")
parser.add_argument('-mode', default="full", choices=["alpha0", "full"],
                    help="alpha0: fix alpha=0 (pure covariance init); full: search over all alphas")

args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

avg_acc_list = []

max_iter = 5
regs = [10, 1, .1, 1e-2, 1e-3]
normalize = [False]
alphas = [0.0] if args.mode == "alpha0" else [0.0, 1e-3, 1e-2, 1e-1, 1.0]
epsilons = [0.1, 1.0, 2.0]

outf = open(f'{args.outdir}/results_rfam_{args.mode}.log', "w")

robust_headers = "\t".join([f"Robust Acc (eps={e})" for e in epsilons])
print(
    f"Dataset\tSize\tNumFeatures\tNumClasses\tAlpha\tTest Acc\t{robust_headers}\tEffective Rank\tNormal Alignment",
    file=outf
)

for idx, dataset in enumerate(sorted(os.listdir(args.datadir))):

    if not os.path.isdir(args.datadir + "/" + dataset):
        continue
    if not os.path.isfile(args.datadir + "/" + dataset + "/" + dataset + ".txt"):
        continue

    dic = dict()
    for k, v in map(lambda x: x.split(), open(args.datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v

    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])

    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])

    n_tot = n_train_val + n_test

    print(idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)

    with open(os.path.join(args.datadir, dataset, dic["fich1="]), "r") as data_file:
        f = data_file.readlines()[1:]

    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))

    fold = list(map(lambda x: list(map(int, x.split())),
                    open(args.datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))

    train_fold, val_fold = fold[0], fold[1]

    best_acc, best_reg, best_alpha, best_iter, best_M = 0, 0, 0, 0, 0
    best_normalize = False

    print("Cross Validating")

    for reg in regs:
        for alpha in alphas:
            for n in normalize:
                if dataset == 'balance-scale':
                    n = False

                acc, iter_v, M = rfm.hyperparam_train(
                    X[train_fold], y[train_fold],
                    X[val_fold], y[val_fold], c,
                    iters=max_iter, reg=reg,
                    normalize=n, alpha=alpha
                )

                if acc > best_acc:
                    best_acc = acc
                    best_reg = reg
                    best_iter = iter_v
                    best_M = M
                    best_normalize = n
                    best_alpha = alpha

    avg_acc = 0.0
    avg_rob_accs = {e: 0.0 for e in epsilons}
    avg_alignments = []
    avg_ers = []

    with open(os.path.join(args.datadir, dataset, "conxuntos_kfold.dat"), "r") as kfold_file:
        fold = [list(map(int, line.split())) for line in kfold_file]

    print(f"Training - {best_alpha}")

    for repeat in range(4):

        train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]

        acc, rob_accs, alignment, er = rfm.train(
            X[train_fold], y[train_fold],
            X[test_fold], y[test_fold],
            c, best_M,
            iters=best_iter,
            reg=best_reg,
            normalize=best_normalize,
            epsilons=epsilons
        )

        avg_acc += 0.25 * acc

        for e in epsilons:
            avg_rob_accs[e] += 0.25 * rob_accs[e]

        avg_alignments.append(alignment)
        avg_ers.append(er)

    final_alignment = np.mean(avg_alignments)
    final_er = np.mean(avg_ers)

    print(f"acc: {avg_acc:.4f} normalize: {best_normalize} alignment: {final_alignment:.4f} er: {final_er:.4f}\n")

    robust_values = "\t".join([f"{avg_rob_accs[e] * 100:.2f}" for e in epsilons])

    print(
        f"{dataset}\t{n_tot}\t{d}\t{c}\t{best_alpha}\t"
        f"{avg_acc * 100:.2f}\t{robust_values}\t"
        f"{final_alignment:.4f}\t{final_er:.4f}",
        file=outf,
        flush=True
    )

    avg_acc_list.append(avg_acc)

print("avg_acc:", np.mean(avg_acc_list) * 100)
outf.close()
