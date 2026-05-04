import pandas as pd
from scipy import stats

def main():

    alpha1_key = 'RFAM (alpha=1)'
    full_key   = 'RFAM (full)'

    try:
        df_full = pd.read_csv('outputs/results_rfam_full.log', sep='\t')
    except Exception as e:
        print(f"Could not load full results: {e}")
        return

    try:
        df_alpha1 = pd.read_csv('outputs/results_rfam_alpha1.log', sep='\t')
    except Exception as e:
        print(f"Could not load alpha=1 results: {e}")
        return

    # Use positional rows in full where Alpha < 1 to filter both dataframes
    if 'Alpha' in df_full.columns:
        mask = df_full['Alpha'] < 1
    else:
        mask = pd.Series([True] * len(df_full))

    df_full   = df_full[mask].reset_index(drop=True)
    df_alpha1 = df_alpha1[mask].reset_index(drop=True)

    n = len(df_full)

    dfs = {alpha1_key: df_alpha1, full_key: df_full}

    def pval_str(a, b, alternative='two-sided'):
        _, p = stats.ttest_rel(a, b, nan_policy='omit', alternative=alternative)
        return f"p={p:.3f}" if p >= 0.001 else "p<0.001"

    def diff_line(full_vals, alpha1_vals, alternative='two-sided'):
        a, b = alpha1_vals.values, full_vals.values
        rel = (b - a) / pd.Series(a).abs().values
        s = pd.Series(rel)
        # alternative is framed as ttest_rel(alpha1, full), so:
        #   'less'    → H1: alpha1 < full  (full is higher, good for acc/alignment)
        #   'greater' → H1: alpha1 > full  (full is lower,  good for rank/ASR)
        print(f"  rel diff (full-alpha1)/|alpha1|: mean={s.mean():.4f} ± {s.std():.4f}  {pval_str(a, b, alternative)}  (n={s.notna().sum()})")

    # ----- Test Accuracy -----
    print(25 * '-' + "Test Acc" + 25 * '-')
    for name, df in dfs.items():
        if 'Test Acc' in df.columns:
            v = df['Test Acc']
            print(f"  {v.mean():.2f}% ± {v.std():.2f}%  ({n})  {name}")
    diff_line(df_full['Test Acc'], df_alpha1['Test Acc'], alternative='less')

    # ----- Normal Alignment -----
    if 'Normal Alignment' in df_full.columns:
        print(25 * '-' + "Normal Alignment" + 25 * '-')
        for name, df in dfs.items():
            if 'Normal Alignment' in df.columns:
                v = df['Normal Alignment']
                print(f"  {v.mean():.4f} ± {v.std():.4f}  ({n})  {name}")
        diff_line(df_full['Normal Alignment'], df_alpha1['Normal Alignment'], alternative='less')

    # ----- Normalized Effective Rank -----
    er_cols = ['Effective Rank', 'NumClasses', 'NumFeatures']
    if all(c in df_full.columns for c in er_cols):
        print(25 * '-' + "Normalized Effective Rank" + 25 * '-')
        for name, df in dfs.items():
            if all(c in df.columns for c in er_cols):
                v = df['Effective Rank'] / df[['NumClasses', 'NumFeatures']].min(axis=1)
                print(f"  {v.mean():.4f} ± {v.std():.4f}  ({n})  {name}")
        ner_f = df_full['Effective Rank'] / df_full[['NumClasses', 'NumFeatures']].min(axis=1)
        ner_a = df_alpha1['Effective Rank'] / df_alpha1[['NumClasses', 'NumFeatures']].min(axis=1)
        diff_line(ner_f, ner_a, alternative='greater')

    # ----- Alpha Distribution (full mode only) -----
    if 'Alpha' in df_full.columns:
        print(25 * '-' + "Alpha Distribution (full)" + 25 * '-')
        counts = df_full['Alpha'].value_counts().sort_index()
        total  = counts.sum()
        for alpha, count in counts.items():
            print(f"  alpha={alpha:.3f}  {count:3d} datasets  ({100 * count / total:.1f}%)")

    # ----- Attack Success Rate -----
    robust_cols = [c for c in df_full.columns if "Robust Acc" in c]
    for col in robust_cols:
        if col not in df_alpha1.columns:
            continue
        print(25 * '-' + f"ASR ({col})" + 25 * '-')
        for name, df in dfs.items():
            if col in df.columns and 'Test Acc' in df.columns:
                asr = 100 * (df['Test Acc'] - df[col]) / df['Test Acc']
                print(f"  {asr.mean():.2f}% ± {asr.std():.2f}%  ({n})  {name}")
        asr_f = 100 * (df_full['Test Acc'] - df_full[col]) / df_full['Test Acc']
        asr_a = 100 * (df_alpha1['Test Acc'] - df_alpha1[col]) / df_alpha1['Test Acc']
        d = pd.Series(asr_f.values - asr_a.values)
        print(f"  abs diff (full-alpha1): mean={d.mean():.2f}pp ± {d.std():.2f}pp  {pval_str(asr_a.values, asr_f.values, alternative='greater')}  (n={d.notna().sum()})")

    # ----- Runtime -----
    rt_col = 'Runtime (s)'
    if rt_col in df_full.columns or rt_col in df_alpha1.columns:
        print(25 * '-' + "Runtime (s)" + 25 * '-')
        for name, df in dfs.items():
            if rt_col in df.columns:
                v = df[rt_col]
                print(f"  {name}: total={v.sum():.1f}s  mean={v.mean():.1f}s ± {v.std():.1f}s  ({n})")
        if rt_col in df_full.columns and rt_col in df_alpha1.columns:
            diff_line(df_full[rt_col], df_alpha1[rt_col])

if __name__ == "__main__":
    main()
