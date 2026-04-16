import pandas as pd

def load(path, N=10000):
    df = pd.read_csv(path, sep='\t')
    if 'Size' in df.columns:
        df = df[df['Size'] <= N]
    if 'Alpha' in df.columns:
        df = df[df['Alpha'] < 1]
    return df

def main(N=10000):

    runs = {
        'RFAM (alpha=0)': 'outputs/results_rfam_alpha0.log',
        'RFAM (full)':    'outputs/results_rfam_full.log',
    }

    dfs = {}
    for name, path in runs.items():
        try:
            dfs[name] = load(path, N)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    if not dfs:
        print("No valid files loaded.")
        return

    min_rows = min(len(df) for df in dfs.values())

    def row(df):
        return df.iloc[:min_rows]

    # ----- Test Accuracy -----
    print(25 * '-' + "Test Acc" + 25 * '-')
    for name, df in dfs.items():
        if 'Test Acc' in df.columns:
            avg = row(df)['Test Acc'].mean()
            print(f"  {avg:.2f}%  ({min_rows})  {name}")

    # ----- Normal Alignment -----
    print(25 * '-' + "Normal Alignment" + 25 * '-')
    for name, df in dfs.items():
        if 'Normal Alignment' in df.columns:
            avg = row(df)['Normal Alignment'].mean()
            print(f"  {avg:.4f}  ({min_rows})  {name}")

    # ----- Normalized Effective Rank -----
    print(25 * '-' + "Normalized Effective Rank" + 25 * '-')
    for name, df in dfs.items():
        if all(c in df.columns for c in ['Effective Rank', 'NumClasses', 'NumFeatures']):
            trimmed = row(df)
            denom = trimmed[['NumClasses', 'NumFeatures']].min(axis=1)
            avg = (trimmed['Effective Rank'] / denom).mean()
            print(f"  {avg:.4f}  ({min_rows})  {name}")

    # ----- Alpha Distribution (full mode only) -----
    full_key = 'RFAM (full)'
    if full_key in dfs and 'Alpha' in dfs[full_key].columns:
        print(25 * '-' + "Alpha Distribution (full)" + 25 * '-')
        counts = dfs[full_key]['Alpha'].value_counts().sort_index()
        total = counts.sum()
        for alpha, count in counts.items():
            print(f"  alpha={alpha:.3f}  {count:3d} datasets  ({100 * count / total:.1f}%)")

    # ----- Attack Success Rate -----
    robust_cols = []
    for df in dfs.values():
        robust_cols = [c for c in df.columns if "Robust Acc" in c]
        if robust_cols:
            break

    for col in robust_cols:
        print(25 * '-' + f"ASR ({col})" + 25 * '-')
        for name, df in dfs.items():
            if col in df.columns and 'Test Acc' in df.columns:
                trimmed = row(df)
                asr = 100 * (trimmed['Test Acc'] - trimmed[col]) / trimmed['Test Acc']
                print(f"  {asr.mean():.2f}%  ({min_rows})  {name}")

if __name__ == "__main__":
    main()
