import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import glob
from Config_1_Settings import (
    DATA_DIR,
    OUTPUT_DIR,
    PLOT_DATA_DIR,
    CLEAN_FEATURES_CSV,
    ALL_FEATURES,
    TARGET_ATOMIC_FEATURES,
    TARGET_STRUCT_FEATURES
)
from Utils_1_DataLoader import load_clean_data

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "final_verification_data_with_sign.csv")
FIGURE_COMPARISON = os.path.join(OUTPUT_DIR, "fig_ai_vs_dft_comparison.png")
FIGURE_AI_SENSITIVITY = os.path.join(OUTPUT_DIR, "fig_ai_sensitivity.png")
FIGURE_AI_SIGNED_SENSITIVITY = os.path.join(OUTPUT_DIR, "fig_ai_signed_sensitivity.png")

DFT_PATTERNS = [
    os.path.join(OUTPUT_DIR, "dft_micro_adjustment_sensitivity.csv"),
    os.path.join(os.path.dirname(OUTPUT_DIR), "DFT_VAL", "results", "dft_micro_adjustment_sensitivity.csv"),
    "./results/dft_micro_adjustment_sensitivity.csv",
    "../DFT_VAL/results/dft_micro_adjustment_sensitivity.csv"
]
DFT_RESULT_FILE = None
for p in DFT_PATTERNS:
    if os.path.exists(p):
        DFT_RESULT_FILE = p
        break
DELTA_RATIO = 0.01


def calculate_numerical_gradient(model, X, feat_idx):
    x_col = X.iloc[:, feat_idx].values
    eps = np.where(x_col == 0, np.std(x_col) * 0.01, np.abs(x_col) * DELTA_RATIO)
    eps = np.maximum(eps, 1e-6)
    X_plus = X.copy()
    X_minus = X.copy()
    X_plus.iloc[:, feat_idx] = x_col + eps
    X_minus.iloc[:, feat_idx] = x_col - eps
    y_plus = model.predict(X_plus)
    y_minus = model.predict(X_minus)
    return (y_plus - y_minus) / (2 * eps)


def main():
    print(">>> Step 2: AI Sensitivity Analysis with Direct DFT Verification (Signed)...")

    df_orig = load_clean_data()
    print(f"✅ Data loaded successfully: {len(df_orig)} samples.")

    feature_stats = {}
    for feat in ALL_FEATURES:
        feature_stats[feat] = df_orig[feat].std() if feat in df_orig.columns else 1.0

    target_col = 'predicted_bandgap' if 'predicted_bandgap' in df_orig.columns else 'band_gap'
    X = df_orig[ALL_FEATURES]
    y = df_orig[target_col]

    print("\n🚀 Training AI Model and Calculating Sensitivity...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X, y, verbose=False)

    ai_results = []
    for i, feat in enumerate(ALL_FEATURES):
        raw_grads = calculate_numerical_gradient(model, X, i)
        sigma = feature_stats[feat]

        signed_sens = raw_grads * sigma
        abs_sens = np.abs(signed_sens)
        f_type = 'Structural' if feat in TARGET_STRUCT_FEATURES else 'Atomic'

        mean_signed_sens = np.mean(signed_sens)
        mean_abs_sens = np.mean(abs_sens)
        ai_results.append({
            'feature': feat,
            'type': f_type,
            'ai_sensitivity_alpha': mean_abs_sens,
            'ai_signed_sensitivity': mean_signed_sens,
            'ai_std_dev': sigma
        })

        sign_str = "↑ (Positive)" if mean_signed_sens > 0 else "↓ (Negative)"
        print(f" {feat}: Magnitude={mean_abs_sens:.4f} eV, Direction={mean_signed_sens:.4f} eV {sign_str}")

    df_ai = pd.DataFrame(ai_results)

    print("\n🔍 Searching for DFT Verification Data...")
    if DFT_RESULT_FILE:
        print(f" ✅ DFT Result Found: {DFT_RESULT_FILE}")
        try:
            df_dft_raw = pd.read_csv(DFT_RESULT_FILE)

            cols = [c.lower() for c in df_dft_raw.columns]
            type_col = next((c for c in df_dft_raw.columns if c.lower() in ['feature_type', 'type', 'category']), None)
            val_col = next((c for c in df_dft_raw.columns if c.lower() in ['sensitivity_alpha', 'alpha', 'slope']), None)

            if type_col and val_col:
                print(f" Columns detected: Type='{type_col}', Value='{val_col}'")
                print(f" DFT Data Sample:\n{df_dft_raw[[type_col, val_col]].head()}")

                dft_agg_signed = df_dft_raw.groupby(type_col)[val_col].mean().to_dict()
                dft_agg_abs = df_dft_raw[val_col].abs().groupby(df_dft_raw[type_col]).mean().to_dict()
                print(f" DFT Aggregated Results (Signed): {dft_agg_signed}")

                def get_dft_signed(row):
                    f = row['feature']
                    t = row['type']
                    if t == 'Structural':
                        if f == 'bond_length':
                            return dft_agg_signed.get('bond_length', np.nan)
                        if f == 'bond_angle':
                            return dft_agg_signed.get('bond_angle', np.nan)
                    elif t == 'Atomic':
                        return dft_agg_signed.get(f, np.nan)
                    return np.nan

                def get_dft_abs(row):
                    f = row['feature']
                    t = row['type']
                    if t == 'Structural':
                        if f == 'bond_length':
                            return dft_agg_abs.get('bond_length', np.nan)
                        if f == 'bond_angle':
                            return dft_agg_abs.get('bond_angle', np.nan)
                    elif t == 'Atomic':
                        return dft_agg_abs.get(f, np.nan)
                    return np.nan

                df_ai['dft_signed_sensitivity'] = df_ai.apply(get_dft_signed, axis=1)
                df_ai['dft_sensitivity_alpha'] = df_ai.apply(get_dft_abs, axis=1)
                matched_count = df_ai['dft_signed_sensitivity'].notna().sum()
                print(f" ✅ DFT Data Aligned. {matched_count}/{len(df_ai)} features matched.")
            else:
                print(f" ⚠️ Suitable columns not found. Current columns: {list(df_dft_raw.columns)}")
                df_ai['dft_signed_sensitivity'] = np.nan
                df_ai['dft_sensitivity_alpha'] = np.nan
        except Exception as e:
            print(f" ❌ Failed to read or process DFT file: {e}")
            df_ai['dft_signed_sensitivity'] = np.nan
            df_ai['dft_sensitivity_alpha'] = np.nan
    else:
        print(f" ⚠️ DFT result file not found in any preset path.")
        df_ai['dft_signed_sensitivity'] = np.nan
        df_ai['dft_sensitivity_alpha'] = np.nan

    df_ai.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Final verification data (with signs) saved to: {OUTPUT_CSV}")

    valid_df_abs = df_ai.dropna(subset=['dft_sensitivity_alpha'])
    if len(valid_df_abs) > 1:
        print("\n📊 Generating [Absolute Value] Comparison Scatter Plot...")
        plt.figure(figsize=(8, 8))
        x = valid_df_abs['ai_sensitivity_alpha']
        y = valid_df_abs['dft_sensitivity_alpha']
        pr, pp = pearsonr(x, y)
        plt.scatter(x, y, s=100, c='#2E86AB', edgecolors='black', linewidth=1.2, alpha=0.8)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", label=f'Fit: $y={z[0]:.2f}x+{z[1]:.2f}$')
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k:', label='Perfect Agreement', linewidth=1)
        plt.title(f'AI vs DFT Sensitivity (Magnitude)\nPearson $r$={pr:.2f}', fontsize=14, fontweight='bold')
        plt.xlabel('AI Sensitivity Magnitude (eV)', fontsize=12)
        plt.ylabel('DFT Sensitivity Magnitude (eV)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(FIGURE_COMPARISON, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Absolute value comparison chart saved: {FIGURE_COMPARISON}")

    valid_df_sign = df_ai.dropna(subset=['dft_signed_sensitivity'])
    if len(valid_df_sign) > 1:
        print("\n📊 Generating [Signed] Comparison Scatter Plot (Positive/Negative Correlation)...")
        plt.figure(figsize=(8, 8))
        x = valid_df_sign['ai_signed_sensitivity']
        y = valid_df_sign['dft_signed_sensitivity']
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        plt.scatter(x, y, s=100, c='#D7191C' if pr > 0 else '#2E86AB', edgecolors='black', linewidth=1.2, alpha=0.8)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "k--", label=f'Fit: $y={z[0]:.2f}x+{z[1]:.2f}$')
        plt.axhline(0, color='gray', linewidth=1)
        plt.axvline(0, color='gray', linewidth=1)
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        plt.title(f'AI vs DFT Sensitivity (Direction/Signed)\nPearson $r$={pr:.2f}, Spearman $\\rho$={sr:.2f}', fontsize=14, fontweight='bold')
        plt.xlabel('AI Signed Sensitivity (eV)\n(Positive = Increases Band Gap)', fontsize=12)
        plt.ylabel('DFT Signed Sensitivity (eV)\n(Positive = Increases Band Gap)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        fig_signed_path = os.path.join(OUTPUT_DIR, "fig_ai_vs_dft_signed_comparison.png")
        plt.savefig(fig_signed_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Signed comparison chart saved: {fig_signed_path}")

        print(f"\n🏆 Direction Correlation Analysis:")
        print(f" Pearson r = {pr:.4f} (p = {pp:.4f})")
        if pr > 0.7 and pp < 0.05:
            print(" ✅ Conclusion: AI and DFT are highly consistent in direction!")
        elif pr > 0.4:
            print(" ⚠️ Conclusion: Moderate directional consistency exists.")
        else:
            print(" ⚠️ Conclusion: Low directional consistency. Check physical mechanisms.")

    plt.figure(figsize=(10, 6))
    df_sorted = df_ai.sort_values('ai_signed_sensitivity', ascending=True)
    colors = ['#D7191C' if x > 0 else '#2E86AB' for x in df_sorted['ai_signed_sensitivity']]
    plt.barh(df_sorted['feature'], df_sorted['ai_signed_sensitivity'], color=colors, edgecolor='black', linewidth=1.2)
    plt.axvline(0, color='black', linewidth=1.5)
    plt.title('AI Signed Feature Sensitivity\n(Red: Positive Correlation, Blue: Negative Correlation)', fontsize=14, fontweight='bold')
    plt.xlabel('Signed Sensitivity (eV per Std Dev)\n(>0 means increasing feature increases band gap)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURE_AI_SIGNED_SENSITIVITY, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Signed Sensitivity Bar Chart saved: {FIGURE_AI_SIGNED_SENSITIVITY}")

    print("\n🎉 Step 2 Completed! All charts and data containing signs have been generated.")


if __name__ == "__main__":
    main()