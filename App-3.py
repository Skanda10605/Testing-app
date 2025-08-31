import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency, binomtest

# =========================
# --- BACKEND FUNCTIONS ---
# =========================

def analyze_pairwise_cooccurrence_optimized(df: pd.DataFrame, abnormal_columns: list[str]) -> dict:
    """
    Analyzes co-occurrence patterns for all pairs of binary columns.
    This version uses pd.crosstab for efficient contingency table creation
    and relies on the full output of chi2_contingency for robustness.
    """
    results = {}
    for col1, col2 in combinations(abnormal_columns, 2):
        pair_name = f"{col1.replace('Abnormal ', '')} × {col2.replace('Abnormal ', '')}"
        # Use pd.crosstab to create the 2x2 contingency table (observed frequencies)
        contingency_table = pd.crosstab(df[col1], df[col2])
        # Ensure the table is a full 2x2, filling missing cells with 0
        if contingency_table.shape != (2, 2):
            contingency_table = contingency_table.reindex([0, 1], axis=0, fill_value=0)
            contingency_table = contingency_table.reindex([0, 1], axis=1, fill_value=0)
        
        # Perform chi-square test
        try:
            chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)
        except ValueError:
            # This can happen if a row/column sums to zero.
            chi2, p_value, expected_freq = np.nan, np.nan, np.full((2, 2), np.nan)
        
        observed_freq = contingency_table.values
        difference = observed_freq - expected_freq
        
        # Store results
        results[pair_name] = {
            'combinations': {
                '(0,0) Both Normal': {'observed': observed_freq[0, 0], 'expected': expected_freq[0, 0], 'difference': difference[0, 0]},
                '(0,1) First Normal, Second Abnormal': {'observed': observed_freq[0, 1], 'expected': expected_freq[0, 1], 'difference': difference[0, 1]},
                '(1,0) First Abnormal, Second Normal': {'observed': observed_freq[1, 0], 'expected': expected_freq[1, 0], 'difference': difference[1, 0]},
                '(1,1) Both Abnormal': {'observed': observed_freq[1, 1], 'expected': expected_freq[1, 1], 'difference': difference[1, 1]}
            },
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        }
    return results

def create_abnormality_grouped_bar_chart_optimized(df: pd.DataFrame, metrics: list[str]) -> plt.Figure:
    """
    Creates a bar chart to visualize counts of exclusive abnormality combinations.
    This version is significantly more efficient by vectorizing the combination
    identification process instead of using iterative masks.
    """
    abnormal_cols = [f'Abnormal {metric}' for metric in metrics]
    
    # Create a unique signature for each row's combination of abnormalities
    # This is much faster than iterating through every possible combination
    def create_signature(row):
        present = [metric for i, metric in enumerate(metrics) if row.iloc[i] == 1]
        return " ∩ ".join(present) if present else "All Normal"
    
    combo_signatures = df[abnormal_cols].apply(create_signature, axis=1)
    combo_counts = combo_signatures.value_counts()
    
    # Filter out "All Normal" and take the top combinations
    combo_counts = combo_counts[combo_counts.index != 'All Normal'].head(30)
    
    plot_df = combo_counts.reset_index()
    plot_df.columns = ['Metric Combination', 'Count']
    
    # Plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = sns.barplot(x='Count', y='Metric Combination', data=plot_df.sort_values('Count', ascending=True), ax=ax, palette='viridis', orient='h')
    ax.set_xlabel('Number of Individuals')
    ax.set_ylabel('Abnormality Combination')
    ax.set_title('Frequency of Exclusive Abnormal Health Metric Combinations', fontsize=14, fontweight='bold')
    
    for bar in bars.patches:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f'{int(width)}', ha='left', va='center')
    
    ax.margins(x=0.1)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap_optimized(df: pd.DataFrame, title="Correlation Between Abnormal Metrics") -> plt.Figure:
    """
    Generates a bar chart of pairwise correlations to avoid heatmap layout issues.
    This provides a clear, sorted view of the relationships between metrics.
    """
    corr = df.corr()
    # Unstack the matrix to get a series of all pairs
    corr_pairs = corr.unstack().sort_values(kind="quicksort")
    # Remove self-correlations (where the value is 1.0)
    corr_pairs = corr_pairs[corr_pairs != 1.0]
    
    # Create a unique key for each pair to remove duplicates (e.g., (A,B) and (B,A))
    corr_pairs = corr_pairs.reset_index()
    corr_pairs.columns = ['Metric 1', 'Metric 2', 'Correlation']
    corr_pairs['pair_key'] = corr_pairs.apply(lambda row: tuple(sorted((row['Metric 1'], row['Metric 2']))), axis=1)
    
    # Drop duplicates and create nice labels for the plot
    corr_pairs = corr_pairs.drop_duplicates(subset='pair_key').set_index('pair_key')
    corr_pairs['Pair Label'] = corr_pairs.apply(lambda row: f"{row['Metric 1'].replace('Abnormal ', '')} × {row['Metric 2'].replace('Abnormal ', '')}", axis=1)
    
    # Sort by correlation strength for a clean visual
    corr_pairs = corr_pairs.sort_values('Correlation', ascending=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    
    # Use different colors for positive and negative correlations
    colors = ['#d6604d' if c < 0 else '#4393c3' for c in corr_pairs['Correlation']]
    
    bars = ax.barh(corr_pairs['Pair Label'], corr_pairs['Correlation'], color=colors)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Metric Pair')
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--') # Add a zero line for reference
    
    # Add data labels to the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', ha='left' if width > 0 else 'right')

    ax.margins(x=0.15) # Add some padding
    return fig

def create_cooccurrence_heatmap(pairwise_results):
    """
    Creates a heatmap of the difference between observed and expected frequencies.
    This robust version correctly handles NaN values by not plotting text for them
    and includes rotated labels for better visibility.
    """
    pairs = list(pairwise_results.keys())
    combinations_list = list(next(iter(pairwise_results.values()))['combinations'].keys())
    diff_matrix = np.array([[res['combinations'][combo]['difference'] for combo in combinations_list] for res in pairwise_results.values()])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    # np.nanmax safely ignores nan values when finding the maximum
    max_abs_diff = np.nanmax(np.abs(diff_matrix))
    if not np.isfinite(max_abs_diff) or max_abs_diff == 0:
        max_abs_diff = 1  # Set a default if all values are NaN or zero

    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-max_abs_diff, vmax=max_abs_diff)
    ax.set_xticks(range(len(combinations_list)))
    ax.set_yticks(range(len(pairs)))
    
    ax.set_xticklabels([combo.replace(' ', '\n') for combo in combinations_list], fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(pairs, fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference from Expected\n(Red=More than chance, Blue=Less than chance)', rotation=270, labelpad=25, fontsize=12)
    
    # Robust text annotation loop: checks for valid numbers before drawing them.
    for i in range(len(pairs)):
        for j in range(len(combinations_list)):
            value = diff_matrix[i, j]
            # Only proceed if the value is a finite number (not NaN or infinity)
            if np.isfinite(value):
                # Determine text color based on the background cell color for readability
                color = 'white' if abs(value) > max_abs_diff * 0.5 else 'black'
                # Add the text to the cell
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', color=color, fontweight='bold', fontsize=9)
    
    ax.set_title('Co-occurrence Analysis: Deviations from Chance Expectations', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Combination Type', fontsize=12)
    ax.set_ylabel('Metric Pairs', fontsize=12)
    plt.tight_layout(pad=1.5)
    
    return fig

def create_significance_table(pairwise_results):
    significance_data = [
        {
            'Metric Pair': pair_name,
            'Chi-square': data['chi2'],
            'P-value': data['p_value'],
            'Significant (p<0.05)': 'Yes' if data['significant'] else 'No',
            'Interpretation': 'Associated' if data['significant'] else 'Independent'
        }
        for pair_name, data in pairwise_results.items()
    ]
    return pd.DataFrame(significance_data)

def create_detailed_cooccurrence_table(pairwise_results):
    detailed_data = []
    for pair_name, data in pairwise_results.items():
        for combo_name, stats in data['combinations'].items():
            expected = stats['expected']
            difference = stats['difference']
            pct_diff = (difference / expected * 100) if expected > 0 else 0
            detailed_data.append({
                'Metric Pair': pair_name,
                'Combination': combo_name,
                'Observed': int(stats['observed']),
                'Expected': f"{expected:.1f}",
                'Difference': f"{difference:+.1f}",
                'Percent Difference': f"{pct_diff:+.1f}%",
                'Direction': 'More than chance' if difference > 0 else 'Less than chance' if difference < 0 else 'As expected'
            })
    return pd.DataFrame(detailed_data)

def analyze_all_combinations_cooccurrence(df: pd.DataFrame, abnormal_cols: list[str]):
    """
    Analyzes all EXCLUSIVE combinations of abnormalities to find which co-occur
    more often than expected by chance.
    """
    all_combo_results = []
    total_n = len(df)

    probs = {col: df[col].mean() for col in abnormal_cols}
    qrobs = {col: 1 - p for col, p in probs.items()}

    for k in range(1, len(abnormal_cols) + 1):
        for combo in combinations(abnormal_cols, k):
            combo_list = list(combo)
            other_cols = [col for col in abnormal_cols if col not in combo_list]

            mask_present = (df[combo_list] == 1).all(axis=1)
            if other_cols:
                mask_absent = (df[other_cols] == 0).all(axis=1)
                observed_mask = mask_present & mask_absent
            else:
                observed_mask = mask_present
            observed_count = observed_mask.sum()

            p_present = np.prod([probs[col] for col in combo_list])
            if other_cols:
                p_absent = np.prod([qrobs[col] for col in other_cols])
                p_expected = p_present * p_absent
            else:
                p_expected = p_present
            expected_count = p_expected * total_n
            difference = observed_count - expected_count

            if difference > 0 and expected_count > 0 : # Only test if observed > expected
                binom_result = binomtest(k=observed_count, n=total_n, p=p_expected, alternative='greater')
                p_value = binom_result.pvalue
                combo_name = " × ".join([c.replace('Abnormal ', '') for c in combo_list])
                all_combo_results.append({
                    'Abnormality Combination': combo_name,
                    'Observed Count': observed_count,
                    'Expected Count': expected_count,
                    'Difference': difference,
                    'P-value': p_value
                })

    if not all_combo_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_combo_results).sort_values('Observed Count', ascending=False)
    return results_df

def process_raw_data(pdf, bmi_lb, bmi_ub, press_lb, press_ub, haem_lb, haem_ub, glu_lb, glu_ub, wai_lb, wai_ub):
    pdf.columns = pdf.columns.str.lower()
    required_cols = ['v445', 'sb18s', 'sb25s', 'sb29s', 'v453', 'sb74', 's305']
    if not all(col in pdf.columns for col in required_cols):
        missing = [col for col in required_cols if col not in pdf.columns]
        st.error(f"Missing required columns: {', '.join(missing)}")
        return None
    
    df = pd.DataFrame()
    pdf['Mean systolic reading'] = pdf[['sb18s', 'sb25s', 'sb29s']].mean(axis=1)
    df['Normal waist'] = pdf['s305'].between(wai_lb, wai_ub).astype(int)
    df['Normal bmi'] = pdf['v445'].between(bmi_lb, bmi_ub).astype(int)
    df['Normal systolic pressure'] = pdf['Mean systolic reading'].between(press_lb, press_ub).astype(int)
    df['Normal haemoglobin'] = pdf['v453'].between(haem_lb, haem_ub).astype(int)
    df['Normal glucose'] = pdf['sb74'].between(glu_lb, glu_ub).astype(int)
    
    df['BMI_value'] = pdf['v445']
    df['Systolic_value'] = pdf['Mean systolic reading']
    df['Haemoglobin_value'] = pdf['v453']
    df['Glucose_value'] = pdf['sb74']
    df['Waist_value'] = pdf['s305']
    
    return df.join(pdf)

def analyze_health_data(df):
    abnormality_cols = ['Abnormal bmi', 'Abnormal systolic pressure', 'Abnormal glucose', 'Abnormal haemoglobin', 'Abnormal waist']
    normal_cols = ['Normal bmi', 'Normal systolic pressure', 'Normal glucose', 'Normal haemoglobin', 'Normal waist']
    
    for i, col in enumerate(abnormality_cols):
        df[col] = 1 - df[normal_cols[i]]
    
    df['abnormality_count'] = df[abnormality_cols].sum(axis=1)
    abnormality_count_table = df['abnormality_count'].value_counts().sort_index()
    combo_counts = df[abnormality_cols].value_counts()
    normal_combo_counts = df[normal_cols].value_counts()
    
    return abnormality_count_table, combo_counts, normal_combo_counts, df[abnormality_cols], df

def plot_metric_distribution(df, value_col, metric_name, unit, lb=None, ub=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.histplot(data=df, x=value_col, kde=True, ax=ax, color='#4393c3', edgecolor='white', alpha=0.7)
    
    data_clean = df[value_col].dropna()
    if not data_clean.empty:
        min_val, max_val = data_clean.min(), data_clean.max()
        if lb is not None:
            ax.axvspan(min_val, lb, color='#d6604d', alpha=0.2, label='Abnormal (Low)')
        if ub is not None:
            ax.axvspan(ub, max_val, color='#d6604d', alpha=0.2, label='Abnormal (High)')

    if lb is not None: 
        ax.axvline(lb, color='#b2182b', linestyle='--', label=f'Lower Bound: {lb}')
    if ub is not None: 
        ax.axvline(ub, color='#b2182b', linestyle='--', label=f'Upper Bound: {ub}')
    
    ax.set_title(f'{metric_name} Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f"Value ({unit})", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if lb is not None or ub is not None: 
        ax.legend(loc='upper right')
    
    return fig

# ================================
# --- ENHANCED DIETARY ANALYSIS ---
# ================================

def render_dietary_section(df: pd.DataFrame, comparison_df: pd.DataFrame = None):
    """
    Renders dietary analysis with percentage labels on bars and no emojis.
    """
    dietary_vars = {
        's731a': 'Milk/Curd', 's731b': 'Pulses/Beans', 's731c': 'Dark Green Leafy Veg',
        's731d': 'Fruits', 's731e': 'Eggs', 's731f': 'Fish',
        's731g': 'Chicken/Meat', 's731h': 'Fried Food', 's731i': 'Aerated Drinks'
    }
    freq_labels = {0: 'Never', 1: 'Daily', 2: 'Weekly', 3: 'Occasionally'}
    
    available_vars = {k: v for k, v in dietary_vars.items() if k in df.columns and df[k].notna().any()}
    
    if not available_vars:
        st.warning("No dietary variables (s731a–s731i) found in this dataset.")
        return

    st.markdown(f"**Sample Size**: {len(df)} individuals")
    
    # Helper function to add labels to bars
    def _autolabel_bars(rects, ax_obj):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax_obj.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

    cols = st.columns(3)
    for i, (var, name) in enumerate(available_vars.items()):
        with cols[i % 3]:
            subgroup_counts_raw = df[var].value_counts().sort_index()
            subgroup_counts_normalized = df[var].map(freq_labels).value_counts(normalize=True).reindex(freq_labels.values()).fillna(0)
            
            fig, ax = plt.subplots(figsize=(6.5, 5))
            
            p_value_text = ""
            if comparison_df is not None and not comparison_df.empty and var in comparison_df.columns and comparison_df[var].notna().any():
                comparison_counts_raw = comparison_df[var].value_counts().sort_index()
                comparison_counts_normalized = comparison_df[var].map(freq_labels).value_counts(normalize=True).reindex(freq_labels.values()).fillna(0)

                contingency_table = pd.DataFrame({'Subgroup': subgroup_counts_raw, 'Overall': comparison_counts_raw}).fillna(0)
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1 and contingency_table.sum().sum() > 0:
                    try:
                        chi2, p, dof, ex = chi2_contingency(contingency_table)
                        p_value_text = f" (p={p:.3f})"
                        if p < 0.05: p_value_text += " *"  # Use asterisk for significance
                    except ValueError:
                        p_value_text = " (p=N/A)"

                x = np.arange(len(freq_labels))
                width = 0.35
                rects1 = ax.bar(x - width/2, subgroup_counts_normalized.values * 100, width, label=f'This Group (n={len(df)})', color='#ff7f0e')
                rects2 = ax.bar(x + width/2, comparison_counts_normalized.values * 100, width, label=f'Overall (n={len(comparison_df)})', color='#1f77b4', alpha=0.7)
                
                _autolabel_bars(rects1, ax)
                _autolabel_bars(rects2, ax)

                ax.set_xticks(x)
                ax.set_xticklabels(freq_labels.values(), rotation=45, ha='right')
                ax.legend()
            else:
                bars = sns.barplot(x=subgroup_counts_normalized.index, y=subgroup_counts_normalized.values * 100, palette="muted", ax=ax, order=freq_labels.values())
                _autolabel_bars(bars.patches, ax)
                plt.xticks(rotation=45, ha='right')

            ax.set_title(name + p_value_text, fontsize=14, fontweight='bold')
            ax.set_ylabel("Percent of Individuals (%)", fontsize=10)
            ax.set_ylim(0, 120) 
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# =========================
# --- MAIN APP ---
# =========================

def main():
    st.set_page_config(page_title="Health Metrics + Diet Visualizer", page_icon="📊", layout="wide")
    st.title("Health Metrics + Diet Visualizer")
    st.markdown("Upload your raw NFHS-5 CSV to analyze health metrics, co-occurrence patterns, and dietary habits.")
    
    st.sidebar.header("Health Thresholds")
    bmi_lb = st.sidebar.number_input("BMI Lower Bound (kg/m²)", value=18.5)
    bmi_ub = st.sidebar.number_input("BMI Upper Bound (kg/m²)", value=23)
    press_lb = st.sidebar.number_input("Systolic Pressure Lower Bound (mmHg)", value=90.0)
    press_ub = st.sidebar.number_input("Systolic Pressure Upper Bound (mmHg)", value=120.0)
    haem_lb = st.sidebar.number_input("Haemoglobin Lower Bound (g/dL)", value=12.0)
    haem_ub = st.sidebar.number_input("Haemoglobin Upper Bound (g/dL)", value=15.0)
    glu_lb = st.sidebar.number_input("Glucose Lower Bound (mg/dL)", value=80.0)
    glu_ub = st.sidebar.number_input("Glucose Upper Bound (mg/dL)", value=200.0)
    wai_lb = st.sidebar.number_input("Waist Lower Bound (cm)", value=70.0)
    wai_ub = st.sidebar.number_input("Waist Upper Bound (cm)", value=90.0)
    
    with st.expander("Expected Data Format"):
        st.markdown("""
        Your CSV should contain the following columns (case-insensitive):
        - **Health Metrics**: `v445`, `sb18s`, `sb25s`, `sb29s`, `v453`, `sb74`, `s305`
        - **Dietary Habits**: `s731a`, `s731b`, `s731c`, `s731d`, `s731e`, `s731f`, `s731g`, `s731h`, `s731i`
        """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            pdf = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV with {len(pdf)} rows and {len(pdf.columns)} columns")
            
            with st.expander("Raw Data Preview"):
                st.dataframe(pdf.head())
            
            if st.button("Analyse Data"):
                with st.spinner("Processing health data..."):
                    df = process_raw_data(pdf, bmi_lb, bmi_ub, press_lb, press_ub, haem_lb, haem_ub, glu_lb, glu_ub, wai_lb, wai_ub)
                
                if df is None: st.stop()
                
                st.success("Health indicators created successfully!")
                
                abnormality_count_table, combo_counts, normal_combo_counts, ab_df, processed_df = analyze_health_data(df)
                
                main_tab1, main_tab2 = st.tabs(["Health Profile Analysis", "Dietary Pattern Analysis"])

                with main_tab1:
                    with st.expander("Processed Data Summary", expanded=True):
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Normal BMI", f"{processed_df['Normal bmi'].sum()}/{len(processed_df)}", f"{processed_df['Normal bmi'].mean()*100:.1f}%")
                        col2.metric("Normal BP", f"{processed_df['Normal systolic pressure'].sum()}/{len(processed_df)}", f"{processed_df['Normal systolic pressure'].mean()*100:.1f}%")
                        col3.metric("Normal Hb", f"{processed_df['Normal haemoglobin'].sum()}/{len(processed_df)}", f"{processed_df['Normal haemoglobin'].mean()*100:.1f}%")
                        col4.metric("Normal Glucose", f"{processed_df['Normal glucose'].sum()}/{len(processed_df)}", f"{processed_df['Normal glucose'].mean()*100:.1f}%")
                        col5.metric("Normal Waist", f"{processed_df['Normal waist'].sum()}/{len(processed_df)}", f"{processed_df['Normal waist'].mean()*100:.1f}%")

                    st.header("Overall Health Distribution")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Abnormality Count Distribution")
                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                        abnormality_count_table.plot(kind="bar", ax=ax1, color="skyblue")
                        ax1.set_title("Distribution of Total Abnormalities per Person")
                        ax1.set_xlabel("Number of Abnormalities")
                        ax1.set_ylabel("Number of Individuals")
                        plt.xticks(rotation=0)
                        st.pyplot(fig1)
                        plt.close(fig1)
                        st.dataframe(abnormality_count_table.to_frame(name="Count"))
                    
                    with col2:
                        st.subheader("Correlation Between Abnormal Metrics")
                        fig2 = plot_correlation_heatmap_optimized(ab_df)
                        st.pyplot(fig2)
                        plt.close(fig2)

                    st.header("Metric Combinations Analysis")
                    st.markdown("This chart shows the counts for all single and combined abnormal metric intersections.")
                    metrics = ['bmi', 'systolic pressure', 'glucose', 'haemoglobin', 'waist']
                    fig_grouped = create_abnormality_grouped_bar_chart_optimized(processed_df, metrics)
                    st.pyplot(fig_grouped)
                    plt.close(fig_grouped)
                    
                    st.header("Pairwise Co-occurrence Analysis")
                    abnormal_cols = ['Abnormal bmi', 'Abnormal systolic pressure', 'Abnormal glucose', 'Abnormal haemoglobin', 'Abnormal waist']
                    pairwise_results = analyze_pairwise_cooccurrence_optimized(processed_df, abnormal_cols)
                    
                    st.subheader("Statistical Significance Summary")
                    significance_df = create_significance_table(pairwise_results)
                    st.dataframe(significance_df.set_index('Metric Pair').style.format({'Chi-square': "{:.2f}", 'P-value': "{:.4f}"}))
                    
                    st.subheader("Detailed Deviations from Chance")
                    detailed_df = create_detailed_cooccurrence_table(pairwise_results)
                    st.dataframe(detailed_df.set_index(['Metric Pair', 'Combination']))
                    
                    st.subheader("Deviations Heatmap")
                    fig_cooccurrence = create_cooccurrence_heatmap(pairwise_results)
                    st.pyplot(fig_cooccurrence)
                    plt.close(fig_cooccurrence)
                    
                    st.subheader("Top Exclusive Co-occurring Abnormality Combinations (More than Chance)")
                    cooccurrence_df = analyze_all_combinations_cooccurrence(processed_df, abnormal_cols)
                    if not cooccurrence_df.empty:
                        df_to_display = cooccurrence_df.head(10).copy()
                        df_to_display['Expected Count'] = df_to_display['Expected Count'].map('{:.2f}'.format)
                        df_to_display['Difference'] = df_to_display['Difference'].map('{:+.2f}'.format)
                        st.dataframe(df_to_display.set_index('Abnormality Combination').style.format({'P-value': "{:.4f}"}))
                    else:
                        st.info("No exclusive combinations found to co-occur significantly more than chance.")
                    
                    st.header("Detailed Metric Analysis")
                    metrics_to_plot = [
                        {'value_col': 'BMI_value', 'normal_col': 'bmi', 'name': 'BMI', 'unit': 'kg/m²', 'lb': bmi_lb, 'ub': bmi_ub},
                        {'value_col': 'Systolic_value', 'normal_col': 'systolic pressure', 'name': 'Systolic Pressure', 'unit': 'mmHg', 'lb': press_lb, 'ub': press_ub},
                        {'value_col': 'Haemoglobin_value', 'normal_col': 'haemoglobin', 'name': 'Haemoglobin', 'unit': 'g/dL', 'lb': haem_lb, 'ub': haem_ub},
                        {'value_col': 'Glucose_value', 'normal_col': 'glucose', 'name': 'Glucose', 'unit': 'mg/dL', 'lb': glu_lb, 'ub': glu_ub},
                        {'value_col': 'Waist_value', 'normal_col': 'waist', 'name': 'Waist Circumference', 'unit': 'cm', 'lb': wai_lb, 'ub': wai_ub}
                    ]

                    for metric in metrics_to_plot:
                        with st.container(border=True):
                            st.subheader(f"{metric['name']} Analysis")
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                fig = plot_metric_distribution(processed_df, metric['value_col'], metric['name'], metric['unit'], metric['lb'], metric['ub'])
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            with col2:
                                st.markdown("#### Key Statistics")
                                data = processed_df[metric['value_col']].dropna()
                                abnormal_col_name = f"Abnormal {metric['normal_col']}"
                                abnormal_count = processed_df[abnormal_col_name].sum()
                                total_count = len(processed_df)
                                abnormal_pct = (abnormal_count / total_count) * 100 if total_count > 0 else 0
                                
                                st.metric(label=f"Mean {metric['name']}", value=f"{data.mean():.2f} {metric['unit']}" if not data.empty else "N/A")
                                st.metric(label=f"Median {metric['name']}", value=f"{data.median():.2f} {metric['unit']}" if not data.empty else "N/A")
                                st.metric(
                                    label="Abnormal Count", 
                                    value=f"{abnormal_count} / {total_count}",
                                    help=f"{abnormal_pct:.1f}% of individuals fall outside the normal range."
                                )
                                
                with main_tab2:
                    st.header("Dietary Pattern Analysis")
                    abnormal_cols = ['Abnormal bmi', 'Abnormal systolic pressure', 'Abnormal glucose', 'Abnormal haemoglobin', 'Abnormal waist']
                    metric_ids = ['bmi', 'systolic pressure', 'glucose', 'haemoglobin', 'waist']
                    cooccurrence_df = analyze_all_combinations_cooccurrence(processed_df, abnormal_cols)
                    
                    diet_tab1, diet_tab2, diet_tab3 = st.tabs(["Top Abnormality Groups", "All Participants", "Interactive Explorer"])
                    
                    with diet_tab1:
                        st.subheader("Comparing Top 10 Co-Abnormality Groups")
                        st.info("Dietary profiles for the top groups found to co-occur more than chance. Charts compare the group's diet to the overall population. An asterisk (*) in the title indicates a statistically significant difference (p<0.05).")
                        top10 = cooccurrence_df.head(10)
                        
                        if not top10.empty:
                            for i, row in top10.iterrows():
                                combo_name = row['Abnormality Combination']
                                with st.container(border=True):
                                    st.markdown(f"### {combo_name}")
                                    
                                    abnormal_metrics_in_combo = [f'Abnormal {m.strip()}' for m in combo_name.split(' × ')]
                                    other_metrics = [c for c in abnormal_cols if c not in abnormal_metrics_in_combo]
                                    mask_present = (processed_df[abnormal_metrics_in_combo] == 1).all(axis=1)
                                    if other_metrics:
                                        mask_absent = (processed_df[other_metrics] == 0).all(axis=1)
                                        subgroup_mask = mask_present & mask_absent
                                    else:
                                        subgroup_mask = mask_present
                                    subgroup_df = processed_df.loc[subgroup_mask]
                                    
                                    render_dietary_section(subgroup_df, comparison_df=processed_df)
                        else:
                            st.info("No significant co-occurring abnormality groups found to analyze.")

                    with diet_tab2:
                        st.subheader("Dietary Patterns for the Entire Analyzed Group")
                        render_dietary_section(processed_df)

                    with diet_tab3:
                        st.subheader("Build Your Own Analysis Group")
                        st.markdown("Select one or more abnormalities to see the dietary habits for individuals who have **exactly** that combination and no others.")
                        
                        selected_metrics = st.multiselect(
                            "Select abnormalities for your group:", 
                            options=metric_ids, 
                            default=[]
                        )
                        
                        if selected_metrics:
                            abnormal_selected = [f'Abnormal {m}' for m in selected_metrics]
                            abnormal_not_selected = [f'Abnormal {m}' for m in metric_ids if m not in selected_metrics]
                            
                            mask = (processed_df[abnormal_selected] == 1).all(axis=1)
                            if abnormal_not_selected:
                                mask &= (processed_df[abnormal_not_selected] == 0).all(axis=1)
                            
                            subgroup_df = processed_df.loc[mask]

                            if not subgroup_df.empty:
                                render_dietary_section(subgroup_df, comparison_df=processed_df)
                            else:
                                st.warning("No individuals found with this exact combination of abnormalities.")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()