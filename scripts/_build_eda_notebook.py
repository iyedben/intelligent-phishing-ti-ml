from pathlib import Path
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# PhiUSIIL Phishing URL Dataset: Exploratory Data Analysis

This reproducible notebook analyzes the verified UCI CSV without modifying it. It does not train a model, select features, resample, normalize, or prepare splits. UCI labels remain unchanged: `0 = phishing`, `1 = legitimate`. The eventual deployment detector receives a URL and must not fetch or execute the destination, so feature availability is assessed separately from observed discrimination.''')
md('''## 1. Imports and reproducibility

Full-data summaries use all rows. Dense plots use an explicitly stratified, fixed-seed sample and do not write sampled data.''')
code('''from pathlib import Path
import os, tempfile
ROOT = Path.cwd()
if not (ROOT / "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv").exists(): ROOT = ROOT.parent
DATA_PATH = ROOT / "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
assert DATA_PATH.is_file(), DATA_PATH
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phiusiil-matplotlib"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import mannwhitneyu
from IPython.display import display
SEED = 2026
pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 100); pd.set_option("display.max_colwidth", 110)
sns.set_theme(style="whitegrid", context="notebook")
df = pd.read_csv(DATA_PATH)
print("pandas", pd.__version__, "|", DATA_PATH)
''')
md('''## 2. Dataset overview and quality

Read the source CSV directly. No labels, rows, or columns are changed.''')
code('''print(f"Shape: {df.shape[0]:,} × {df.shape[1]:,}")
print(f"Deep in-memory usage: {df.memory_usage(deep=True).sum():,} bytes")
overview = pd.DataFrame({"dtype": df.dtypes.astype(str), "unique": df.nunique(dropna=False),
                         "missing": df.isna().sum(), "missing_pct": (df.isna().mean()*100).round(4)})
display(overview)
print("Fully duplicate rows:", int(df.duplicated().sum()))
if "URL" in df:
    print("Duplicate URL occurrences beyond first:", int(df.URL.duplicated().sum()))
    print("Distinct URL values appearing more than once:", int(df.loc[df.URL.duplicated(keep=False), "URL"].nunique()))
print("Constant columns:", [c for c in df if df[c].nunique(dropna=False) <= 1])
''')
md('''### Target counts and percentages

Labels are shown as stored: `0 = phishing`; `1 = legitimate`.''')
code('''TARGET = "label"
LABEL_MEANING = {0: "Phishing", 1: "Legitimate"}
counts = df[TARGET].value_counts().sort_index()
class_summary = pd.DataFrame({"meaning": [LABEL_MEANING[int(i)] for i in counts.index],
                              "count": counts, "percent": (counts/len(df)*100).round(4)})
display(class_summary)
ax = sns.barplot(x=class_summary.index.astype(str), y=class_summary["count"], color="#3977a8")
ax.set(title="Target class distribution", xlabel="Source label (0 = phishing, 1 = legitimate)", ylabel="Rows")
for bar, n in zip(ax.patches, class_summary["count"]):
    ax.annotate(f"{n:,}", (bar.get_x()+bar.get_width()/2, bar.get_height()), ha="center", va="bottom", xytext=(0,4), textcoords="offset points")
plt.tight_layout(); plt.show()
''')
md('''## 3. Feature inventory and availability

UCI marks `FILENAME` as Other and `label` as target. UCI's per-field descriptions are blank, so inferred categories below use the column names and its dataset-level statement that URL and webpage/source-code properties are present. `YES` means obtainable from the supplied URL string only, without DNS, HTTP, downloads, JavaScript, or rendering. `UNCERTAIN` means the published field definition/formula is insufficient. Domain/TLD fields are identified separately. Lists are methodological candidates, not final feature selection.''')
code('''URL_ONLY_FEATURES = ["URL", "URLLength", "Domain", "DomainLength", "IsDomainIP", "HasObfuscation",
 "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL",
 "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
 "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS"]
DOMAIN_TLD_FEATURES = ["Domain", "DomainLength", "IsDomainIP", "TLD", "TLDLength", "NoOfSubDomain", "TLDLegitimateProb"]
POST_FETCH_FEATURES = ["LineOfCode", "LargestLineLength", "HasTitle", "Title", "DomainTitleMatchScore", "URLTitleMatchScore",
 "HasFavicon", "Robots", "IsResponsive", "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
 "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField",
 "Bank", "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef"]
METADATA_FIELDS = ["FILENAME"]
availability, category, leakage, requirement = {}, {}, {}, {}
for c in URL_ONLY_FEATURES:
    availability[c] = "YES"; category[c] = "URL string/parser"
    requirement[c] = "Parse supplied URL only; no target contact."
    leakage[c] = "Derived/overlapping URL counts or ratios; raw URL can memorize exact samples."
domain_yes = {"Domain":"Host parsing; no DNS.", "DomainLength":"Length of parsed host.", "IsDomainIP":"Syntactic IP-literal check; do not resolve DNS."}
domain_uncertain = {"TLD":"Public-suffix extraction convention undocumented; observed anomalies need review.",
 "TLDLength":"Depends on exact suffix extraction convention.", "NoOfSubDomain":"Registered-domain/public-suffix rule is undocumented.",
 "TLDLegitimateProb":"Requires reference population/lookup; formula/provenance undocumented and could encode source/labels."}
derived_uncertain = {"URLSimilarityIndex":"Reference set/formula undocumented; could encode target-informed or source-specific information.",
 "CharContinuationRate":"UCI calls it derived but does not provide its formula.",
 "URLCharProb":"Probability reference/formula undocumented; may require corpus statistics."}
for c, why in {**domain_yes, **domain_uncertain}.items():
    category[c] = "Domain/TLD"
    availability[c] = "YES" if c in domain_yes else "UNCERTAIN"
    requirement[c] = why; leakage[c] = "Site identity / correlated parsed fields; " + why
for c, why in derived_uncertain.items():
    category[c] = "Derived URL statistic (definition unclear)"; availability[c] = "UNCERTAIN"
    requirement[c] = why; leakage[c] = why
post_why = {
 "LineOfCode":"Fetched HTML/source.", "LargestLineLength":"Fetched HTML/source.", "HasTitle":"Parse response document.",
 "Title":"Page title/content; high-cardinality site identifier.", "DomainTitleMatchScore":"Page title plus URL/domain.",
 "URLTitleMatchScore":"Page title plus URL; formula undocumented.", "HasFavicon":"Inspect markup or request favicon.",
 "Robots":"Robots.txt/network response or crawl observation; exact encoding unclear.", "IsResponsive":"Page markup/rendering assessment.",
 "NoOfURLRedirect":"Request and observe redirect chain.", "NoOfSelfRedirect":"Requests and compare redirect destinations.",
 "HasDescription":"Parse page metadata.", "NoOfPopup":"Page behavior/script analysis or rendering.", "NoOfiFrame":"Parse HTML.",
 "HasExternalFormSubmit":"Parse forms and destinations.", "HasSocialNet":"Inspect page content/links.", "HasSubmitButton":"Parse page controls/content.",
 "HasHiddenFields":"Parse form/HTML fields.", "HasPasswordField":"Parse form inputs.", "Bank":"Likely page-content category; UCI definition absent.",
 "Pay":"Likely page-content category; UCI definition absent.", "Crypto":"Likely page-content category; UCI definition absent.",
 "HasCopyrightInfo":"Inspect page text/source.", "NoOfImage":"Parse markup/resources.", "NoOfCSS":"Parse markup/resources.",
 "NoOfJS":"Parse markup/resources.", "NoOfSelfRef":"Parse links/resources and compare origins.",
 "NoOfEmptyRef":"Parse links/resources.", "NoOfExternalRef":"Parse links/resources and compare origins."}
for c in POST_FETCH_FEATURES:
    category[c]="Webpage/content/post-fetch"; availability[c]="NO"; requirement[c]=post_why[c]
    leakage[c]=post_why[c] + " Deployment mismatch for URL-only track."
for c in METADATA_FIELDS:
    category[c]="Metadata/identifier"; availability[c]="NO"; requirement[c]="Dataset row identifier; not derivable from submitted URL."
    leakage[c]="UCI says it can be ignored; unique row ID may encode acquisition order/source."
category[TARGET]="Target label"; availability[TARGET]="NO"; requirement[TARGET]="Outcome label, never an inference input."
leakage[TARGET]="Direct target leakage; never a predictor."
assert set(category)==set(df.columns), set(df.columns)-set(category)
inventory = pd.DataFrame({"feature_name":df.columns, "data_type":[str(df[c].dtype) for c in df],
 "unique_values":[int(df[c].nunique(dropna=False)) for c in df], "missing_count":[int(df[c].isna().sum()) for c in df],
 "likely_feature_category":[category[c] for c in df], "URL_only_availability":[availability[c] for c in df],
 "post_fetch_requirement_or_reason":[requirement[c] for c in df], "potential_leakage_concern":[leakage[c] for c in df]})
display(inventory)
print("URL_ONLY_FEATURES:", URL_ONLY_FEATURES)
print("POST_FETCH_FEATURES:", POST_FETCH_FEATURES)
print("UNCERTAIN:", [c for c in df if availability[c]=="UNCERTAIN"])
''')
md('''## 4. Numerical summaries, skew, and IQR outlier screen

Statistics use all rows. IQR flags (`< Q1−1.5×IQR` or `> Q3+1.5×IQR`) are descriptive only, especially limited for binary/count fields; no records are removed.''')
code('''numeric_features = [c for c in df.select_dtypes(include=np.number).columns if c != TARGET]
numeric_desc = df[numeric_features].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
numeric_desc["skew"] = df[numeric_features].skew(numeric_only=True)
display(numeric_desc)
class_stats = df.groupby(TARGET)[numeric_features].agg(["mean","median","std"])
display(class_stats.T)
q1, q3 = df[numeric_features].quantile(.25), df[numeric_features].quantile(.75)
iqr = q3-q1
flag = df[numeric_features].lt(q1-1.5*iqr) | df[numeric_features].gt(q3+1.5*iqr)
outlier_summary = pd.DataFrame({"iqr_flag_count":flag.sum().astype(int), "iqr_flag_pct":(flag.mean()*100).round(3),
                                "skew":df[numeric_features].skew().round(3)}).sort_values("iqr_flag_pct",ascending=False)
display(outlier_summary.head(20))
''')
md('''## 5. Class-conditional discrimination and selected distributions

Full data is used for rank-based univariate AUC. It is `P(phishing value > legitimate value)`; 0.5 indicates no rank separation and direction is retained. It is not model performance or final feature importance. Plots use a fixed-seed sample of at most 8,000 per class; this stratified sample is for readability only. Selected nonnegative count axes use log1p for display only.''')
code('''phish, legit = df[df[TARGET]==0], df[df[TARGET]==1]
auc_rows=[]
for c in numeric_features:
    if df[c].nunique(dropna=True)>1:
        u=mannwhitneyu(phish[c].dropna(),legit[c].dropna(),alternative="two-sided",method="asymptotic").statistic
        a=float(u/(phish[c].notna().sum()*legit[c].notna().sum()))
        auc_rows.append({"feature":c,"phishing_higher_value_auc":a,"abs_distance_from_0_5":abs(a-.5),"cliffs_delta":2*a-1,
         "phishing_median":phish[c].median(),"legitimate_median":legit[c].median(),
         "phish_q25":phish[c].quantile(.25),"phish_q75":phish[c].quantile(.75),
         "legit_q25":legit[c].quantile(.25),"legit_q75":legit[c].quantile(.75)})
univariate=pd.DataFrame(auc_rows).sort_values("abs_distance_from_0_5",ascending=False)
display(univariate.head(25))
plot_df=pd.concat([g.sample(n=min(8000,len(g)),random_state=2026) for _,g in df.groupby(TARGET,sort=True)],ignore_index=True)
print(f"Plot sample: {len(plot_df):,} rows, up to 8,000/class, random_state=2026")
preferred=["URLLength","DomainLength","NoOfSubDomain","URLSimilarityIndex","TLDLegitimateProb","NoOfObfuscatedChar",
 "IsHTTPS","LineOfCode","LargestLineLength","DomainTitleMatchScore","NoOfURLRedirect","NoOfExternalRef"]
selected=[c for c in preferred if c in numeric_features]
fig,axes=plt.subplots(4,3,figsize=(16,15))
log_fields={"URLLength","DomainLength","NoOfSubDomain","NoOfObfuscatedChar","LineOfCode","LargestLineLength","NoOfURLRedirect","NoOfExternalRef"}
for ax,c in zip(axes.flat,selected):
    z=plot_df[[TARGET,c]].copy(); use_log=c in log_fields and z[c].min()>=0
    z["display_value"]=np.log1p(z[c]) if use_log else z[c]
    sns.violinplot(data=z,x=TARGET,y="display_value",hue=TARGET,legend=False,inner="quartile",cut=0,density_norm="width",ax=ax)
    ax.set(title=c+(" (log1p axis)" if use_log else ""),xlabel="Source label (0 = phishing, 1 = legitimate)",ylabel="log1p(value)" if use_log else c)
for ax in axes.flat[len(selected):]: ax.remove()
fig.suptitle("Selected numeric distributions by class (stratified plot sample)",y=1.01)
plt.tight_layout(); plt.show()
''')
md('''## 6. Categorical columns and cardinality

Analyze Domain, TLD, Title, and FILENAME without one-hot encoding. Singleton rates and repeated-group label mixing help expose memorization risks; top-category target composition is descriptive and does not establish generalization.''')
code('''focus=[c for c in ["Domain","TLD","Title","FILENAME"] if c in df]
cat_rows=[]; top_cats={}
notes={"Domain":"High-cardinality site identifier; investigate grouped evaluation.",
 "TLD":"Moderate-cardinality; suffix convention and observed parsing anomalies need review.",
 "Title":"High-cardinality page content; post-fetch and site/template memorization risk.",
 "FILENAME":"Unique row identifier; UCI says it can be ignored; do not use as predictor."}
for c in focus:
    vc=df[c].value_counts(dropna=False); label_n=df.groupby(c,dropna=False)[TARGET].nunique()
    cat_rows.append({"feature":c,"unique_values":df[c].nunique(dropna=False),"unique_ratio":df[c].nunique(dropna=False)/len(df),
      "singleton_rows_pct":(vc==1).sum()/len(df)*100,"max_category_rows":vc.max(),"repeated_categories":(vc>1).sum(),
      "repeated_groups_with_both_labels":(label_n>1).sum(),"suitability":notes[c]})
    top_cats[c]=vc.head(12)
display(pd.DataFrame(cat_rows).set_index("feature"))
for c in focus:
    print("Top categories:",c); display(top_cats[c].rename("count").to_frame())
if "TLD" in df:
    order=df.TLD.value_counts().head(12).index.tolist()
    counts_tld=df[df.TLD.isin(order)].groupby(["TLD",TARGET]).size().rename("rows").reset_index()
    plt.figure(figsize=(13,5)); sns.barplot(data=counts_tld,x="TLD",y="rows",hue=TARGET,order=order)
    plt.title("Class counts for 12 most frequent observed TLD values"); plt.xlabel("Observed TLD value"); plt.ylabel("Rows")
    plt.xticks(rotation=35,ha="right"); plt.legend(title="Source label (0 = phishing, 1 = legitimate)")
    plt.tight_layout(); plt.show()
''')
md('''## 7. Correlation and redundancy

Spearman rank correlation is suitable for this mixture of skewed counts, bounded ratios, and binary fields. The heatmap is a curated subset; a scan of the full numerical set identifies strong pairs. The label is excluded. Correlation flags possible redundancy/multicollinearity, not a removal instruction.''')
code('''heat_features=[c for c in ["URLLength","DomainLength","IsDomainIP","NoOfSubDomain","TLDLength","URLSimilarityIndex",
 "CharContinuationRate","TLDLegitimateProb","URLCharProb","NoOfObfuscatedChar","ObfuscationRatio","NoOfLettersInURL",
 "NoOfDegitsInURL","SpacialCharRatioInURL","IsHTTPS","LineOfCode","LargestLineLength","DomainTitleMatchScore",
 "URLTitleMatchScore","NoOfURLRedirect","NoOfSelfRedirect","NoOfImage","NoOfCSS","NoOfJS","NoOfSelfRef","NoOfExternalRef"] if c in numeric_features]
plt.figure(figsize=(16,13)); sns.heatmap(df[heat_features].corr(method="spearman"),cmap="vlag",center=0,vmin=-1,vmax=1,square=True,
 linewidths=.25,cbar_kws={"label":"Spearman rank correlation"})
plt.title("Spearman correlations for selected numerical features"); plt.tight_layout(); plt.show()
full_corr=df[numeric_features].corr(method="spearman")
pairs=[(a,b,float(full_corr.loc[a,b])) for i,a in enumerate(numeric_features) for b in numeric_features[i+1:] if pd.notna(full_corr.loc[a,b])]
pair_table=pd.DataFrame(pairs,columns=["feature_1","feature_2","spearman_rho"]); pair_table["abs_rho"]=pair_table.spearman_rho.abs()
display(pair_table.sort_values("abs_rho",ascending=False).head(30))
print("Pairs with |rho| >= .90:",int((pair_table.abs_rho>=.90).sum()))
''')
md('''## 8. Leakage and deployment risks

No columns are dropped. Provisional actions are experiment controls, not final feature selection.''')
code('''risks=[
 ("FILENAME","Unique row ID / source-order leakage","235,795 unique; UCI says it can be ignored; generation provenance is unclear.","Exclude from predictors; inspect provenance."),
 ("URL","Exact indicator memorization / duplicate partition leakage","235,370 unique and 425 repeated occurrences.","Keep as submitted input/audit key; future validation should group repeats."),
 ("Domain","Site identity/group leakage","220,086 distinct hosts can be memorized.","Investigate domain-grouped validation; report tracks separately."),
 ("Title","Page content, site identity, deployment mismatch","197,874 unique; requires fetched page.","Benchmark-only/investigate; exclude URL-only track."),
 ("Repeated URLs","Same indicator can cross partitions","425 duplicate URL occurrences.","Preserve raw data; group duplicates during future split design and inspect label conflicts."),
 ("Similarity/probability/derived fields","Possible target/source-informed statistics","UCI notes some fields are derived but publishes no formulas/reference data.","Investigate provenance before use; keep URL-only status uncertain."),
 ("TLDLegitimateProb","Potential target or source prior","Requires a reference population and may encode dataset composition.","Investigate; require independently defined, versioned lookup."),
 ("URL counts/ratios","Mathematical redundancy","Lengths, counts, ratios, and obfuscation metrics reuse the same string.","Benchmark candidates; quantify redundancy later."),
 ("Post-fetch features","Deployment mismatch and collection artifacts","Require HTML/page/network behavior; collection details are underspecified.","Full-track benchmark only; no URL-only performance claim."),
 ("Dataset/source ordering","Temporal or acquisition shortcut","UCI variable metadata does not fully describe sampling/order.","Investigate provenance and future time/group-aware evaluation.")]
leakage_table=pd.DataFrame(risks,columns=["Feature","Concern","Reason","Action for future experiments"])
display(leakage_table)
''')
md('''## 9. Interpretation and limitations

This notebook establishes descriptive evidence, not model performance. UCI's field descriptions are blank, particularly limiting formula/provenance assessment for probability and similarity measures. A `YES` assessment permits parsing the supplied URL only; it does not allow checking DNS resolution or whether HTTPS actually negotiates TLS. TLD conventions and corpus-derived values remain uncertain where their definitions are unpublished. The analysis does not alter raw data, deduplicate, select features, create processed data, split records, or train a model.''')

nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}})
out=Path("ml/notebooks/01_eda.ipynb"); out.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,out)
print("Wrote",out,"cells",len(cells))
