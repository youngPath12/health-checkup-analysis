# -*- coding: utf-8 -*-

# %%
# ── 0. 환경 설정 (Setup) ──────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # 화면 없이 파일로만 저장 (창을 띄우지 않음)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import roc_auc_score, roc_curve

# 분석·검정·시각화 함수는 utils 모듈에서 import
from utils.utils import (
    plot_box, plot_crosstab,
    run_2group, run_multigroup, posthoc, run_chi2,
)

# %%
# ── [결과 폴더] results/figures, results/tables 를 만들어 둡니다 ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
TBL_DIR     = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# 그림 파일명에 붙일 일련번호 (01, 02, ... 순서대로 정렬되게)
_fig_counter = {'n': 0}

def save_current_fig(name):
    """현재 활성 figure를 results/figures/ 에 번호를 붙여 저장하고 닫습니다."""
    _fig_counter['n'] += 1
    fname = f"{_fig_counter['n']:02d}_{name}.png"
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  [그림 저장] results/figures/{fname}")

def save_table(df_or_obj, name):
    """표(DataFrame)를 results/tables/ 에 CSV로 저장합니다 (한글 깨짐 방지 utf-8-sig)."""
    path = os.path.join(TBL_DIR, f"{name}.csv")
    obj = df_or_obj
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, encoding='utf-8-sig')
        print(f"  [표 저장]   results/tables/{name}.csv")
    return df_or_obj

# ── [핵심] utils.py 함수는 내부에서 plt.show() 를 부릅니다.
#    utils.py 를 수정하지 않고도 그 그림들을 저장하기 위해,
#    plt.show 를 '현재 그림을 저장하고 닫는' 동작으로 바꿔 끼웁니다(가로채기).
_pending_name = {'v': 'figure'}

def _show_and_save(*args, **kwargs):
    save_current_fig(_pending_name['v'])

plt.show = _show_and_save   # 이후 모든 plt.show() 는 저장으로 동작

def name_next(name):
    """다음에 그려질(=utils 함수가 show 할) 그림의 파일 이름을 지정."""
    _pending_name['v'] = name

# %%
# ── [한글 폰트] 그래프에 한글이 깨지지 않도록 설정 ────────────────
def set_korean_font():
    import matplotlib.font_manager as fm
    from matplotlib import rcParams
    candidates = ['Malgun Gothic', 'AppleGothic', 'NanumGothic']
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            rcParams['font.family'] = name
            break
    else:
        print("한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")
    rcParams['axes.unicode_minus'] = False

set_korean_font()
sns.set_theme(style='whitegrid', palette='colorblind')
set_korean_font()

# %%
# ── [경로 · 난수 · 변수 정의] ─────────────────────────────────
DATA_PATH = os.path.join(BASE_DIR, 'data', '건강검진_실습데이터.xlsx')

SEED = 42
np.random.seed(SEED)

num_cols = ['age', 'bmi', 'waist', 'sbp', 'glucose', 'tg', 'hdl', 'crp']
cat_cols = ['sex', 'region', 'smoking', 'exercise_freq', 'metabolic_syndrome']
cat_vars = ['sex', 'region', 'smoking', 'exercise_freq']

# %%
# ── STEP 0. 데이터 불러오기 ───────────────────────────────────
df = pd.read_excel(DATA_PATH)
df_raw = df.copy()
print("STEP 0. 데이터 로드:", df.shape)
print(df.head())

# %%
# ── STEP 1. 데이터 진단 — 구조 · 타입 · 결측 ──────────────────
print("\n[STEP 1] 구조 · 타입 · 결측")
print(df.shape)
df.info()

# %%
# 연속형 변수 전체 분포 — 히스토그램
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, c in zip(axes.ravel(), num_cols):
    ax.hist(df[c].dropna(), bins=25, color='#9DB4D4', edgecolor='white')
    ax.axvline(df[c].mean(),   color='#C00000', ls='--', label='평균')
    ax.axvline(df[c].median(), color='#1F3864', ls='-',  label='중앙값')
    ax.set_title(f"{c}  (왜도 {df[c].skew():.2f})")
    ax.legend(fontsize=8)
plt.suptitle('연속형 변수 분포', fontsize=14)
plt.tight_layout()
save_current_fig('continuous_distributions')

# %%
# 범주형 변수 전체 분포 — 막대그래프
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, c in zip(axes, cat_cols):
    df[c].value_counts().sort_index().plot(kind='bar', ax=ax, color='#1F3864')
    ax.set_title(c)
    ax.set_xlabel('')
plt.tight_layout()
save_current_fig('categorical_distributions')

# %%
# 결측 현황 — 표로 저장
miss = df_raw.isnull().sum()
miss_pct = (miss / len(df_raw) * 100).round(1)
miss_tbl = pd.DataFrame({'결측수': miss, '결측%': miss_pct})[miss > 0]
print("\n[결측 현황]")
print(miss_tbl)
save_table(miss_tbl, 'missing_summary')

# %%
# 결측 대치 — 중앙값으로 통일
for c in ['hdl', 'crp']:
    df[c] = df[c].fillna(df[c].median())
print('결측 합계:', df.isnull().sum().sum())

# %%
# ── STEP 2. 기술통계 ─────────────────────────────────────────
desc = df[num_cols].agg(['mean', 'median', 'std']).T
desc['skew'] = df[num_cols].skew()
desc['CV(%)'] = (df[num_cols].std() / df[num_cols].mean() * 100).round(1)
desc = desc.round(2)
print("\n[STEP 2] 연속형 기술통계")
print(desc)
save_table(desc, 'descriptive_continuous')

# %%
# 범주형 변수 빈도 — 개수와 비율(%) (하나의 표로 합쳐 저장)
print("\n[STEP 2] 범주형 빈도")
freq_rows = []
for c in cat_cols:
    counts = df[c].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(1)
    tbl = pd.DataFrame({'변수': c, '범주': counts.index,
                        '개수': counts.values, '비율(%)': pct.values})
    freq_rows.append(tbl)
    print(f'── {c} ──')
    print(tbl.to_string(index=False))
    print()
freq_all = pd.concat(freq_rows, ignore_index=True)
save_table(freq_all, 'frequency_categorical')

# %%
# ── STEP 3. 연속형 검정 ──────────────────────────────────────
# metabolic_syndrome (2집단)
print("\n[STEP 3] metabolic_syndrome (2집단) 비교")
name_next('boxplot_by_metabolic_syndrome')
plot_box(df, 'metabolic_syndrome', num_cols)
res_2g = run_2group(df, 'metabolic_syndrome', num_cols)
print(res_2g)
save_table(res_2g, 'test_2group_metabolic_syndrome')

# %%
# exercise_freq (3집단)
print("\n[STEP 3] exercise_freq (3집단) 비교")
name_next('boxplot_by_exercise_freq')
plot_box(df, 'exercise_freq', num_cols)
res_mg = run_multigroup(df, 'exercise_freq', num_cols)
print(res_mg)
save_table(res_mg, 'test_multigroup_exercise_freq')

# %%
# 사후검정: exercise_freq × hdl
print("\n[STEP 3] 사후검정: exercise_freq x hdl")
posthoc_res = posthoc(df, 'exercise_freq', 'hdl', 'ANOVA')
try:
    save_table(pd.DataFrame(posthoc_res._results_table.data[1:],
                            columns=posthoc_res._results_table.data[0]),
               'posthoc_exercise_freq_hdl')
except Exception:
    pass

# %%
# ── STEP 4. 범주형 검정 ──────────────────────────────────────
print("\n[STEP 4] 범주형 검정 (vs metabolic_syndrome)")
res_chi2 = run_chi2(df, cat_vars, 'metabolic_syndrome')
print(res_chi2)
save_table(res_chi2, 'test_chi2_categorical')

# %%
# 교차표 시각화
name_next('crosstab_by_metabolic_syndrome')
plot_crosstab(df, cat_vars, 'metabolic_syndrome')

# %%
# ── STEP 5. 상관 · 회귀 ──────────────────────────────────────
# 상관행렬
corr = df[num_cols].corr(method='pearson')
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True)
plt.title('연속형 변수 상관행렬 (Pearson)')
plt.tight_layout()
save_current_fig('correlation_heatmap')
save_table(corr.round(2), 'correlation_matrix')

# %%
# 더미 변환 + VIF
df['male'] = (df['sex'] == '남').astype(int)
df['smk']  = (df['smoking'] == '흡연').astype(int)

predictors = ['age', 'bmi', 'waist', 'glucose', 'tg', 'hdl', 'crp', 'male', 'smk']
X = add_constant(df[predictors])
vif = pd.DataFrame({
    '변수': predictors,
    'VIF': [variance_inflation_factor(X.values, i + 1) for i in range(len(predictors))]
})
vif = vif.sort_values('VIF', ascending=False).round(2)
print("\n[STEP 5] VIF (다중공선성)")
print(vif)
save_table(vif, 'vif')

# %%
# 로지스틱 회귀
model = smf.logit(
    'metabolic_syndrome ~ age + bmi + waist + glucose + tg + hdl + crp + male + smk',
    data=df
).fit()
print("\n[STEP 5] 로지스틱 회귀 요약")
print(model.summary())
with open(os.path.join(TBL_DIR, 'logit_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(model.summary().as_text())
print("  [표 저장]   results/tables/logit_summary.txt")

# %%
# 조정 오즈비
res = pd.DataFrame({
    'OR': np.exp(model.params),
    'p': model.pvalues
}).join(np.exp(model.conf_int()).rename(columns={0: 'CI_low', 1: 'CI_high'}))
res['유의'] = np.where(res['p'] < 0.05, '*', '')
or_tbl = res.drop('Intercept').round(3).sort_values('p')
print("\n[STEP 5] 조정 오즈비 (adjusted OR)")
print(or_tbl)
save_table(or_tbl, 'adjusted_odds_ratio')

# %%
# forest plot
plot_df = res.drop('Intercept').sort_values('OR')
plt.figure(figsize=(7, 5))
plt.errorbar(plot_df['OR'], range(len(plot_df)),
             xerr=[plot_df['OR'] - plot_df['CI_low'],
                   plot_df['CI_high'] - plot_df['OR']],
             fmt='o', color='#1F3864', capsize=3)
plt.axvline(1, color='#C00000', ls='--')
plt.yticks(range(len(plot_df)), plot_df.index)
plt.xscale('log')
plt.xlabel('조정 오즈비 (log scale)')
plt.title('대사증후군 위험요인 — 조정 오즈비')
plt.tight_layout()
save_current_fig('forest_plot_odds_ratio')

# %%
# ROC 곡선 / AUC
used = model.model.data.row_labels
y_true = df.loc[used, 'metabolic_syndrome']
pred = model.predict(df.loc[used])

auc = roc_auc_score(y_true, pred)
fpr, tpr, _ = roc_curve(y_true, pred)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('1 - 특이도')
plt.ylabel('민감도')
plt.title('ROC 곡선')
plt.legend()
plt.tight_layout()
save_current_fig('roc_curve')

save_table(pd.DataFrame({'지표': ['AUC'], '값': [round(auc, 3)]}), 'auc')

print(f"\n분석 완료. 결과는 results/ 폴더에 저장되었습니다.")
print(f"  - 그림: results/figures/  (PNG {_fig_counter['n']}개)")
print(f"  - 표:   results/tables/   (CSV)")