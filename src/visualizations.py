# ============================================================
#  중형 영화 생존 분석 — 시각화 코드 모음 (KOBIS × TMDB)
#  9조 파워레인저 매직포스
# ------------------------------------------------------------
#  구성
#    [SETUP]  공통 설정 (라이브러리 · 한글폰트 · 색상 · 데이터 로드)
#    전처리1  결측치 현황 + scr_cnt 파이차트
#    전처리2  추가 발견 문제 카드 3개
#    전처리3  log1p 변환 전/후 분포 비교
#    EDA 1    규모별 분포
#    EDA 2    규모별 상영유지율 / 생존율 (H1 사전 증거)
#    EDA 3    TMDB 평점 vs 유지율 (H2 사전 증거)
#    EDA 4    상관관계 히트맵
#    H1       스크린 이탈 검정 시각화 (바이올린 · 생존율 · Cliff's δ)
#    H2       평점 괴리 검정 시각화 (평점 박스플롯 · 산점도)
#    H3       예측 모델 시각화 (ROC Curve)
#
#  ※ 코랩에서는 "# %%" 구분선마다 새 코드 셀로 나눠 붙이면 됩니다.
#  ※ 컬럼명은 실제 CSV 기준으로 한 번 확인하세요
#     (show_retention / vote_average / w1_show / w1_audi /
#      log_final_audi / ip_flag / price_increased / pandemic_flag /
#      runtime / scale / open_year / movie_cd)
# ============================================================


# %% ─────────────────────────────────────────────────────────
# [SETUP] 공통 설정
# ─────────────────────────────────────────────────────────────
# !pip install pandas numpy matplotlib seaborn scipy scikit-learn -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 (Google Colab) ──────────────────────────────────
import subprocess
subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)
fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ── 데이터 로드 ───────────────────────────────────────────────
from google.colab import files
uploaded = files.upload()          # movie_features_with_tmdb.csv 업로드
import io
fname = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[fname]))
print(f"로드 완료: {len(df)}편 × {len(df.columns)}열")

# (로컬 VS Code에서 실행할 경우 위 4줄 대신)
# df = pd.read_csv("movie_features_with_tmdb.csv")

# ── 규모(체급) 순서 · 색상 팔레트 ─────────────────────────────
SCALE_ORDER  = ['블록버스터', '대형', '중형', '소형', '초소형']
SCALE_COLORS = ['#0C447C', '#185FA5', '#EF9F27', '#888780', '#D3D1C7']
SCALE_CMAP   = dict(zip(SCALE_ORDER, SCALE_COLORS))

# 생존 기준: 2주차 상영유지율 ≥ 0.6 → 생존(1), 미만 → 0  (변수 정의표 기준)
SURVIVE_TH = 0.6
df['survived']   = (df['show_retention'] >= SURVIVE_TH).astype(int)
df['early_drop'] = 1 - df['show_retention']   # 초반 낙폭


# %% ─────────────────────────────────────────────────────────
# 전처리 진단 1 — 결측치 현황 + scr_cnt 분포
#   ※ 필터링 전 원본(1,545편) 기준. raw_df 에 원본을 로드해서 실행
#      (scr_cnt · w2_show · open_dt 등 전처리 전 컬럼이 필요)
# ─────────────────────────────────────────────────────────────
raw_df = df            # 원본을 따로 들고 있다면 raw_df 로 교체
N = len(raw_df)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ① 주요 컬럼 결측치 (가로 막대)
ax = axes[0]
miss_cols = {
    'open_dt'       : '개봉일',
    'w2_show'       : '2주차 상영수',
    'w1_show'       : '1주차 상영수',
    'show_retention': '상영유지율',
}
labels, vals = [], []
for c, lab in miss_cols.items():
    if c in raw_df.columns:
        labels.append(f'{c}\n({lab})')
        vals.append(int(raw_df[c].isna().sum()))
# 심각(>500) 빨강 / 주의(>100) 주황 / 그 외 파랑
colors = ['#E24B4A' if v > 500 else '#EF9F27' if v > 100 else '#9EC9E8' for v in vals]
bars = ax.barh(labels, vals, color=colors, edgecolor='white')
for b, v in zip(bars, vals):
    ax.text(v + N*0.01, b.get_y() + b.get_height()/2,
            f'{v}개\n({v/N*100:.1f}%)', va='center', fontsize=10, fontweight='bold')
ax.axvline(N*0.3, color='#E24B4A', ls='--', alpha=0.7)
ax.text(N*0.3, -0.7, '30%\n기준선', color='#E24B4A', ha='center', fontsize=9)
ax.set_title(f'① 주요 컬럼 결측치\n(전체 {N:,}편 기준)', fontsize=13, fontweight='bold')
ax.set_xlabel('결측치 개수'); ax.set_xlim(0, N*0.72)

# ② scr_cnt 파이차트
ax = axes[1]
if 'scr_cnt' in raw_df.columns:
    zero    = int((raw_df['scr_cnt'] == 0).sum())
    nonzero = int((raw_df['scr_cnt'] >  0).sum())
    ax.pie([zero, nonzero], colors=['#E24B4A', '#9EC9E8'],
           startangle=90, wedgeprops=dict(edgecolor='white'))
    ax.legend([f'scr_cnt = 0  ({zero:,}편, {zero/N*100:.1f}%)',
               f'scr_cnt > 0  ({nonzero}편, {nonzero/N*100:.1f}%)'],
              loc='lower center', bbox_to_anchor=(0.5, -0.22), fontsize=10)
ax.set_title('② scr_cnt(스크린 수) 분포\nAPI 미지원 → 사실상 전부 0',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('pre_01_missing_scrcnt.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 전처리 진단 1 저장")


# %% ─────────────────────────────────────────────────────────
# 전처리 진단 2 — 추가 발견 문제 카드 3개
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 9))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('③ 추가 발견 문제', fontsize=16, fontweight='bold', pad=10)

problems = [
    {'title': '이상치 — 옥토넛 영화',
     'detail': 'scr_cnt 최대값: 28,588\n(한국 전체 스크린 수 ≈ 3,300)\naudi_acc 음수값 발견',
     'bg': '#FAECE7', 'tc': '#A32D2D', 'y': 7.8},
    {'title': '2019년 이전 개봉 영화 혼입',
     'detail': '태극기 휘날리며(2004) 등 305편\n개봉 1주차 데이터 없음\n→ 유지율 계산 불가',
     'bg': '#FAEEDA', 'tc': '#854F0B', 'y': 5.0},
    {'title': 'days_since_open 전부 -1',
     'detail': 'open_dt가 datetime 객체로 저장\n→ 경과일 계산 실패\n→ 1주차/2주차 분리 불가',
     'bg': '#E6F1FB', 'tc': '#185FA5', 'y': 2.2},
]
from matplotlib.patches import FancyBboxPatch
for p in problems:
    ax.add_patch(FancyBboxPatch((0.5, p['y'] - 0.9), 9.0, 2.0,
                 boxstyle="round,pad=0.1",
                 facecolor=p['bg'], edgecolor=p['tc'], linewidth=1.8))
    ax.text(5, p['y'] + 0.55, p['title'], ha='center', va='center',
            fontsize=12, fontweight='bold', color=p['tc'])
    ax.text(5, p['y'] - 0.25, p['detail'], ha='center', va='center',
            fontsize=10, color=p['tc'], linespacing=1.6)

plt.tight_layout()
plt.savefig('pre_02_problems.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 전처리 진단 2 저장")


# %% ─────────────────────────────────────────────────────────
# 전처리 진단 3 — log1p 변환 전/후 분포 비교 (왜도)
#   상단: 변환 전(극단 분포) / 하단: log1p 변환 후(정규화)
# ─────────────────────────────────────────────────────────────
vars_info = [
    ('final_audi', '누적 관객 수',   '명'),   # 컬럼명은 실제 CSV로 확인
    ('w1_show',    '1주차 상영 횟수', '회'),
    ('w1_audi',    '1주차 관객 수',   '명'),
]
GREEN = '#2E8B73'
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col_idx, (col, label, unit) in enumerate(vars_info):
    data     = df[col].dropna()
    data_log = np.log1p(data)

    # ── 상단: 변환 전 ──────────────────────────────────────────────
    ax = axes[0, col_idx]
    ax.hist(data, bins=40, color='#3B7DB8', edgecolor='white',
            linewidth=0.4, alpha=0.85)
    ax.axvline(data.mean(),   color='#E24B4A', lw=2, ls='--',
               label=f'평균 {data.mean():,.0f}')
    ax.axvline(data.median(), color='#EF9F27', lw=2, ls='-',
               label=f'중앙값 {data.median():,.0f}')
    ax.set_title(f'{label}\n[변환 전]  범위: {data.min():,.0f} ~ {data.max():,.0f} {unit}',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel(f'{label} ({unit})', fontsize=10); ax.set_ylabel('빈도', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis='y')
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.97, 0.95, f'왜도: {data.skew():.2f}\n(오른쪽 꼬리 ↑)',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            color='#A32D2D',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAECE7', alpha=0.9))

    # ── 하단: 변환 후 ──────────────────────────────────────────────
    ax2 = axes[1, col_idx]
    ax2.hist(data_log, bins=40, color=GREEN, edgecolor='white',
             linewidth=0.4, alpha=0.85)
    ax2.axvline(data_log.mean(),   color='#E24B4A', lw=2, ls='--',
                label=f'평균 {data_log.mean():.2f}')
    ax2.axvline(data_log.median(), color='#EF9F27', lw=2, ls='-',
                label=f'중앙값 {data_log.median():.2f}')
    ax2.set_title(f'{label}\n[log1p 변환 후]  정규분포에 근접',
                  fontweight='bold', fontsize=11)
    ax2.set_xlabel(f'log1p({label})', fontsize=10); ax2.set_ylabel('빈도', fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25, axis='y')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.text(0.97, 0.95, f'왜도: {data_log.skew():.2f}\n(정규화 완료)',
             transform=ax2.transAxes, fontsize=9, va='top', ha='right',
             color='#0F6E56',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1F5EE', alpha=0.9))

fig.text(0.01, 0.75, '변환 전\n(극단적 분포)', va='center', ha='left',
         fontsize=12, fontweight='bold', color='#185FA5', rotation=90)
fig.text(0.01, 0.27, '변환 후\n(정규화)', va='center', ha='left',
         fontsize=12, fontweight='bold', color='#0F6E56', rotation=90)

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig('pre_03_log_transform.png', dpi=150, bbox_inches='tight')
plt.show()

# 수치 요약
print("\n=== 변환 전후 수치 비교 ===")
for col, label, unit in vars_info:
    d = df[col].dropna()
    print(f"\n[{label}]")
    print(f"  원본  — 최솟값:{d.min():>12,.0f}  최댓값:{d.max():>12,.0f}  왜도:{d.skew():.2f}")
    print(f"  log1p — 최솟값:{np.log1p(d).min():>12.2f}  최댓값:{np.log1p(d).max():>12.2f}  왜도:{np.log1p(d).skew():.2f}")
print("✅ 전처리 진단 3 저장")


# %% ─────────────────────────────────────────────────────────
# EDA 1 — 규모별 분포
#   좌: 전체 규모별 편수 / 우: 연도별 규모별 누적 막대
# ─────────────────────────────────────────────────────────────
total = df['scale'].value_counts().reindex(SCALE_ORDER).fillna(0)

year_scale = (df.groupby(['open_year', 'scale'])['movie_cd']
                .count().unstack(fill_value=0))
for s in SCALE_ORDER:
    if s not in year_scale.columns:
        year_scale[s] = 0
year_scale = year_scale[SCALE_ORDER]
years = year_scale.index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('EDA 1. 수집 데이터 규모별 분포 (2019~2025)',
             fontsize=16, fontweight='bold', y=1.02)

# 좌 — 전체 규모별 편수
ax = axes[0]
bars = ax.bar(SCALE_ORDER, total.values, color=SCALE_COLORS, edgecolor='white')
for b, v in zip(bars, total.values):
    pct = v / total.sum() * 100
    ax.text(b.get_x() + b.get_width()/2, v + total.max()*0.01,
            f'{int(v)}편\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_title('전체 규모별 편수', fontsize=13)
ax.set_ylabel('편수')
ax.set_ylim(0, total.max() * 1.15)

# 우 — 연도별 규모별 누적 막대
ax = axes[1]
bottom = np.zeros(len(years))
for s, c in zip(SCALE_ORDER, SCALE_COLORS):
    ax.bar(years, year_scale[s].values, bottom=bottom, label=s, color=c)
    bottom += year_scale[s].values
ax.set_title('연도별 규모별 분포', fontsize=13)
ax.set_xlabel('개봉 연도'); ax.set_ylabel('편수')
ax.legend(title='규모', fontsize=9)

plt.tight_layout()
plt.savefig('eda_01_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA 1 저장")


# %% ─────────────────────────────────────────────────────────
# EDA 2 — 규모별 상영유지율 / 생존율 (H1 사전 증거)
#   좌: 규모별 유지율 박스플롯 / 우: 규모별 생존율 막대
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('EDA 2. 규모별 상영유지율과 생존율 (H1)',
             fontsize=16, fontweight='bold', y=1.02)

# 좌 — 규모별 유지율 박스플롯
ax = axes[0]
box_data = [df.loc[df['scale'] == s, 'show_retention'].dropna() for s in SCALE_ORDER]
bp = ax.boxplot(box_data, labels=SCALE_ORDER, patch_artist=True, showmeans=True)
for patch, c in zip(bp['boxes'], SCALE_COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_title('규모별 2주차 상영유지율', fontsize=13)
ax.set_ylabel('상영유지율')

# 우 — 규모별 생존율 막대
ax = axes[1]
surv = df.groupby('scale')['survived'].mean().reindex(SCALE_ORDER) * 100
bars = ax.bar(SCALE_ORDER, surv.values, color=SCALE_COLORS, edgecolor='white')
for b, v in zip(bars, surv.values):
    ax.text(b.get_x() + b.get_width()/2, v + 1,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('규모별 생존율 (유지율 기준 통과 비율)', fontsize=13)
ax.set_ylabel('생존율 (%)'); ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('eda_02_h1_screen_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA 2 저장")


# %% ─────────────────────────────────────────────────────────
# EDA 3 — TMDB 평점 vs 유지율 (H2 사전 증거)
#   좌: 규모별 TMDB 평점 박스플롯 / 우: 중형 평점 vs 유지율 산점도
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('EDA 3. TMDB 평점과 상영유지율 (H2)',
             fontsize=16, fontweight='bold', y=1.02)

# 좌 — 규모별 평점 박스플롯
ax = axes[0]
rate_data = [df.loc[df['scale'] == s, 'vote_average'].dropna() for s in SCALE_ORDER]
bp = ax.boxplot(rate_data, labels=SCALE_ORDER, patch_artist=True, showmeans=True)
for patch, c in zip(bp['boxes'], SCALE_COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_title('규모별 TMDB 평점', fontsize=13)
ax.set_ylabel('TMDB 평점 (vote_average)')

# 우 — 중형 평점 vs 유지율 산점도 + 추세선
ax = axes[1]
mid = df[df['scale'] == '중형'].dropna(subset=['vote_average', 'show_retention'])
ax.scatter(mid['vote_average'], mid['show_retention'],
           color=SCALE_CMAP['중형'], alpha=0.6, edgecolor='white')
if len(mid) > 2:
    z = np.polyfit(mid['vote_average'], mid['show_retention'], 1)
    xs = np.linspace(mid['vote_average'].min(), mid['vote_average'].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), color='#993C1D', lw=2, ls='--')
    r, p = stats.pearsonr(mid['vote_average'], mid['show_retention'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}', transform=ax.transAxes,
            va='top', fontsize=11,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
ax.set_title('중형 영화: 평점 vs 유지율', fontsize=13)
ax.set_xlabel('TMDB 평점'); ax.set_ylabel('상영유지율')

plt.tight_layout()
plt.savefig('eda_03_h2_rating_gap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA 3 저장")


# %% ─────────────────────────────────────────────────────────
# EDA 4 — 상관관계 히트맵
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
corr_cols = {
    'show_retention' : '상영유지율',
    'vote_average'   : 'TMDB평점',
    'w1_show'        : '1주차상영수',
    'w1_audi'        : '1주차관객수',
    'log_final_audi' : '누적관객(log)',
    'ip_flag'        : 'IP여부',
    'price_increased': '가격인상',
    'pandemic_flag'  : '팬데믹',
    'runtime'        : '러닝타임',
}
corr_data = df[list(corr_cols.keys())].rename(columns=corr_cols)
corr_mat  = corr_data.corr()

sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 10}, ax=ax, square=True)
ax.set_title('주요 변수 상관관계 히트맵\n(TMDB 평점 포함)',
             fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('eda_04_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA 4 저장")


# %% ─────────────────────────────────────────────────────────
# H1 — 스크린 이탈 검정 시각화
#   바이올린(규모별 유지율) · 생존율 막대 · Cliff's delta
# ─────────────────────────────────────────────────────────────
def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))

# 검정 (중형 vs 대형 단측 비교 예시)
groups = [df.loc[df['scale'] == s, 'show_retention'].dropna() for s in SCALE_ORDER]
H, p_kw = stats.kruskal(*[g for g in groups if len(g) > 0])
mid = df.loc[df['scale'] == '중형', 'show_retention'].dropna()
big = df.loc[df['scale'] == '대형', 'show_retention'].dropna()
u, p_mw = stats.mannwhitneyu(mid, big, alternative='less')
delta = cliffs_delta(mid, big)
print(f"Kruskal-Wallis H={H:.2f}, p={p_kw:.4f}")
print(f"Mann-Whitney U p={p_mw:.4f} | Cliff's δ={delta:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('H1. 규모별 스크린 이탈 검정', fontsize=16, fontweight='bold', y=1.02)

# ① 바이올린
ax = axes[0]
parts = ax.violinplot([g.values for g in groups], showmeans=True)
for pc, c in zip(parts['bodies'], SCALE_COLORS):
    pc.set_facecolor(c); pc.set_alpha(0.7)
ax.set_xticks(range(1, len(SCALE_ORDER) + 1)); ax.set_xticklabels(SCALE_ORDER)
ax.set_title(f'규모별 유지율 분포\n(Kruskal-Wallis p={p_kw:.4f})', fontsize=12)
ax.set_ylabel('상영유지율')

# ② 생존율 막대
ax = axes[1]
surv = df.groupby('scale')['survived'].mean().reindex(SCALE_ORDER) * 100
bars = ax.bar(SCALE_ORDER, surv.values, color=SCALE_COLORS, edgecolor='white')
for b, v in zip(bars, surv.values):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('규모별 생존율', fontsize=12); ax.set_ylabel('생존율 (%)'); ax.set_ylim(0, 105)

# ③ Cliff's delta 수평 막대
ax = axes[2]
ax.barh(['중형 vs 대형'], [delta], color='#993C1D')
ax.axvline(0, color='gray', lw=1)
ax.set_xlim(-1, 1)
ax.set_title(f"효과 크기 Cliff's δ = {delta:.3f}", fontsize=12)
ax.text(delta, 0, f'  {delta:.3f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('h1_screen_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ H1 저장")


# %% ─────────────────────────────────────────────────────────
# H2 — 평점 괴리 검정 시각화
#   좌: 규모별 평점 박스플롯 / 우: 중형 평점 vs 유지율 산점도
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('H2. 평점 괴리 검정', fontsize=16, fontweight='bold', y=1.02)

# 평점 차이 검정 (중형 vs 대형)
mid_r = df.loc[df['scale'] == '중형', 'vote_average'].dropna()
big_r = df.loc[df['scale'] == '대형', 'vote_average'].dropna()
u2, p_rate = stats.mannwhitneyu(mid_r, big_r)

ax = axes[0]
rate_data = [df.loc[df['scale'] == s, 'vote_average'].dropna() for s in SCALE_ORDER]
bp = ax.boxplot(rate_data, labels=SCALE_ORDER, patch_artist=True, showmeans=True)
for patch, c in zip(bp['boxes'], SCALE_COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_title(f'규모별 TMDB 평점\n(중형 vs 대형 Mann-Whitney p={p_rate:.3f})', fontsize=12)
ax.set_ylabel('TMDB 평점')

ax = axes[1]
m = df[df['scale'] == '중형'].dropna(subset=['vote_average', 'show_retention'])
ax.scatter(m['vote_average'], m['show_retention'],
           color=SCALE_CMAP['중형'], alpha=0.6, edgecolor='white')
if len(m) > 2:
    z = np.polyfit(m['vote_average'], m['show_retention'], 1)
    xs = np.linspace(m['vote_average'].min(), m['vote_average'].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), color='#993C1D', lw=2, ls='--')
    r, p = stats.pearsonr(m['vote_average'], m['show_retention'])
    ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\np = {p:.3f}',
            transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
ax.set_title('중형: 평점 vs 유지율', fontsize=12)
ax.set_xlabel('TMDB 평점'); ax.set_ylabel('상영유지율')

plt.tight_layout()
plt.savefig('h2_rating_gap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ H2 저장")


# %% ─────────────────────────────────────────────────────────
# H3 — 예측 모델 시각화 (로지스틱 회귀 ROC Curve, 5-Fold CV)
# ─────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

FEATURES = ['w1_show', 'w1_audi', 'vote_average',
            'ip_flag', 'pandemic_flag', 'price_increased']
data = df.dropna(subset=FEATURES + ['survived']).copy()
# 1주차 흥행 지표는 로그 변환
for col in ['w1_show', 'w1_audi']:
    data[col] = np.log1p(data[col])

X = StandardScaler().fit_transform(data[FEATURES])
y = data['survived'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tprs, aucs = [], []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(8, 7))
for i, (tr, te) in enumerate(skf.split(X, y), 1):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X[tr], y[tr])
    prob = model.predict_proba(X[te])[:, 1]
    fpr, tpr, _ = roc_curve(y[te], prob)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0
    ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i} (AUC={roc_auc:.3f})')

mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
ax.plot(mean_fpr, mean_tpr, color='#0C447C', lw=3,
        label=f'평균 ROC (AUC={mean_auc:.3f})')
ax.plot([0, 1], [0, 1], ls='--', color='gray', label='기준선 (0.5)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('H3. 로지스틱 회귀 ROC Curve (5-Fold CV)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('h3_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ H3 저장 | 5-Fold 평균 AUC = {mean_auc:.3f}")
