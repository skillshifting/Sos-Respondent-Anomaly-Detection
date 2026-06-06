"""
Полусеместровый контроль №4 — Поиск аномальных респондентов в активности SoS.

Запуск:
    python solution_FIO_GROUP.py
    python solution_FIO_GROUP.py --data path/to/data_train --out output

После запуска создаётся папка output/ с обязательными файлами:
    output/anomalies.csv
    output/anomaly_reasons.csv
    output/plots/total_ots_before_after.png
    output/plots/category_ots_change.png
    output/plots/daily_anomaly_count.png

КРАТКОЕ ОПИСАНИЕ АЛГОРИТМА (подробнее — в README.md):
  Единица анализа (триггер): (SubjectID, researchdate, BrandID, CategoryDelivery).
  Единица удаления:          (SubjectID, researchdate).
  daily_ots(i, j, k) = Weight(i, k) * count_rows(i, j, k), где Weight(i, k) —
  дневной вес респондента (медиана веса в пределах респондент-дня, т.к. в данных
  встречается небольшой разброс), count_rows — число строк респондента по этому
  бренду/категории за день (только BrandinDelivery == 1 и непустой CategoryDelivery).

  Аномалия = чрезмерно ВЫСОКИЙ daily_ots относительно ИСТОРИЧЕСКОГО уровня самого
  бренда. Для каждого бренда строится эталонное распределение log(daily_ots) по всем
  его (респондент, день) наблюдениям за период. Сила аномалии:
      score = 0.6745 * (L - median_L) / MAD_L           (робастный modified z-score,
                                                          Iglewicz & Hoaglin, 1993)
  где L = log1p(daily_ots). Лог-пространство выбрано, т.к. OTS мультипликативен и
  сильно скошен; в логах внутрибрендовое распределение приближённо симметрично,
  что и требуется для modified z-score.

  Порог threshold считается ИЗ ДАННЫХ и свой для каждого бренда:
      threshold = max( z(Q3 + 3*IQR),  3.5 )
  — это "дальняя" ограда Тьюки (экстремальный выброс, k=3), пересчитанная в те же
  z-единицы, но не ниже классического порога 3.5. Решение: score > threshold.

  Защита малых выборок / редких брендов: если у бренда < N_MIN наблюдений, эталон
  берётся на уровне CategoryDelivery (иерархический фолбэк). Если MAD == 0 — фолбэк
  на категориальный/глобальный MAD.

  Защита от удаления малого OTS: дополнительно требуется daily_ots >= глобальной
  медианы daily_ots (абсолютный пол). Малый OTS не может стать причиной удаления.

  Никаких захардкоженных SubjectID / дат / брендов / целевого процента удалений.
  Все пороги — стандартные статистические константы либо квантили самих данных,
  поэтому алгоритм воспроизводим и устойчив к смене периода/категорий/брендов.
"""

import argparse
import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_MIN = 30
MODZ_FLOOR = 3.5
TUKEY_K = 3.0

def find_data_dir(user_path):
    """Находит каталог с parquet-данными. Без хардкода конкретных файлов."""
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates += [
        "data_train", os.path.join("data", "data_train"),
        os.path.join("data", "data_train", "data_train"), "data", ".",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            if glob.glob(os.path.join(c, "**", "*.parquet"), recursive=True):
                return c
    raise FileNotFoundError(
        "Не найдены parquet-файлы. Укажите путь: --data path/to/data_train"
    )

def load_data(data_dir):
    """Читает все parquet рекурсивно и собирает в один DataFrame.

    Файлы партиционированы по месяцам; диапазоны researchdate не пересекаются,
    поэтому простая конкатенация не создаёт дублей.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"),
                             recursive=True))
    files = [f for f in files if "invalidResp" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError("Не найдены parquet-файлы в " + data_dir)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    if "CategoryDelivery" not in df.columns:
        if "CategoryNameDelivery" in df.columns:
            df["CategoryDelivery"] = df["CategoryNameDelivery"]
        else:
            raise KeyError("Нет столбца CategoryDelivery / CategoryNameDelivery")

    for c in ["Weight", "week_weight", "month_weight"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    df["researchdate"] = pd.to_datetime(df["researchdate"]).dt.date
    return df

def compute_daily_ots(df):
    """Считает daily_ots на уровне триггера (Subject, date, CategoryDelivery, BrandID).

    Берутся только строки поставки: BrandinDelivery == 1 и непустой CategoryDelivery.
    Weight(i, k) — дневной вес респондента (медиана веса в пределах респондент-дня).
    """
    mask = (df["BrandinDelivery"] == 1)
    cat = df["CategoryDelivery"].astype("string").str.strip()
    mask &= cat.notna() & (cat != "")
    d1 = df.loc[mask].copy()

    dweight = (d1.groupby(["SubjectID", "researchdate"])["Weight"]
                 .median().rename("dweight").reset_index())

    cnt = (d1.groupby(["SubjectID", "researchdate",
                       "CategoryDelivery", "BrandID"])
             .size().rename("count_rows").reset_index())

    cnt = cnt.merge(dweight, on=["SubjectID", "researchdate"], how="left")
    cnt["daily_ots"] = cnt["dweight"] * cnt["count_rows"]

    bname = (d1.dropna(subset=["Brand"]).groupby("BrandID")["Brand"]
               .first().rename("Brand").reset_index())
    cnt = cnt.merge(bname, on="BrandID", how="left")
    cnt["Brand"] = cnt["Brand"].fillna("")
    return cnt

def _robust_group_stats(cnt, keys, value="L"):
    """median, MAD, Q1, Q3, n по группе keys (transform -> массивы по строкам)."""
    g = cnt.groupby(keys)[value]
    med = g.transform("median").to_numpy()
    mad = g.transform(lambda x: (x - x.median()).abs().median()).to_numpy()
    q1 = g.transform(lambda x: x.quantile(0.25)).to_numpy()
    q3 = g.transform(lambda x: x.quantile(0.75)).to_numpy()
    n = g.transform("size").to_numpy()
    return med, mad, q1, q3, n

def detect_anomalies(cnt):
    """Помечает аномальные триггеры. Возвращает cnt с колонками score/threshold/flag."""
    cnt = cnt.copy()
    cnt["L"] = np.log1p(cnt["daily_ots"].to_numpy())

    bmed, bmad, bq1, bq3, bn = _robust_group_stats(
        cnt, ["CategoryDelivery", "BrandID"])
    cmed, cmad, cq1, cq3, _ = _robust_group_stats(cnt, ["CategoryDelivery"])

    use_brand = bn >= N_MIN
    med = np.where(use_brand, bmed, cmed)
    mad = np.where(use_brand, bmad, cmad)
    q1 = np.where(use_brand, bq1, cq1)
    q3 = np.where(use_brand, bq3, cq3)

    global_mad = max(float(np.median(np.abs(cnt["L"] - cnt["L"].median()))), 1e-6)
    mad = np.where(mad <= 1e-9,
                   np.where(cmad > 1e-9, cmad, global_mad), mad)

    iqr = np.maximum(q3 - q1, 1e-9)

    score = 0.6745 * (cnt["L"].to_numpy() - med) / mad

    far_L = q3 + TUKEY_K * iqr
    threshold = np.maximum(0.6745 * (far_L - med) / mad, MODZ_FLOOR)

    ots_floor = float(cnt["daily_ots"].median())

    cnt["score"] = np.round(score, 4)
    cnt["threshold"] = np.round(threshold, 4)
    cnt["ref_level"] = np.where(use_brand, "brand", "category")
    cnt["ref_n"] = bn.astype(int)
    cnt["flag"] = (score > threshold) & (cnt["daily_ots"].to_numpy() >= ots_floor)
    cnt.attrs["ots_floor"] = ots_floor
    return cnt

def build_reasons(cnt_flagged):
    """Формирует anomaly_reasons по помеченным триггерам."""
    f = cnt_flagged.loc[cnt_flagged["flag"]].copy()

    def _reason(r):
        return (f"daily_ots={r.daily_ots:,.0f} превышает '{r.ref_level}'-эталон бренда: "
                f"робастный z={r.score} > порог {r.threshold} "
                f"(дальняя ограда Тьюки Q3+3·IQR, ≥3.5); "
                f"count_rows={int(r.count_rows)}; выше глобального пола OTS")

    f["reason"] = f.apply(_reason, axis=1)
    cols = ["SubjectID", "researchdate", "BrandID", "Brand",
            "CategoryDelivery", "daily_ots", "score", "threshold", "reason"]
    f = f[cols].sort_values(["researchdate", "SubjectID", "daily_ots"],
                            ascending=[True, True, False]).reset_index(drop=True)
    f["daily_ots"] = f["daily_ots"].round(3)
    return f

_INK = "#22303f"
_C_BEFORE = "#c0392b"
_C_AFTER = "#1f7a5a"
_C_BAR = "#2f6f8f"
_C_POS = "#1f7a5a"
_C_NEG = "#c0392b"

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", color=_INK, pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=_INK)
    ax.set_ylabel(ylabel, fontsize=10, color=_INK)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=_INK)

def plot_total_ots_before_after(cnt, removed_pairs, path):
    """Суммарный дневной OTS до и после удаления аномалий."""
    idx = cnt.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = cnt.groupby("researchdate")["daily_ots"].sum()
    after = (cnt.loc[~is_removed].groupby("researchdate")["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    b, a = before / 1000.0, after / 1000.0
    ab, aa = b.mean(), a.mean()
    pct = 100 * aa / ab if ab else 0.0

    fig, ax = plt.subplots(figsize=(15, 7))
    x = list(range(len(before)))
    ax.fill_between(x, b.values, color=_C_BEFORE, alpha=0.15)
    ax.plot(x, b.values, color=_C_BEFORE, linewidth=1.6, label="До очистки")
    ax.plot(x, a.values, color=_C_AFTER, linewidth=1.8, label="После очистки")
    ax.set_xticks(x[::max(1, len(x) // 30)])
    ax.set_xticklabels([before.index[i].strftime("%m-%d")
                        for i in x[::max(1, len(x) // 30)]],
                       rotation=90, fontsize=7)
    _style(ax, "Суммарный OTS по дням: до и после удаления аномалий",
           "Дата", "OTS, тыс.")
    ax.text(0.99, 0.04, f"Сохранено OTS: {pct:.2f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=_INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=_C_AFTER, alpha=0.9))
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def plot_category_ots_change(cnt, removed_pairs, path):
    """Гистограмма изменения OTS по CategoryDelivery в процентах."""
    idx = cnt.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = cnt.groupby("CategoryDelivery")["daily_ots"].sum()
    after = (cnt.loc[~is_removed].groupby("CategoryDelivery")["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    change = (after - before) / before * 100.0
    change = change.sort_index()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(len(change)), change.values)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(change)))
    ax.set_xticklabels(change.index, rotation=90, fontsize=7)
    ax.set_xlabel("Категория"); ax.set_ylabel("%")
    ax.set_title("Гистограмма изменения суммарного OTS по категориям, % (после − до)")
    for i, v in enumerate(change.values):
        ax.annotate(f"{v:.1f}", (i, v), ha="center",
                    va="top" if v < 0 else "bottom", fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def plot_daily_anomaly_count(reasons, path):
    """Столбчатый график числа аномальных респондентов по дням."""
    pairs = reasons[["SubjectID", "researchdate"]].drop_duplicates()
    per_day = pairs.groupby("researchdate").size()
    all_days = pd.Series(0, index=sorted(reasons["researchdate"].unique())) \
        if len(reasons) else pd.Series(dtype=int)
    per_day = per_day.reindex(per_day.index.union(all_days.index), fill_value=0) \
        if len(per_day) else per_day
    total = len(pairs); uniq = pairs["SubjectID"].nunique()

    fig, ax = plt.subplots(figsize=(15, 6))
    if len(per_day):
        ax.bar(range(len(per_day)), per_day.values, color=_C_BAR, alpha=0.9)
        step = max(1, len(per_day) // 30)
        ax.set_xticks(range(0, len(per_day), step))
        ax.set_xticklabels([per_day.index[i].strftime("%m-%d")
                            for i in range(0, len(per_day), step)],
                           rotation=90, fontsize=7)
        for i, v in enumerate(per_day.values):
            if v:
                ax.annotate(str(int(v)), (i, v), ha="center", va="bottom",
                            fontsize=7, color=_INK)
    _style(ax, f"Удалённые респонденты по дням "
           f"(всего {total}, уникальных {uniq})", "Дата", "Респондентов за день")
    ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def before_after_by_dimension(df, removed_pairs, dim_col, out_png=None, cnt=None):
    """OTS до/после очистки в разрезе любого признака респондента/ресурса/категории.

    dim_col — напр. 'Пол', 'Возраст', 'Регион', 'Федеральный_округ',
    'ResourceName', 'ResourceType', 'Platform', 'UseType',
    'CategoryDelivery', 'Category1', 'Category2', 'Category3'.
    Возвращает DataFrame [dim, ots_before, ots_after, change_pct].
    cnt можно передать заранее посчитанным (ускорение).
    """
    if cnt is None:
        cnt = compute_daily_ots(df)
    key = ["SubjectID", "researchdate", "CategoryDelivery", "BrandID"]
    if dim_col in cnt.columns:
        cnt = cnt.copy()
        cnt[dim_col] = cnt[dim_col].fillna("—")
    else:
        src = df.loc[(df["BrandinDelivery"] == 1)].copy()
        if dim_col not in src.columns:
            raise KeyError(f"Нет столбца {dim_col}")
        dim = (src.dropna(subset=[dim_col]).groupby(key)[dim_col]
                  .first().rename(dim_col).reset_index())
        cnt = cnt.merge(dim, on=key, how="left")
        cnt[dim_col] = cnt[dim_col].fillna("—")

    idx = cnt.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = cnt.groupby(dim_col)["daily_ots"].sum()
    after = (cnt.loc[~is_removed].groupby(dim_col)["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    res = pd.DataFrame({"ots_before": before, "ots_after": after})
    res["change_pct"] = (res["ots_after"] - res["ots_before"]) / res["ots_before"] * 100
    res = res.sort_values("ots_before", ascending=False)

    if out_png:
        b = res["ots_before"].values / 1000.0
        a = res["ots_after"].values / 1000.0
        chg = res["change_pct"].values
        y = list(range(len(res)))
        h = 0.4
        fig, ax = plt.subplots(figsize=(13, max(5, 0.55 * len(res))))
        ax.barh([i - h / 2 for i in y], b, height=h, color=_C_BEFORE,
                alpha=0.85, label="До очистки")
        ax.barh([i + h / 2 for i in y], a, height=h, color=_C_AFTER,
                alpha=0.85, label="После очистки")
        xmax = max(b.max(), a.max())
        for i in y:
            ax.annotate(f"{b[i]:,.0f}", (b[i], i - h / 2), va="center", ha="left",
                        fontsize=7, color=_INK, xytext=(3, 0),
                        textcoords="offset points")
            ax.annotate(f"{a[i]:,.0f}", (a[i], i + h / 2), va="center", ha="left",
                        fontsize=7, color=_INK, xytext=(3, 0),
                        textcoords="offset points")
            ax.annotate(f"{chg[i]:+.1f}%", (xmax * 1.17, i), va="center",
                        ha="left", fontsize=8, fontweight="bold",
                        color=_C_NEG if chg[i] < 0 else _C_POS)
        ax.set_xlim(0, xmax * 1.32)
        ax.set_yticks(y)
        ax.set_yticklabels(res.index, fontsize=8)
        ax.invert_yaxis()
        _style(ax, f"OTS до и после очистки по признаку «{dim_col}» (тыс. + % изменения)",
               "OTS, тыс.", dim_col)
        ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.legend(frameon=False, fontsize=9, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.tight_layout(); fig.savefig(out_png, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return res

def querytext_table(df, subject_id, date):
    """Таблица поисковых запросов QueryText для выбранного респондента и дня."""
    if not isinstance(date, (pd.Timestamp,)) and not hasattr(date, "year"):
        date = pd.to_datetime(date).date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    m = (df["SubjectID"] == subject_id) & (df["researchdate"] == date)
    cols = [c for c in ["Start", "QueryText", "Brand", "CategoryDelivery",
                        "ResourceName", "BrandinDelivery"] if c in df.columns]
    return df.loc[m, cols].sort_values("Start") if "Start" in cols else df.loc[m, cols]

def brand_ots_by_day(df, removed_pairs, brand_id, category=None, out_png=None, cnt=None):
    """Изменение OTS по дням для выбранного бренда до и после очистки."""
    if cnt is None:
        cnt = compute_daily_ots(df)
    m = cnt["BrandID"] == brand_id
    if category is not None:
        m &= cnt["CategoryDelivery"] == category
    sub = cnt.loc[m]
    idx = sub.set_index(["SubjectID", "researchdate"]).index
    is_removed = idx.isin(removed_pairs)
    before = sub.groupby("researchdate")["daily_ots"].sum()
    after = (sub.loc[~is_removed].groupby("researchdate")["daily_ots"].sum()
               .reindex(before.index, fill_value=0.0))
    if out_png:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = list(range(len(before)))
        ax.fill_between(x, before.values, color=_C_BEFORE, alpha=0.15)
        ax.plot(x, before.values, color=_C_BEFORE, linewidth=1.6,
                marker="o", ms=3, label="До очистки")
        ax.plot(x, after.values, color=_C_AFTER, linewidth=1.8,
                marker="o", ms=3, label="После очистки")
        step = max(1, len(before) // 30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([before.index[i].strftime("%m-%d") for i in x[::step]],
                           rotation=90, fontsize=7)
        _style(ax, f"OTS по дням для бренда {brand_id}: до и после очистки",
               "Дата", "OTS")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return pd.DataFrame({"ots_before": before, "ots_after": after})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="каталог с parquet (data_train)")
    ap.add_argument("--out", default="output", help="каталог вывода")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data)
    out_dir = args.out
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[1/5] Чтение данных из: {data_dir}")
    df = load_data(data_dir)
    print(f"      строк: {len(df):,}, респондентов: {df['SubjectID'].nunique():,}, "
          f"дней: {df['researchdate'].nunique()}")

    print("[2/5] Агрегация daily_ots ...")
    cnt = compute_daily_ots(df)
    print(f"      триггеров (Subject,date,Cat,Brand): {len(cnt):,}")

    print("[3/5] Поиск аномалий ...")
    cnt = detect_anomalies(cnt)
    reasons = build_reasons(cnt)
    anomalies = (reasons[["SubjectID", "researchdate"]]
                 .drop_duplicates().sort_values(["researchdate", "SubjectID"])
                 .reset_index(drop=True))
    removed_pairs = pd.MultiIndex.from_frame(
        anomalies[["SubjectID", "researchdate"]])
    print(f"      триггеров-аномалий: {len(reasons)} | "
          f"удаляемых пар (Subject,date): {len(anomalies)} | "
          f"уникальных респондентов: {anomalies['SubjectID'].nunique()}")
    print(f"      доля удалённых респондент-дней: "
          f"{100*len(anomalies)/cnt[['SubjectID','researchdate']].drop_duplicates().shape[0]:.4f}%")

    print("[4/5] Запись CSV ...")
    anomalies.to_csv(os.path.join(out_dir, "anomalies.csv"), index=False)
    reasons.to_csv(os.path.join(out_dir, "anomaly_reasons.csv"), index=False)

    print("[5/5] Построение обязательных графиков ...")
    plot_total_ots_before_after(
        cnt, removed_pairs, os.path.join(plots_dir, "total_ots_before_after.png"))
    plot_category_ots_change(
        cnt, removed_pairs, os.path.join(plots_dir, "category_ots_change.png"))
    plot_daily_anomaly_count(
        reasons, os.path.join(plots_dir, "daily_anomaly_count.png"))

    extra = os.path.join(plots_dir, "extra")
    os.makedirs(extra, exist_ok=True)
    for dim in ["Пол", "Возраст", "Регион", "Федеральный_округ",
                "ResourceType", "Platform", "UseType", "CategoryDelivery"]:
        try:
            before_after_by_dimension(
                df, removed_pairs, dim, cnt=cnt,
                out_png=os.path.join(extra, f"by_{dim}.png"))
        except Exception as e:
            print(f"      (разрез {dim} пропущен: {e})")
    if len(anomalies):
        s0, d0 = anomalies.iloc[0]["SubjectID"], anomalies.iloc[0]["researchdate"]
        qt = querytext_table(df, s0, d0)
        qt.to_csv(os.path.join(extra, "querytext_example.csv"), index=False)
        b0 = reasons.iloc[0]["BrandID"]
        brand_ots_by_day(df, removed_pairs, b0, cnt=cnt,
                         out_png=os.path.join(extra, "brand_ots_by_day_example.png"))

    print("Готово. Результаты в:", os.path.abspath(out_dir))

if __name__ == "__main__":
    main()
