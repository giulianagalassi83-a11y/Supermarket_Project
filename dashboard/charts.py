"""Altair charts for the dashboard. Every function takes a DataFrame and returns a finished chart.

Chain colors can't be told apart by colour alone (red and green fail the colour-blindness check), so a
chart that uses them also names each chain on its axis. Sequential scales use one hue; signed changes
use two hues around a grey midpoint.
"""

import altair as alt
import pandas as pd
import numpy as np

INK, MUTED, BEST = "#17242D", "#5F6E78", "#12805C"
QUIET = "#9AA7AE"
RISE, FALL = "#C97132", "#1E5736"
SEQUENTIAL = ["#E4F1EC", "#9CCDBB", "#4E9F88", "#17685A"]
EMPTY = "#EEF1EF"
SURFACE = "#FFFFFF"


def styled(chart):
    fit = alt.AutoSizeParams(type="fit-x", contains="padding")
    return chart.properties(autosize=fit, padding={"left": 6, "right": 14, "top": 8, "bottom": 6}).configure(
        font="Archivo", background=SURFACE,
    ).configure_view(strokeWidth=0).configure_axis(
        labelColor=MUTED, titleColor=MUTED, gridColor="#E8EDEA", domainColor="#C9D2CD", labelFontSize=12, titleFontSize=12,
    ).configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, titleFontSize=12)


def chain_scale(colors, names):
    return alt.Scale(domain=[names.get(k, k) for k in colors], range=list(colors.values()))


def _axis(title, **kwargs):
    return alt.Axis(title=title, ticks=False, domain=False, **kwargs)


def _height(rows, step=34, extra=44):
    return rows * step + extra

def _unwrap_scalar(val, default=None):
    while isinstance(val, (pd.DataFrame, pd.Series, list, tuple)):
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return default
            val = val.iloc[0, 0]
        elif isinstance(val, pd.Series):
            if val.empty:
                return default
            val = val.iloc[0]
        else:
            if len(val) == 0:
                return default
            val = val[0]
    return default if val is None else val


def _to_float(val, default=0.0):
    scalar = _unwrap_scalar(val, default)
    try:
        num = float(scalar)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if pd.isna(num) else num


def _to_int(val, default=0):
    return int(round(_to_float(val, float(default))))


def worth_going_further(cards, colors, names, cost_per_km, cost_matrix=None):
    """Diverging bar chart of net saving per available item vs. the closest store."""
    if cards is None or not isinstance(cards, pd.DataFrame) or cards.empty:
        return styled(
            alt.Chart(pd.DataFrame({"message": ["No stores available in range"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=160)
        )

    c_df = cards.loc[:, ~cards.columns.duplicated()].copy()
    required = {"chain", "basket_eur", "have", "distance_km"}
    if not required.issubset(c_df.columns):
        return styled(
            alt.Chart(pd.DataFrame({"message": ["Missing basket columns for comparison"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=160)
        )

    have_num = pd.to_numeric(c_df["have"], errors="coerce")
    dist_num = pd.to_numeric(c_df["distance_km"], errors="coerce")
    basket_num = pd.to_numeric(c_df["basket_eur"], errors="coerce")

    valid = c_df[basket_num.notna() & dist_num.notna() & (have_num.fillna(0) > 0)].reset_index(drop=True)
    if valid.empty:
        return styled(
            alt.Chart(pd.DataFrame({"message": ["No basket items available in nearby stores"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=160)
        )

    cm = (
        cost_matrix.loc[:, ~cost_matrix.columns.duplicated()]
        if isinstance(cost_matrix, pd.DataFrame) and not cost_matrix.empty
        else None
    )

    max_have_raw = pd.to_numeric(valid["have"], errors="coerce").max()
    total_basket_items = max(1, int(len(cm)) if cm is not None else _to_int(max_have_raw, 1))

    valid_dist = pd.to_numeric(valid["distance_km"], errors="coerce")
    closest_pos = int(valid_dist.values.argmin())
    closest = valid.iloc[closest_pos]

    closest_chain = str(_unwrap_scalar(closest["chain"], ""))
    closest_dist = _to_float(closest["distance_km"], 0.0)
    closest_basket = _to_float(closest["basket_eur"], 0.0)
    closest_have = max(1, _to_int(closest["have"], 1))
    rate = _to_float(cost_per_km, 0.0)

    rows = []
    for r in valid.itertuples(index=False):
        chain = str(_unwrap_scalar(getattr(r, "chain", ""), ""))
        name = str(names.get(chain, chain))
        dist_km = _to_float(getattr(r, "distance_km", 0.0), 0.0)
        basket_eur = _to_float(getattr(r, "basket_eur", 0.0), 0.0)
        have_n = max(1, _to_int(getattr(r, "have", 1), 1))

        extra_km = max(0.0, (dist_km - closest_dist) * 2.0 * 1.3)
        travel_cost = extra_km * rate

        if cm is not None and chain in cm.columns and closest_chain in cm.columns:
            cols = [closest_chain] if chain == closest_chain else [closest_chain, chain]
            both = cm[cols].dropna()
            if not both.empty and chain != closest_chain:
                s_closest = pd.to_numeric(_unwrap_scalar(both[closest_chain], both[closest_chain]), errors="coerce")
                s_chain = pd.to_numeric(_unwrap_scalar(both[chain], both[chain]), errors="coerce")
                if isinstance(s_closest, pd.DataFrame):
                    s_closest = s_closest.iloc[:, 0]
                if isinstance(s_chain, pd.DataFrame):
                    s_chain = s_chain.iloc[:, 0]
                diff = pd.Series(s_closest) - pd.Series(s_chain)
                gross_saving = _to_float(diff.sum(), 0.0)
                n_comp = max(1, int(diff.notna().sum()))
            else:
                gross_saving = 0.0
                n_comp = have_n
        else:
            ref_avg = closest_basket / closest_have
            cur_avg = basket_eur / have_n
            gross_saving = (ref_avg - cur_avg) * have_n
            n_comp = have_n

        net_saving = gross_saving - travel_cost
        per_item = net_saving / max(1, n_comp)

        is_closest = (chain == closest_chain) and abs(dist_km - closest_dist) < 1e-6
        plot_val = 0.0 if is_closest else round(per_item, 3)

        rows.append({
            "Chain": name,
            "net_per_item": plot_val,
            "net_total": round(net_saving, 2),
            "gross_saving": round(gross_saving, 2),
            "travel_cost": round(travel_cost, 2),
            "items_txt": f"{have_n}/{total_basket_items} items",
            "coverage_pct": round(100 * have_n / total_basket_items, 1),
            "distance_km": round(dist_km, 2),
            "verdict": (
                "Closest store" if is_closest
                else ("Worth going further" if per_item > 1e-4 else "Not worth it")
            ),
            "is_closest": is_closest,
        })


    
    c = pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)
    order = c["Chain"].tolist()

    # Keep zero visible and add some breathing room to the horizontal axis.
    lo = min(0.0, float(c["net_per_item"].min()))
    hi = max(0.0, float(c["net_per_item"].max()))
    pad = max(0.08, (hi - lo) * 0.12)

    min_v = lo - pad
    max_v = hi + pad

    x_enc = alt.X(
        "net_per_item:Q",
        title="Net saving per comparable item (€/item)",
        scale=alt.Scale(domain=[min_v, max_v]),
        axis=alt.Axis(format="+.2f", tickCount=6),
    )

    y_enc = alt.Y(
        "coverage_pct:Q",
        title="Basket coverage (%)",
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(format=".0f", values=[0, 20, 40, 60, 80, 100]),
    )

    zero_rule = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color=INK, strokeWidth=1.5, strokeDash=[5, 4])
        .encode(x="x:Q")
    )

    points = (
        alt.Chart(c)
        .mark_circle(size=180, opacity=0.9, stroke="white", strokeWidth=1.5)
        .encode(
            x=x_enc,
            y=y_enc,
            color=alt.Color(
                "Chain:N",
                scale=chain_scale(colors, names),
                legend=alt.Legend(title="Supermarket", orient="top", columns=4),
            ),
            tooltip=[
                alt.Tooltip("Chain:N", title="Supermarket"),
                alt.Tooltip("verdict:N", title="Assessment"),
                alt.Tooltip("items_txt:N", title="Available products"),
                alt.Tooltip("coverage_pct:Q", title="Basket coverage (%)", format=".1f"),
                alt.Tooltip("net_per_item:Q", title="Net saving per item (€)", format="+.2f"),
                alt.Tooltip("net_total:Q", title="Net saving total (€)", format="+.2f"),
                alt.Tooltip("gross_saving:Q", title="Gross product saving (€)", format="+.2f"),
                alt.Tooltip("travel_cost:Q", title="Extra travel cost (€)", format=".2f"),
                alt.Tooltip("distance_km:Q", title="Distance (km)", format=".2f"),
            ],
        )
    )

    chart = (zero_rule + points).properties(
        height=_height(len(order), 38),
    )

    return styled(chart)



def sensitivity(sens, current_km):
    long = pd.concat([
        pd.DataFrame({"cost": sens["cost_per_km"], "Total": sens["best_total"], "Plan": "Best plan", "z": 0}),
        pd.DataFrame({"cost": sens["cost_per_km"], "Total": sens["single_total"], "Plan": "One store only", "z": 1}),
    ])
    domain = ["One store only", "Best plan"]
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        order="z:Q",
        x=alt.X("cost:Q", title="Travel cost (€/km)"),
        y=alt.Y("Total:Q", title="Total cost (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("Plan:N", scale=alt.Scale(domain=domain, range=[QUIET, BEST]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("Plan:N", scale=alt.Scale(domain=domain, range=[[7, 5], [1, 0]]), legend=None),
        tooltip=[alt.Tooltip("Plan:N"), alt.Tooltip("cost:Q", title="€/km"), alt.Tooltip("Total:Q", format=".2f", title="Total (€)")],
    )
    now = alt.Chart(pd.DataFrame({"x": [current_km]})).mark_rule(color=INK, strokeDash=[2, 3]).encode(x="x:Q")
    return styled((lines + now).properties(height=280))


def moved_share(summary, names, fall_color="#1E5736", rise_color="#F0883E", same_color="#ABB4AD"):
    """Share of prices that rose, stayed and fell, per chain."""
    t = summary.assign(Chain=summary["chain"].map(lambda x: names.get(x, x)))
    order = t["Chain"].tolist()
    long = t.melt(id_vars="Chain", value_vars=["down_pct", "same_pct", "up_pct"], var_name="m", value_name="Share")
    domain = ["Fell", "Unchanged", "Rose"]
    long["Price"] = long["m"].map({"down_pct": "Fell", "same_pct": "Unchanged", "up_pct": "Rose"})
    long["o"] = long["Price"].map({k: i for i, k in enumerate(domain)})
    bars = alt.Chart(long).mark_bar(height=22, stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("Share:Q", title="Share of products (%)", stack="zero", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(tickCount=5)),
        y=alt.Y("Chain:N", sort=order, axis=_axis(None)), order="o:Q",
        color=alt.Color("Price:N", scale=alt.Scale(domain=domain, range=[fall_color, same_color, rise_color]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Chain", "Price", alt.Tooltip("Share:Q", format=".1f", title="Share (%)")],
    )
    return styled(bars.properties(height=_height(len(order), 40)))


def moves(table, names, color, rounded_left=False):
    """Largest price changes, as bars from zero."""
    t = table.assign(Chain=table["supermarket"].map(lambda x: names.get(x, x)))
    t["Product"] = t["Chain"] + ": " + t["name"].str.slice(0, 34)
    # Invisible suffix: two products cut to the same label must stay two bars, not stack into one
    t["Product"] = [label + "\u200b" * i for i, label in enumerate(t["Product"])]
    t["text"] = t["change_pct"].map(lambda v: f"{v:+.0f}%")
    t["Was"] = t["old_price"].map(lambda v: f"€{v:.2f}")
    t["Now"] = t["new_price"].map(lambda v: f"€{v:.2f}")
    radius = (
        {"cornerRadiusTopLeft": 4, "cornerRadiusBottomLeft": 4}
        if rounded_left
        else {"cornerRadiusTopRight": 4, "cornerRadiusBottomRight": 4}
    )
    y = alt.Y("Product:N", sort=t["Product"].tolist(), axis=_axis(None, labelLimit=225, minExtent=225))
    bars = alt.Chart(t).mark_bar(height=16, color=color, **radius).encode(
        x=alt.X("change_pct:Q", title="Price change (%)", axis=alt.Axis(tickCount=5)), y=y,
        tooltip=["Product", "Was", "Now", alt.Tooltip("change_pct:Q", format="+.1f", title="Change (%)")],
    )
    return styled(bars.properties(height=_height(len(t), 28)))


def basket_category_donut(basket_df, categories=None, bg_color="#F4EFE4"):
    """Donut chart showing basket spending share (%) and amount (€) by category."""
    def _donut_style(chart):
        return (
            chart.properties(
                autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
                padding={"left": 6, "right": 14, "top": 8, "bottom": 6},
            )
            .configure(font="Archivo", background=bg_color)
            .configure_view(fill=bg_color, strokeWidth=0)
            .configure_axis(
                labelColor=MUTED,
                titleColor=MUTED,
                gridColor="#E8EDEA",
                domainColor="#C9D2CD",
                labelFontSize=12,
                titleFontSize=12,
            )
            .configure_legend(
                labelColor=INK,
                titleColor=MUTED,
                labelFontSize=13,
                titleFontSize=13,
            )
        )

    if basket_df is None or not isinstance(basket_df, pd.DataFrame) or basket_df.empty:
        return _donut_style(
            alt.Chart(pd.DataFrame({"message": ["No basket items to display"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=180)
        )

    df = basket_df.loc[:, ~basket_df.columns.duplicated()].copy()

    cat_col = next((c for c in ["category", "Category", "cat"] if c in df.columns), None)
    amt_col = next(
        (c for c in ["amount_eur", "cost_eur", "price_eur", "total_eur", "subtotal_eur", "spend_eur"] if c in df.columns),
        None,
    )

    if amt_col is None:
        return _donut_style(
            alt.Chart(pd.DataFrame({"message": ["Missing spend column for category donut"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=180)
        )

    df["_amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

    if cat_col is not None:
        cat_series = df[cat_col].fillna("Uncategorized").astype(str)

        if isinstance(categories, dict) and categories:
            cat_series = cat_series.map(lambda c: str(categories.get(c, c)))
        elif callable(categories):
            cat_series = cat_series.map(lambda c: str(categories(c)))

        df["_category"] = cat_series
    else:
        df["_category"] = "Basket"

    grouped = (
        df.groupby("_category", as_index=False)["_amount"]
        .sum()
        .sort_values("_amount", ascending=False)
    )
    grouped = grouped[grouped["_amount"] > 1e-6].reset_index(drop=True)

    total_eur = float(grouped["_amount"].sum())
    if total_eur <= 0.0 or grouped.empty:
        return _donut_style(
            alt.Chart(pd.DataFrame({"message": ["Basket total is €0.00"]}))
            .mark_text(fontSize=14, color=MUTED)
            .encode(text="message:N")
            .properties(height=180)
        )

    grouped["share_pct"] = grouped["_amount"] / total_eur * 100.0
    grouped["pct_label"] = grouped["share_pct"].map(lambda p: f"{p:.0f}%")
    grouped["legend_label"] = grouped.apply(
        lambda r: f"{r['_category']} ({r['share_pct']:.0f}% · €{r['_amount']:.2f})",
        axis=1,
    )

    def _lerp_hex(c1, c2, t):
        t = max(0.0, min(1.0, float(t)))
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(round(r1 + (r2 - r1) * t))
        g = int(round(g1 + (g2 - g1) * t))
        b = int(round(b1 + (b2 - b1) * t))
        return f"#{r:02X}{g:02X}{b:02X}"

    def _spaced_shades(n, start_hex, end_hex):
        if n <= 1:
            return [start_hex]
        raw = [_lerp_hex(start_hex, end_hex, i / (n - 1)) for i in range(n)]
        return raw[0::2] + raw[1::2]

    domain = grouped["legend_label"].tolist()
    n_cats = len(domain)
    n_green = (n_cats + 1) // 2
    n_orange = n_cats // 2

    green_shades = _spaced_shades(n_green, "#0F3D28", "#98D8BE")
    orange_shades = _spaced_shades(n_orange, "#C85519", "#F7B267")

    palette = []
    gi = oi = 0
    for idx in range(n_cats):
        if idx % 2 == 0 and gi < len(green_shades):
            palette.append(green_shades[gi])
            gi += 1
        elif oi < len(orange_shades):
            palette.append(orange_shades[oi])
            oi += 1
        elif gi < len(green_shades):
            palette.append(green_shades[gi])
            gi += 1

    base = alt.Chart(grouped).encode(
        theta=alt.Theta("_amount:Q", stack=True),
        color=alt.Color(
            "legend_label:N",
            scale=alt.Scale(domain=domain, range=palette[: len(domain)]),
            legend=alt.Legend(
                title="Category share",
                orient="right",
                offset=18,
                symbolSize=120,
                rowPadding=6,
                labelLimit=280,
                titleFontSize=13,
                labelFontSize=13,
            ),
        ),
        tooltip=[
            alt.Tooltip("_category:N", title="Category"),
            alt.Tooltip("_amount:Q", format=".2f", title="Spend (€)"),
            alt.Tooltip("share_pct:Q", format=".1f", title="Share (%)"),
        ],
    )

    arcs = base.mark_arc(
        innerRadius=54,
        outerRadius=92,
        stroke=bg_color,
        strokeWidth=2,
    )

    pct_labels = base.mark_text(radius=73, fontSize=11, fontWeight=700, color=SURFACE).encode(
        text=alt.condition("datum.share_pct >= 9", "pct_label:N", alt.value(""))
    )

    center_amount = (
        alt.Chart(pd.DataFrame({"txt": [f"€{total_eur:.2f}"]}))
        .mark_text(dy=-6, fontSize=15, fontWeight=700, color=INK)
        .encode(text="txt:N")
    )

    center_sub = (
        alt.Chart(pd.DataFrame({"txt": ["Plan items"]}))
        .mark_text(dy=11, fontSize=10.5, color=MUTED)
        .encode(text="txt:N")
    )

    chart_height = max(225, min(420, len(grouped) * 26))
    return _donut_style((arcs + pct_labels + center_amount + center_sub).properties(height=chart_height))


# ============================================================
# SHOPPER VIEW AND NEW ANALYST CHARTS (24 Nov)
# ============================================================

def list_cost(table, colors, names):
    """What the same shopping list costs at each chain, buying the cheapest product of each type."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)))
    t["text"] = [f"{c:.2f} €" + ("  cheapest" if e < 0.005 else f"  +{e:.2f} €")
                 for c, e in zip(t["cheapest"], t["extra"])]
    order = t["Chain"].tolist()
    y = alt.Y("Chain:N", sort=order, axis=_axis(None, labelFontSize=13, labelColor=INK))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("cheapest:Q", title="List cost (€)", axis=alt.Axis(tickCount=5),
                scale=alt.Scale(domain=[0, float(t["cheapest"].max()) * 1.3])),
        y=y, color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None),
        tooltip=["Chain", alt.Tooltip("cheapest:Q", format=".2f", title="Cheapest products (€)"),
                 alt.Tooltip("typical:Q", format=".2f", title="Typical products (€)"),
                 alt.Tooltip("extra:Q", format="+.2f", title="Over the cheapest chain (€)")],
    )
    text = alt.Chart(t).mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="cheapest:Q", y=y, text="text:N")
    return styled((bars + text).properties(height=_height(len(order), 38)))


def optimizer_savings(table):
    """Saving of the exact optimizer over the price list, one column per random basket (largest first)."""
    t = table.sort_values("saving_eur", ascending=False).reset_index(drop=True)
    t["rank"] = range(1, len(t) + 1)
    t["Result"] = t["saving_eur"].map(lambda v: "Cheaper with the optimizer" if v > 0.005 else "Same cost")
    t["plot"] = t["saving_eur"].clip(lower=0.05)  # a stub so "same cost" baskets stay visible
    domain = ["Cheaper with the optimizer", "Same cost"]
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=3, width=alt.RelativeBandSize(0.8)).encode(
        x=alt.X("rank:O", title="Random baskets, largest saving first", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("plot:Q", title="Saving (€)"),
        color=alt.Color("Result:N", scale=alt.Scale(domain=domain, range=[BEST, QUIET]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("saving_eur:Q", format=".2f", title="Saving (€)"),
                 alt.Tooltip("greedy_eur:Q", format=".2f", title="Price list total (€)"),
                 alt.Tooltip("exact_eur:Q", format=".2f", title="Optimizer total (€)"),
                 alt.Tooltip("greedy_stops:Q", title="Price list stops"), alt.Tooltip("exact_stops:Q", title="Optimizer stops")],
    )
    return styled(bars.properties(height=240))


def price_over_time(series, colors, names, unit_text):
    """One line per chain over the collection dates; every chain that sells the product is in the legend."""
    t = series.assign(Chain=series["supermarket"].map(lambda x: names.get(x, x)),
                      date=pd.to_datetime(series["collection_date"]))
    present = [k for k in colors if k in set(series["supermarket"])]
    scale = alt.Scale(domain=[names.get(k, k) for k in present], range=[colors[k] for k in present])
    tips = [alt.Tooltip("Chain:N"), alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
            alt.Tooltip("price:Q", format=".2f", title=f"€/{unit_text}")]
    if "pack_price" in t:
        tips.append(alt.Tooltip("pack_price:Q", format=".2f", title="Pack price (€)"))
    if "products" in t:
        tips.append(alt.Tooltip("products:Q", title="Products in the median"))
    hover = alt.selection_point(fields=["Chain"], on="pointerover", nearest=True, empty=False)
    # The axis spans at least ±10% of the middle price, so a 5-cent move doesn't look like a crash
    lo, hi = float(t["price"].min()), float(t["price"].max())
    mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.15, 0.1 * (lo + hi) / 2)
    base = alt.Chart(t).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, grid=False,
                                                     tickCount={"interval": "week", "step": 1}, labelOverlap=True)),
        y=alt.Y("price:Q", title=f"€ per {unit_text}", scale=alt.Scale(domain=[max(0, mid - half), mid + half], nice=True)),
        color=alt.Color("Chain:N", scale=scale,
                        legend=alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=3)),
    )
    lines = base.mark_line().encode(strokeWidth=alt.condition(hover, alt.value(3.5), alt.value(2)))
    points = base.mark_circle(size=80, opacity=1, stroke=SURFACE, strokeWidth=2).encode(tooltip=tips)
    hits = base.mark_circle(size=500, opacity=0).encode(tooltip=tips).add_params(hover)  # bigger hover target
    return styled((lines + points + hits).properties(height=340))

def aisle_runnerup_dumbbell(rows_df, chain_colors, chain_names):
    """Dumbbell chart comparing the cheapest chain (0%) against the runner-up (+%) per aisle."""
    domain = list(chain_colors.keys())
    range_colors = [chain_colors[c] for c in domain]

    display_df = rows_df.copy()
    display_df["winner_label"] = display_df["winner"].replace(chain_names)
    display_df["runner_label"] = display_df["runner"].fillna("").replace(chain_names)
    display_df["gap_text"] = display_df.apply(
        lambda r: (
            "only 1 chain"
            if pd.isna(r["extra_pct"])
            else (
                f"{r['runner_label']} (same price)"
                if r["extra_pct"] < 0.5
                else f"{r['runner_label']} (+{r['extra_pct']:.0f}%)"
            )
        ),
        axis=1,
    )
    # Keep a tiny visual offset when tied so both chain colors stay visible
    display_df["runner_x"] = display_df["extra_pct"].apply(
        lambda v: np.nan if pd.isna(v) else (0.35 if v < 0.5 else float(v))
    )
    display_df["label_x"] = display_df["runner_x"].fillna(0.0)

    order = display_df.sort_values(["extra_pct", "aisle"], ascending=[False, True])["aisle"].tolist()

    # Connecting horizontal segment
    rule = (
        alt.Chart(display_df.dropna(subset=["runner_x"]))
        .mark_rule(strokeWidth=2.5, color="#B9C4C9")
        .encode(
            y=alt.Y("aisle:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12, labelFontWeight=600)),
            x=alt.X(
                "zero:Q",
                title="Extra cost at the second cheapest chain (%)",
                axis=alt.Axis(format=".0f", labelExpr="datum.value + '%'"),
            ),
            x2="runner_x:Q",
        )
    )

    # Winner dot at 0%
    winner_pts = (
        alt.Chart(display_df)
        .mark_circle(size=130, opacity=1, stroke="#F4EFE4", strokeWidth=1.5)
        .encode(
            y=alt.Y("aisle:N", sort=order),
            x="zero:Q",
            color=alt.Color(
                "winner:N",
                scale=alt.Scale(domain=domain, range=range_colors),
                legend=alt.Legend(
                    title="Chain",
                    orient="top",
                    labelExpr="datum.label == 'Bm' ? 'BM' : datum.label",
                ),
            ),
            tooltip=[
                alt.Tooltip("aisle:N", title="Aisle"),
                alt.Tooltip("winner_label:N", title="Cheapest (0%)"),
                alt.Tooltip("runner_label:N", title="2nd cheapest"),
                alt.Tooltip("gap_text:N", title="Comparison"),
                alt.Tooltip("n_chains:Q", title="Chains compared"),
            ],
        )
    )

    # Winner chain name to the left of the 0% dot
    winner_labels = (
        alt.Chart(display_df)
        .mark_text(align="right", dx=-10, fontSize=11.5, fontWeight=700, color="#26332D")
        .encode(
            y=alt.Y("aisle:N", sort=order),
            x="zero:Q",
            text="winner_label:N",
        )
    )

    # Runner-up dot
    runner_pts = (
        alt.Chart(display_df.dropna(subset=["runner_x"]))
        .mark_circle(size=130, opacity=1, stroke="#F4EFE4", strokeWidth=1.5)
        .encode(
            y=alt.Y("aisle:N", sort=order),
            x="runner_x:Q",
            color=alt.Color("runner:N", scale=alt.Scale(domain=domain, range=range_colors), legend=None),
            tooltip=[
                alt.Tooltip("aisle:N", title="Aisle"),
                alt.Tooltip("winner_label:N", title="Cheapest (0%)"),
                alt.Tooltip("runner_label:N", title="2nd cheapest"),
                alt.Tooltip("gap_text:N", title="Comparison"),
                alt.Tooltip("n_chains:Q", title="Chains compared"),
            ],
        )
    )

    # Runner-up comparison label to the right
    runner_labels = (
        alt.Chart(display_df)
        .mark_text(align="left", dx=10, fontSize=11.5, color="#5F6E78")
        .encode(
            y=alt.Y("aisle:N", sort=order),
            x="label_x:Q",
            text="gap_text:N",
        )
    )

    max_x = float(display_df["runner_x"].max()) if display_df["runner_x"].notna().any() else 10.0

    return (
        (rule + winner_pts + runner_pts + winner_labels + runner_labels)
        .encode(x=alt.X(scale=alt.Scale(domain=[-6.5, max_x + 9])))
        .properties(height=max(380, len(display_df) * 29), background="#F4EFE4")
        .configure_view(strokeWidth=0, fill="#F4EFE4")
        .configure_axis(gridColor="#E2DACC", domainColor="#E2DACC")
    )