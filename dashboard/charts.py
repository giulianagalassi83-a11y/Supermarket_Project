"""Altair charts for the dashboard. Every function takes a DataFrame and returns a finished chart.

Chain colors can't be told apart by colour alone (red and green fail the colour-blindness check), so a
chart that uses them also names each chain on its axis. Sequential scales use one hue; signed changes
use two hues around a grey midpoint.
"""

import altair as alt
import pandas as pd

INK, MUTED, BEST = "#17242D", "#5F6E78", "#12805C"
QUIET = "#9AA7AE"
RISE, FALL = "#B5472F", "#12805C"
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


def scatter(cards, colors, names, money, distance):
    """Basket price against distance; stores beaten on both are faded."""
    c = cards[cards["complete"]].copy()
    c["Chain"] = c["chain"].map(lambda x: names.get(x, x))
    c["dominated"] = [
        bool(((c["basket_eur"] <= r.basket_eur) & (c["distance_km"] <= r.distance_km)
              & ((c["basket_eur"] < r.basket_eur) | (c["distance_km"] < r.distance_km))).any())
        for r in c.itertuples()
    ]
    c["Basket"] = c["basket_eur"].map(money)
    c["Distance"] = c["distance_km"].map(distance)
    base = alt.Chart(c).encode(
        x=alt.X("distance_km:Q", title="Distance to the store (km)", scale=alt.Scale(zero=True)),
        y=alt.Y("basket_eur:Q", title="Basket price (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None),
        opacity=alt.condition("datum.dominated", alt.value(0.28), alt.value(1)),
        tooltip=["Chain", "Basket", "Distance"],
    )
    labels = base.mark_text(align="left", dx=11, fontSize=12, fontWeight=700, color=INK).encode(text="Chain", color=alt.value(INK))
    return styled((base.mark_circle(size=230) + labels).properties(height=280))


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


def coverage(table, names, categories):
    """Comparable products per category and chain."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)), Category=table["category"].map(categories))
    order = t.groupby("Category")["products"].sum().sort_values(ascending=False).index.tolist()
    base = alt.Chart(t).encode(
        x=alt.X("Chain:N", axis=_axis(None, orient="top", labelAngle=0)),
        y=alt.Y("Category:N", sort=order, axis=_axis(None)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.condition("datum.products == 0", alt.value(EMPTY), alt.Color(
            "products:Q", scale=alt.Scale(type="sqrt", range=SEQUENTIAL), legend=alt.Legend(title="Products", orient="right"))),
        tooltip=["Category", "Chain", alt.Tooltip("products:Q", title="Products")],
    )
    text = base.mark_text(fontSize=12).encode(
        text="products:Q", color=alt.condition("datum.products >= 40", alt.value("#FFFFFF"), alt.value(INK)))
    return styled((cells + text).properties(height=_height(len(order), 30)))


def index_dumbbell(table, colors, names):
    """Chain index by median and by cheapest product; the distance between dots is how far the method moves it."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)))
    order = t["Chain"].tolist()
    long = t.melt(id_vars="Chain", value_vars=["median", "cheapest"], var_name="m", value_name="Index")
    long["Method"] = long["m"].map({"median": "Median price of the type", "cheapest": "Cheapest product of the type"})
    domain = ["Median price of the type", "Cheapest product of the type"]
    x = alt.X("Index:Q", title="Index (Mercadona = 100)", scale=alt.Scale(zero=False, nice=True), axis=alt.Axis(tickCount=8))
    y = alt.Y("Chain:N", sort=order, axis=_axis(None))
    ref = alt.Chart(pd.DataFrame({"Index": [100]})).mark_rule(color=QUIET).encode(x="Index:Q")
    link = alt.Chart(t).mark_rule(color="#C9D2CD", strokeWidth=2).encode(
        x=alt.X("median:Q"), x2="cheapest:Q", y=y)
    dots = alt.Chart(long).mark_circle(size=170, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        x=x, y=y,
        color=alt.Color("Method:N", scale=alt.Scale(domain=domain, range=[BEST, QUIET]), legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("m:N"),  # "median" sorts last, so it draws on top when the two dots coincide
        tooltip=["Chain", "Method", alt.Tooltip("Index:Q", format=".0f")],
    )
    return styled((ref + link + dots).properties(height=_height(len(order), 40)))


def category_heatmap(table, names, categories):
    """Price against the cheapest chain, per category. Chains with too little data are left grey."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)), Category=table["category"].map(categories))
    t["Compared"] = t["relative"].map(lambda v: "cheapest" if v <= 1.0005 else f"+{100 * (v - 1):.0f}%")
    t["shown"] = t["Compared"].where(~t["low_data"], "few")
    t["capped"] = t["relative"].clip(upper=2.5)
    order = sorted(t["Category"].unique())
    chains = t.groupby("Chain")["relative"].median().sort_values().index.tolist()
    base = alt.Chart(t).encode(
        x=alt.X("Chain:N", sort=chains, axis=_axis(None, orient="top", labelAngle=0)),
        y=alt.Y("Category:N", sort=order, axis=_axis(None)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.condition("datum.low_data", alt.value(EMPTY), alt.Color(
            "capped:Q", scale=alt.Scale(domain=[1, 2.5], range=SEQUENTIAL),
            legend=alt.Legend(title="Times the cheapest", format=".1f", orient="right"))),
        tooltip=["Category", "Chain", alt.Tooltip("Compared:N", title="Against cheapest"), alt.Tooltip("products:Q", title="Products")],
    )
    text = base.mark_text(fontSize=11.5).encode(
        text="shown:N", color=alt.condition("datum.capped >= 1.8 && !datum.low_data", alt.value("#FFFFFF"), alt.value(INK)))
    return styled((cells + text).properties(height=_height(len(order), 30)))


def _chain_bars(table, value, title, colors, names, tooltip, fmt):
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)))
    t["text"] = t[value].map(fmt)
    order = t.sort_values(value, ascending=False)["Chain"].tolist()
    y = alt.Y("Chain:N", sort=order, axis=_axis(None))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(tickCount=5)), y=y,
        color=alt.Color("Chain:N", scale=chain_scale(colors, names), legend=None), tooltip=tooltip,
    )
    return styled((bars + alt.Chart(t).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
        x=f"{value}:Q", y=y, text="text:N")).properties(height=_height(len(order))))


def own_brand(table, colors, names):
    return _chain_bars(
        table, "gap_pct", "Own brand saving against national brands (%)", colors, names,
        ["Chain", alt.Tooltip("gap_pct:Q", format=".0f", title="Saving (%)"), alt.Tooltip("types:Q", title="Types compared")],
        lambda v: f"{v:.0f}%")


def own_share(table, colors, names):
    return _chain_bars(
        table, "own_share_pct", "Own brand share of comparable products (%)", colors, names,
        ["Chain", alt.Tooltip("own_share_pct:Q", format=".0f", title="Own brand (%)")], lambda v: f"{v:.0f}%")


def spread(table, categories):
    """Within-chain spread against between-chain spread, per category."""
    t = table.assign(Category=table["category"].map(categories))
    order = t["Category"].tolist()
    long = t.melt(id_vars="Category", value_vars=["within", "between"], var_name="m", value_name="Spread")
    domain = ["Between products of one chain", "Between chains"]
    long["Choice"] = long["m"].map({"within": domain[0], "between": domain[1]})
    x = alt.X("Spread:Q", title="Dearest against cheapest (times)", scale=alt.Scale(type="log", domain=[1, 12]),
              axis=alt.Axis(values=[1, 2, 3, 5, 10]))
    y = alt.Y("Category:N", sort=order, axis=_axis(None))
    link = alt.Chart(t).mark_rule(color="#C9D2CD", strokeWidth=2).encode(x="within:Q", x2="between:Q", y=y)
    dots = alt.Chart(long).mark_circle(size=120, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        x=x, y=y,
        color=alt.Color("Choice:N", scale=alt.Scale(domain=domain, range=[BEST, QUIET]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Category", "Choice", alt.Tooltip("Spread:Q", format=".1f", title="Times")],
    )
    return styled((link + dots).properties(height=_height(len(order), 26)))


def barcode_gap(pairs):
    """Histogram of BM against Consum on the same barcode."""
    width = 4
    centre = ((pairs["gap_pct"].clip(-40, 40) + width / 2) // width * width).astype(int)
    t = centre.value_counts().rename_axis("gap").reset_index(name="products")
    t["side"] = t["gap"].map(lambda v: "BM dearer" if v > 0 else ("Consum dearer" if v < 0 else "About the same (±2%)"))
    domain = ["Consum dearer", "About the same (±2%)", "BM dearer"]
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, width=alt.RelativeBandSize(band=0.9)).encode(
        x=alt.X("gap:O", sort="ascending", title="BM price against Consum (%), capped at ±40",
                scale=alt.Scale(domain=list(range(-40, 41, width))), axis=alt.Axis(labelAngle=0, values=list(range(-40, 41, 20)))),
        y=alt.Y("products:Q", title="Products"),
        color=alt.Color("side:N", scale=alt.Scale(domain=domain, range=[FALL, QUIET, RISE]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("side:N", title="Result"), alt.Tooltip("gap:O", title="Gap (%)"), alt.Tooltip("products:Q", title="Products")],
    )
    return styled(bars.properties(height=280))


def route_savings(table):
    """Histogram of how much the exact optimizer saves over a price-list shopper."""
    bars = alt.Chart(table).mark_bar(color=BEST, cornerRadiusEnd=4, stroke=SURFACE, strokeWidth=1).encode(
        x=alt.X("saving_eur:Q", bin=alt.Bin(step=0.5), title="Saving per basket (€)"),
        y=alt.Y("count():Q", title="Baskets"),
        tooltip=[alt.Tooltip("count():Q", title="Baskets")],
    )
    return styled(bars.properties(height=260))


def moved_share(summary, names):
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
        color=alt.Color("Price:N", scale=alt.Scale(domain=domain, range=[FALL, "#DCE3DE", RISE]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Chain", "Price", alt.Tooltip("Share:Q", format=".1f", title="Share (%)")],
    )
    return styled(bars.properties(height=_height(len(order), 40)))


def moves(table, names, color):
    """Largest price changes, as bars from zero."""
    t = table.assign(Chain=table["supermarket"].map(lambda x: names.get(x, x)))
    t["Product"] = t["Chain"] + ": " + t["name"].str.slice(0, 34)
    # Invisible suffix: two products cut to the same label must stay two bars, not stack into one
    t["Product"] = [label + "\u200b" * i for i, label in enumerate(t["Product"])]
    t["text"] = t["change_pct"].map(lambda v: f"{v:+.0f}%")
    t["Was"] = t["old_price"].map(lambda v: f"€{v:.2f}")
    t["Now"] = t["new_price"].map(lambda v: f"€{v:.2f}")
    y = alt.Y("Product:N", sort=t["Product"].tolist(), axis=_axis(None, labelLimit=225, minExtent=225))
    bars = alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=16, color=color).encode(
        x=alt.X("change_pct:Q", title="Price change (%)", axis=alt.Axis(tickCount=5)), y=y,
        tooltip=["Product", "Was", "Now", alt.Tooltip("change_pct:Q", format="+.1f", title="Change (%)")],
    )
    return styled(bars.properties(height=_height(len(t), 28)))


def category_change(table, names, categories):
    """Average price change per category and chain; only cells with enough products exist."""
    t = table.assign(Chain=table["chain"].map(lambda x: names.get(x, x)), Category=table["category"].map(categories))
    t["text"] = t["average_pct"].map(lambda v: "0.0%" if abs(v) < 0.05 else f"{v:+.1f}%")
    limit = max(5, t["average_pct"].abs().max())
    base = alt.Chart(t).encode(
        x=alt.X("Chain:N", axis=_axis(None, orient="top", labelAngle=0)),
        y=alt.Y("Category:N", sort=sorted(t["Category"].unique()), axis=_axis(None)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color("average_pct:Q", scale=alt.Scale(domain=[-limit, 0, limit], range=[FALL, "#F2F3F1", RISE]),
                        legend=alt.Legend(title="Change (%)", orient="right")),
        tooltip=["Category", "Chain", alt.Tooltip("text:N", title="Average change"), alt.Tooltip("products:Q", title="Products")],
    )
    text = base.mark_text(fontSize=12).encode(
        text="text:N", color=alt.condition(f"abs(datum.average_pct) >= {0.6 * limit}", alt.value("#FFFFFF"), alt.value(INK)))
    return styled((cells + text).properties(height=_height(t["Category"].nunique(), 30)))
