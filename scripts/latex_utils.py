import pandas as pd

# constants for latex table highlighting

N_TOP_HIGHLIGHTED = 3
HIGHLIGHT_SHADE_FROM, HIGHLIGHT_SHADE_TO = 15, 50
HIGHLIGHT_SHADE_RANGE = HIGHLIGHT_SHADE_TO - HIGHLIGHT_SHADE_FROM
HIGHLIGHT_SHADE_STEPSIZE = (HIGHLIGHT_SHADE_RANGE // (N_TOP_HIGHLIGHTED - 1))
HIGHLIGHT_STYLES = [
    f"cellcolor:{{tab_color!{HIGHLIGHT_SHADE_FROM + i * HIGHLIGHT_SHADE_STEPSIZE}}}"
    for i in range(N_TOP_HIGHLIGHTED)
]

def highlight_top_n_cells(styler, df, styles, ascending=False):
    def make_style_func(column):
        sorted_vals = df.sort_values(column, ascending=ascending)[column].values

        def style_func(v):
            for i in range(min(len(sorted_vals), len(styles))):
                if (not ascending and v >= sorted_vals[i]) or (ascending and v <= sorted_vals[i]):
                    return styles[i]
            return ""

        return style_func

    for column in df.columns:
        styler = styler.map(make_style_func(column), subset=column)
    return styler

def print_latex_table(results, scene_names, ascending=False, PRECISION=3):
    df = pd.DataFrame.from_dict(results, orient="index", columns=scene_names)
    df["Average"] = df.mean(axis=1)
    df = df.round(PRECISION)

    styler = df.style.pipe(
        highlight_top_n_cells,
        df=df,
        styles=HIGHLIGHT_STYLES[::-1],
        ascending=ascending,
    )
    styler = styler.format(precision=PRECISION)

    tex_path = "output.tex"
    styler.to_latex(tex_path, hrules=True)
    print(styler.to_latex(hrules=True))


def highlight_top_n_cells_by_column(styler, df, styles, ascending=False, ascending_by_col=None):
    ascending_by_col = ascending_by_col or {}

    def make_style_func(column):
        column_ascending = ascending_by_col.get(column, ascending)
        sorted_vals = df[column].dropna().sort_values(ascending=column_ascending).values

        def style_func(v):
            if pd.isna(v):
                return ""
            for i in range(min(len(sorted_vals), len(styles))):
                if (not column_ascending and v >= sorted_vals[i]) or (
                    column_ascending and v <= sorted_vals[i]
                ):
                    return styles[i]
            return ""

        return style_func

    for column in df.columns:
        styler = styler.map(make_style_func(column), subset=column)
    return styler


def print_latex_table_mesh_nvs(
    results,
    metric_names,
    ascending=False,
    PRECISION=3,
    ascending_by_col=None,
    column_formatters=None,
):
    df = pd.DataFrame.from_dict(results, orient="index", columns=metric_names)
    df = df.round(PRECISION)

    styler = df.style.pipe(
        highlight_top_n_cells_by_column,
        df=df,
        styles=HIGHLIGHT_STYLES[::-1],
        ascending=ascending,
        ascending_by_col=ascending_by_col,
    )
    styler = styler.format(formatter=column_formatters, precision=PRECISION)

    tex_path = "output.tex"
    styler.to_latex(tex_path, hrules=True)
    print(styler.to_latex(hrules=True))