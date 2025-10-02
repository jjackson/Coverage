
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

FA_VERSION = "field-analysis v14 (value+baseline + completion+baseline; shared bins; tooltips; remap)"
def dbg(msg: str):
    try:
        print(f"DEBUG: {msg}")
    except Exception:
        print(f"DEBUG: {msg}")

def add_field_analysis_to_dashboard(dashboard_html_path: str,
                                    index_rel_path: str = 'field_analysis_index.html',
                                    section_title: str = 'Field Analysis',
                                    insert_marker: str = '<!-- FIELD_ANALYSIS_LINKS -->') -> list:
    print('DEBUG: add_field_analysis_to_dashboard helper v9 loaded')
    try:
        if not isinstance(dashboard_html_path, (str, Path)) and not hasattr(dashboard_html_path, '__fspath__'):
            print(f"DEBUG: unexpected dashboard_html_path type: {type(dashboard_html_path).__name__}; no-op")
            return []
        path_obj = Path(dashboard_html_path.__fspath__() if hasattr(dashboard_html_path, '__fspath__') else dashboard_html_path)
        if not path_obj.exists():
            print(f"DEBUG: dashboard file not found: {path_obj}")
            return []
        html = path_obj.read_text(encoding='utf-8', errors='replace')
        card_html = f"""
        <div class="fa-card" style="background:#fff;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.08);padding:16px;margin:16px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">
            <div style="font-weight:600;color:#2c3e50;">{section_title}</div>
            <a href="{index_rel_path}" style="text-decoration:none;background:#1f6feb;color:#fff;padding:8px 12px;border-radius:6px;">Open Field Analysis</a>
          </div>
          <div style="color:#6b7280;font-size:13px;margin-top:6px;">Explore value distributions, fraud overlays, and baseline comparisons.</div>
        </div>
        """
        changed = False
        if insert_marker in html:
            html = html.replace(insert_marker, card_html + insert_marker)
            changed = True
        else:
            lowered = html.lower()
            for tag in ['</main>', '</section>', '</div>', '</body>']:
                pos = lowered.rfind(tag)
                if pos != -1:
                    html = html[:pos] + card_html + html[pos:]
                    changed = True
                    break
        if changed:
            path_obj.write_text(html, encoding='utf-8', errors='replace')
        result = [str(path_obj)] if changed else []
        #print(f"DEBUG: add_field_analysis_to_dashboard returning {result}")
        return result
    except Exception as e:
        print(f"DEBUG: add_field_analysis_to_dashboard error: {e}")
        return []

class BinSpec:
    def __init__(self, values: pd.Series, n_bins: int = 50, right: bool = True):
        vals = pd.to_numeric(values.dropna(), errors='coerce')
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            self.min_val, self.max_val = 0.0, 1.0
        else:
            self.min_val = float(vals.min())
            self.max_val = float(vals.max())
            if self.min_val == self.max_val:
                self.max_val = self.min_val + 1.0
        self.n_bins = int(n_bins)
        self.edges = np.linspace(self.min_val, self.max_val, self.n_bins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2.0
        self.right = bool(right)
        self.bin_size = (self.max_val - self.min_val) / self.n_bins
        #dbg(f"BinSpec: min={self.min_val:.6f}, max={self.max_val:.6f}, bins={self.n_bins}, bin_size={self.bin_size:.6f}")
        #dbg(f"BinSpec: first_edge={self.edges[0]:.6f}, last_edge={self.edges[-1]:.6f}; first_center={self.centers[0]:.6f}, last_center={self.centers[-1]:.6f}")

    def assign(self, values: pd.Series):
        return pd.cut(values, bins=self.edges, include_lowest=True, right=self.right)

    def value_to_pct(self, x: float) -> float:
        span = (self.max_val - self.min_val)
        if span <= 0:
            return 0.0
        return (x - self.min_val) / span * 100.0

    def bin_width_pct(self) -> float:
        return 100.0 / self.n_bins

def build_histogram(values: pd.Series, binspec: BinSpec):
    cats = binspec.assign(values)
    counts = cats.value_counts(sort=False)
    total = int(values.notna().sum())
    percentages = (counts / total * 100.0).fillna(0.0).values if total > 0 else np.zeros(binspec.n_bins)
    #dbg(f"Histogram: total={total}, nonempty_bins={(percentages>0).sum()}/{binspec.n_bins}")
    return binspec.centers, percentages

def build_fraud_line(values: pd.Series, fraud_scores: pd.Series, binspec: BinSpec):
    idx = values.dropna().index.intersection(fraud_scores.dropna().index)
    if len(idx) == 0:
        dbg("Fraud line: no overlapping index after dropping NA.")
        return binspec.centers, np.array([np.nan]*binspec.n_bins)
    cats = binspec.assign(values.loc[idx])
    df = pd.DataFrame({'bin': cats, 'fraud': fraud_scores.loc[idx]})
    med = df.groupby('bin', sort=False)['fraud'].median()
    med = med.reindex(cats.cat.categories)
    vals = med.values.astype(float)
    xs = [round(float(x), 6) for x in binspec.centers[:3]] + ['...'] + [round(float(x), 6) for x in binspec.centers[-3:]]
    ys = [None if not np.isfinite(v) else round(float(v), 4) for v in list(vals[:3]) + [np.nan] + list(vals[-3:])]
    #dbg(f"Fraud line: Xcenters sample={xs}, Y(median) sample={ys}")
    return binspec.centers, vals

class FieldAnalysisPageGenerator:
    def __init__(self, fraud_results_df: pd.DataFrame, baseline_data):
        self.fraud_results = fraud_results_df.copy()
        self.baseline_data = baseline_data if isinstance(baseline_data, dict) else {}
        self.current_data = self.fraud_results[self.fraud_results['flw_id'] != 'BASELINE VALUES'].copy()
        dbg(f"fraud_results_df shape: {self.fraud_results.shape}")
        #dbg(f"baseline_data keys: {list(self.baseline_data.keys()) if isinstance(self.baseline_data, dict) else type(self.baseline_data)}")

    # ---------- Baseline JSON helpers ----------
    def _get_baseline_bins_json(self, value_field: str, fraud_col: str | None):
        try:
            fd = self.baseline_data.get('flw_distributions') if isinstance(self.baseline_data, dict) else None
            if not isinstance(fd, dict):
                return None, None, None, None
            block = fd.get(value_field)
            if not isinstance(block, dict):
                return None, None, None, None
            bins = block.get('bins')
            if not isinstance(bins, list) or len(bins) == 0:
                return None, None, None, None
            edges = [float(bins[0].get('bin_start'))]
            pct = []
            med = []
            for b in bins:
                pct.append(float(b.get('percentage', 0.0)))
                mfs = b.get('median_fraud_scores') or {}
                v = mfs.get(fraud_col) if fraud_col else None
                med.append(float(v) if v is not None else float('nan'))
                edges.append(float(b.get('bin_end')))
            meta = {
                'min_value': float(block.get('min_value', edges[0])),
                'max_value': float(block.get('max_value', edges[-1])),
                'bin_size': float(block.get('bin_size', (edges[-1]-edges[0]) / max(1, (len(edges)-1)))),
            }
            #dbg(f"Baseline JSON(bins) for '{value_field}' with '{fraud_col}': min={meta['min_value']:.6f}, max={meta['max_value']:.6f}, bin_size={meta['bin_size']:.6f}")
            return np.asarray(edges, float), np.asarray(pct, float), np.asarray(med, float), meta
        except Exception as e:
            #dbg(f"Baseline bins JSON read error for '{value_field}': {e}")
            return None, None, None, None

    def _align_baseline_to_current(self, binspec: BinSpec, base_edges, base_pct, base_med):
        cur_edges = binspec.edges
        n = binspec.n_bins
        # percentages via overlap
        out_pct = np.zeros(n, dtype=float)
        i = j = 0
        while i < len(base_pct) and j < n:
            b0, b1 = float(base_edges[i]), float(base_edges[i+1])
            c0, c1 = float(cur_edges[j]), float(cur_edges[j+1])
            overlap = min(b1, c1) - max(b0, c0)
            if overlap > 0:
                denom = (b1 - b0)
                frac = overlap/denom if denom > 0 else 0.0
                out_pct[j] += base_pct[i] * frac
            if b1 <= c1:
                i += 1
            else:
                j += 1
        s = float(out_pct.sum())
        if s > 0: out_pct *= (100.0 / s)
        # medians by center sampling
        out_med = np.full(n, np.nan, dtype=float)
        i = 0
        for j in range(n):
            c = float(binspec.centers[j])
            while i+1 < len(base_edges) and c >= float(base_edges[i+1]):
                i += 1
            if 0 <= i < len(base_pct):
                val = base_med[i]
                out_med[j] = float(val) if np.isfinite(val) else np.nan
        xs = [round(float(x),6) for x in list(binspec.centers[:3]) + list(binspec.centers[-3:])]
        ys = [None if not np.isfinite(v) else round(float(v),4) for v in list(out_med[:3]) + list(out_med[-3:])]
        #dbg(f"Baseline→Current aligned: pct_sum={out_pct.sum():.2f}; med sample @centers {xs} -> {ys}")
        return out_pct, out_med

    # ---------- Public API ----------
    def generate_all_field_pages(self, output_dir: str) -> list:
        print("DEBUG: Generating field analysis pages...")
        os.makedirs(output_dir, exist_ok=True)
        dbg(f"output_dir: {output_dir}")
        field_configs = self._get_field_configurations()
        if not field_configs:
            print("DEBUG: No eligible fields found for analysis pages (0 files will be generated).")
        generated_files = []
        total = len(field_configs)
        for i, cfg in enumerate(field_configs, 1):
            try:
                filename = f"field_analysis_{i:02d}_{cfg['field_name'].lower().replace(' ', '_')}.html"
                filepath = os.path.join(output_dir, filename)
                html = self._generate_field_page(cfg, i, total)
                with open(filepath, "w", encoding="utf-8", errors="replace") as f:
                    f.write(html)
                print(f"Generated field analysis page: {filename}")
                generated_files.append(filepath)
            except Exception as e:
                print(f"DEBUG: Error generating page for {cfg.get('field_name','?')}: {e}")
        index_path = self._generate_field_index(output_dir, field_configs)
        generated_files.append(index_path)
        #print(f"DEBUG: Successfully generated {len(generated_files)-1} field analysis pages")
        print(f"DEBUG: Field analysis pages complete: {len(generated_files)} files")
        return generated_files

    def _get_field_configurations(self):
        from .fraud_detection_core import FIELD_ANALYSIS_CONFIG
        cols = set(self.current_data.columns)
        sample_cols = sorted(list(cols))[:10]
        print(f"DEBUG: current_data columns (sample): {sample_cols} ... (total={len(cols)})")
        configs = []
        for value_field, cfg in FIELD_ANALYSIS_CONFIG.items():
            want = cfg.get('create_page', False)
            has_col = value_field in cols
            #print(f"DEBUG: check field '{value_field}': create_page={want}, present={has_col}")
            if want and has_col:
                configs.append({
                    'field_name': cfg['field_name'],
                    'value_field': value_field,
                    'value_fraud_score': cfg['fraud_score'],
                    'completion_field': cfg.get('completion_field'),
                    'completion_fraud_score': cfg.get('completion_fraud_score'),
                    'description': cfg.get('description', ''),
                    'n_bins': cfg.get('n_bins', 50),
                })
        print(f"DEBUG: eligible fields for pages: {[c['value_field'] for c in configs]} (n={len(configs)})")
        return configs

    def _generate_field_page(self, cfg, page_num, total_pages):
        field_name = cfg['field_name']
        value_field = cfg['value_field']
        value_fraud_score = cfg['value_fraud_score']
        comp_field = cfg.get('completion_field')
        comp_score = cfg.get('completion_fraud_score')
        n_bins = int(cfg.get('n_bins', 50))
        #dbg(f"begin field='{field_name}' value_field={value_field} fraud_score={value_fraud_score} n_bins={n_bins}")

        # ---------- Current (value) ----------
        values = pd.to_numeric(self.current_data[value_field], errors='coerce')
        fraud = pd.to_numeric(self.current_data[value_fraud_score], errors='coerce')
        binspec = BinSpec(values, n_bins=n_bins, right=True)
        centers, bar_pct = build_histogram(values, binspec)
        _, fraud_med = build_fraud_line(values, fraud, binspec)

        current_html = self._render_hist_with_line(
            centers=centers, bar_pct=bar_pct, fraud_med=fraud_med,
            binspec=binspec, value_field=value_field, value_fraud_score=value_fraud_score,
            values=values, panel_style="current"
        )

        # ---------- Baseline (value) ----------
        base_edges, base_bar_pct0, base_fraud_med0, base_meta = self._get_baseline_bins_json(value_field, value_fraud_score)
        baseline_html = ""
        if base_edges is not None:
            base_bar_pct, base_fraud_med = self._align_baseline_to_current(binspec, base_edges, base_bar_pct0, base_fraud_med0)
            baseline_html = f"""
            <section class="analysis-section">
              <h2>Baseline — Value Distribution + Fraud Overlay</h2>
              {self._render_hist_with_line(
                    centers=binspec.centers,
                    bar_pct=np.array(base_bar_pct),
                    fraud_med=np.array(base_fraud_med),
                    binspec=binspec,
                    value_field=value_field,
                    value_fraud_score=value_fraud_score,
                    values=None,
                    panel_style="baseline",
                    override_range=(base_meta.get('min_value'), base_meta.get('max_value'))
                )}
            </section>
            """

        # ---------- Completion (current) ----------
        comp_current_html = ""
        comp_baseline_html = ""
        if comp_field and comp_field in self.current_data.columns:
            comp_values = pd.to_numeric(self.current_data[comp_field], errors='coerce')
            comp_binspec = BinSpec(comp_values, n_bins=n_bins, right=True)
            if comp_score and comp_score in self.current_data.columns:
                _, comp_fraud_med = build_fraud_line(comp_values, pd.to_numeric(self.current_data[comp_score], errors='coerce'), comp_binspec)
                comp_label = comp_score
            else:
                comp_fraud_med = np.array([np.nan]*comp_binspec.n_bins)
                comp_label = comp_score or ''
            comp_centers, comp_bar_pct = build_histogram(comp_values, comp_binspec)

            comp_current_html = f"""
            <section class="analysis-section">
              <h2>Completion — Missingness Distribution + Fraud Overlay</h2>
              {self._render_hist_with_line(
                    centers=comp_centers,
                    bar_pct=comp_bar_pct,
                    fraud_med=comp_fraud_med,
                    binspec=comp_binspec,
                    value_field=comp_field,
                    value_fraud_score=comp_label,
                    values=comp_values,
                    panel_style="current"
                )}
            </section>
            """

            # ---------- Completion (baseline) ----------
            base_edges_c, base_bar_pct0_c, base_fraud_med0_c, base_meta_c = self._get_baseline_bins_json(comp_field, comp_score if comp_score else None)
            if base_edges_c is not None:
                base_bar_pct_c, base_fraud_med_c = self._align_baseline_to_current(comp_binspec, base_edges_c, base_bar_pct0_c, base_fraud_med0_c)
                comp_baseline_html = f"""
                <section class="analysis-section">
                  <h2>Baseline Completion — Missingness Distribution + Fraud Overlay</h2>
                  {self._render_hist_with_line(
                        centers=comp_binspec.centers,
                        bar_pct=np.array(base_bar_pct_c),
                        fraud_med=np.array(base_fraud_med_c),
                        binspec=comp_binspec,
                        value_field=comp_field,
                        value_fraud_score=comp_label,
                        values=None,
                        panel_style="baseline",
                        override_range=(base_meta_c.get('min_value'), base_meta_c.get('max_value'))
                    )}
                </section>
                """

        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Field Analysis: {self._esc(field_name)}</title>
<style>
{self._css()}
</style>
</head>
<body>
<div class="container">
  <header class="analysis-header">
    <h1>Field Analysis: {self._esc(field_name)}</h1>
    <p class="field-description">{self._esc(cfg.get('description',''))}</p>
    <div class="header-meta">
      <span>Page {page_num} of {total_pages}</span>
      <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
  </header>
  {self._generate_navigation(page_num, total_pages)}
  <section class="analysis-section">
    <h2>Value Distribution + Fraud Overlay</h2>
    {current_html}
  </section>
  {baseline_html}
  {comp_current_html}
  {comp_baseline_html}
  {self._generate_navigation(page_num, total_pages)}
</div>
</body>
</html>"""
        return html

    # ---------- Rendering & utilities ----------
    def _render_hist_with_line(self, centers, bar_pct, fraud_med, binspec: BinSpec,
                               value_field, value_fraud_score, values,
                               panel_style="current", override_range=None):
        if panel_style == "baseline":
            bar_class = "bar bar-baseline"
            line_class = "fraud-line fraud-line-baseline"
            point_class = "fraud-point fraud-point-baseline"
        else:
            bar_class = "bar"
            line_class = "fraud-line"
            point_class = "fraud-point"
        bar_pct = np.asarray(bar_pct, dtype=float)
        fraud_med = np.asarray(fraud_med, dtype=float)
        bar_max = float(np.max(bar_pct)) if len(bar_pct) else 1.0
        bar_max = bar_max if bar_max > 0 else 1.0
        bars = []
        width_pct = binspec.bin_width_pct()
        for i in range(binspec.n_bins):
            x_left_pct = binspec.value_to_pct(binspec.edges[i])
            height_pct = (bar_pct[i] / bar_max) * 100.0 if bar_max > 0 else 0.0
            fraud_txt = f"{fraud_med[i]:.3f}" if i < len(fraud_med) and np.isfinite(fraud_med[i]) else "—"
            title = f"% of FLWs: {bar_pct[i]:.1f}% | Median fraud: {fraud_txt}"
            bars.append(f'<div class="{bar_class}" style="left:{x_left_pct:.6f}%;width:{width_pct:.6f}%;height:{height_pct:.2f}%;" title="{title}"></div>')
        points = []
        circles = []
        for c, y in zip(centers, fraud_med):
            if np.isfinite(y):
                x_pct = binspec.value_to_pct(float(c))
                y_pct = (1.0 - float(y)) * 100.0
                points.append(f"{x_pct:.6f},{y_pct:.6f}")
                circles.append(f'<circle cx="{x_pct:.6f}%" cy="{y_pct:.6f}%" r="2" class="{point_class}"/>' )
        path_d = "M " + " L ".join(points) if points else ""
        if values is not None and isinstance(values, pd.Series):
            count_txt = f"{int(values.notna().sum())} FLWs"
            mean_txt = f"{np.nanmean(values):.4f}"
            rng_min, rng_max = binspec.min_val, binspec.max_val
        else:
            count_txt = "—"
            mean_txt = "—"
            if override_range and override_range[0] is not None and override_range[1] is not None:
                rng_min, rng_max = float(override_range[0]), float(override_range[1])
            else:
                rng_min, rng_max = binspec.min_val, binspec.max_val
        label = value_fraud_score if value_fraud_score else "median fraud (0–1)"
        stats_html = f"""
        <div class="histogram-stats">
          <div class="data-stats">
            <strong>Count:</strong> {count_txt} |
            <strong>Mean:</strong> {mean_txt} |
            <strong>Range:</strong> {rng_min:.4f} – {rng_max:.4f} |
            <strong>Bins:</strong> {binspec.n_bins}
          </div>
          <div class="fraud-legend">
            <span class="{line_class} legend-swatch"></span> Median {self._esc(label)} (0–1)
          </div>
        </div>
        """
        ticks = self._choose_ticks(binspec, value_field)
        tick_divs = []
        for tv in ticks:
            try:
                x_pct = binspec.value_to_pct(float(tv))
            except Exception:
                x_pct = 0.0
            label = self._format_tick(tv, value_field, binspec)
            tick_divs.append(f'<div class="tick" style="left:{x_pct:.6f}%"><span class="tick-mark"></span><span class="tick-label">{self._esc(label)}</span></div>')
        axis_html = f'<div class="x-axis">{"".join(tick_divs)}</div>'
        html = f"""
        <div class="hist-with-line">
          <div class="bars-layer">
            {''.join(bars)}
          </div>
          <svg class="fraud-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <path d="{path_d}" class="{line_class}" fill="none"/>
            {''.join(circles)}
          </svg>
          <div class="fraud-y-axis">
            <div class="y-label y-top">1.0</div>
            <div class="y-label y-mid">0.5</div>
            <div class="y-label y-bottom">0.0</div>
          </div>
        </div>
        {stats_html}
        {axis_html}
        """
        return html

    def _is_percentish(self, value_field, binspec):
        name = (value_field or '').lower()
        span = binspec.max_val - binspec.min_val
        if 'pct' in name or 'percent' in name or 'rate' in name:
            return True
        return (binspec.min_val >= -0.001 and binspec.max_val <= 1.001 and span > 0)

    def _choose_ticks(self, binspec, value_field):
        if self._is_percentish(value_field, binspec):
            return [0.0, 0.25, 0.5, 0.75, 1.0]
        return list(np.linspace(binspec.min_val, binspec.max_val, 5))

    def _format_tick(self, v, value_field, binspec):
        if self._is_percentish(value_field, binspec):
            return f"{round(float(v)*100):d}%"
        av = abs(float(v))
        if av >= 1000: return f"{v:.0f}"
        if av >= 10:   return f"{v:.1f}"
        return f"{v:.2f}"

    def _esc(self, s):
        if s is None:
            return ""
        s = str(s)
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;")
                 .replace("'", "&#x27;"))

    def _generate_navigation(self, current_page, total_pages):
        prev_link = ""
        next_link = ""
        if current_page > 1:
            prev_link = f'<a href="#" class="nav-btn prev-btn">⟨ Previous Field</a>'
        if current_page < total_pages:
            next_link = f'<a href="#" class="nav-btn next-btn">Next Field ⟩</a>'
        dashboard_link = '<a href="fraud_dashboard.html" class="nav-btn dashboard-btn">← Back to Dashboard</a>'
        index_link = '<a href="field_analysis_index.html" class="nav-btn index-btn">Field Index</a>'
        return f"""
        <nav class="page-navigation">
          <div class="nav-left">{prev_link}</div>
          <div class="nav-center">{dashboard_link}{index_link}<span class="page-indicator">Page {current_page} of {total_pages}</span></div>
          <div class="nav-right">{next_link}</div>
        </nav>
        """

    def _generate_field_index(self, output_dir, field_configs):
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Field Analysis Index</title>
<style>{self._css()}</style>
</head>
<body>
<div class="container">
  <header class="analysis-header">
    <h1>Field Analysis Index</h1>
    <p class="subtitle">Analysis of field distributions, fraud overlays, and baseline comparisons</p>
  </header>
  <nav class="page-navigation">
    <div class="nav-center"><a href="fraud_dashboard.html" class="nav-btn dashboard-btn">← Back to Dashboard</a></div>
  </nav>
  <section class="analysis-section">
    <h2>Available Field Analyses</h2>
    <div class="field-index">
      {self._links(field_configs)}
    </div>
  </section>
</div>
</body>
</html>
"""
        path = os.path.join(output_dir, "field_analysis_index.html")
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(html)
        return path

    def _links(self, field_configs):
        out = []
        for i, cfg in enumerate(field_configs, 1):
            fn = f"field_analysis_{i:02d}_{cfg['field_name'].lower().replace(' ', '_')}.html"
            out.append(f"""
            <div class="field-link-card">
              <h3><a href="{fn}">{self._esc(cfg['field_name'])}</a></h3>
              <p>{self._esc(cfg.get('description','No description available'))}</p>
              <div class="field-details">
                <span><strong>Value Field:</strong> {self._esc(cfg['value_field'])}</span>
                <span><strong>Fraud Score:</strong> {self._esc(cfg['value_fraud_score'])}</span>
              </div>
            </div>""")
        return "".join(out)

    def _css(self):
        return """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;color:#333;line-height:1.6}
.container{max-width:1200px;margin:0 auto;padding:20px}
.analysis-header{background:#fff;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1);padding:30px;margin-bottom:30px;text-align:center}
.analysis-header h1{color:#2c3e50;margin-bottom:10px}
.field-description{color:#7f8c8d;font-size:16px;margin-bottom:15px}
.header-meta{display:flex;justify-content:center;gap:30px;color:#7f8c8d;font-size:14px;border-top:1px solid #ecf0f1;padding-top:15px}
.page-navigation{display:flex;justify-content:space-between;align-items:center;background:#fff;border-radius:8px;padding:12px 16px;margin:16px 0 24px;box-shadow:0 2px 4px rgba(0,0,0,.06)}
.page-navigation .nav-btn{background:#f0f3f6;color:#2c3e50;text-decoration:none;padding:8px 12px;border-radius:6px;margin:0 6px}
.page-navigation .page-indicator{color:#7f8c8d;margin-left:8px}
.analysis-section{background:#fff;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.08);padding:20px;margin-bottom:24px}
.hist-with-line{position:relative;margin-top:10px;border:1px solid #ecf0f1;border-radius:8px;background:#fafafa}
.bars-layer{position:relative;height:240px;border-bottom:1px solid #ececec}
.bar{position:absolute;bottom:0;background:#6aa3ff;border-radius:3px 3px 0 0}
.bar-baseline{background:#cfd8e3}
.fraud-svg{position:absolute;left:0;right:0;top:0;bottom:0;width:100%;height:240px;pointer-events:none}
.fraud-line{stroke:#e74c3c;stroke-width:2.5;opacity:.95}
.fraud-point{fill:#e74c3c;opacity:.95}
.fraud-line-baseline{stroke:#f59e0b;stroke-dasharray:6 4}
.fraud-point-baseline{fill:#f59e0b}
.fraud-y-axis{position:absolute;right:0;top:0;height:240px;display:flex;flex-direction:column;justify-content:space-between;align-items:flex-end;font-size:11px;color:#7f8c8d;padding-right:2px}
.histogram-stats{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:10px}
.legend-swatch{display:inline-block;width:24px;height:3px;margin-right:6px;vertical-align:middle}
.x-axis{position:relative;height:34px;margin-top:6px;border-top:1px solid #ececec}
.x-axis .tick{position:absolute;top:-1px;transform:translateX(-50%);text-align:center}
.x-axis .tick-mark{display:block;width:1px;height:8px;background:#9aa4b2;margin:0 auto}
.x-axis .tick-label{display:block;margin-top:4px;font-size:11px;color:#667085;white-space:nowrap}
.field-index{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.field-link-card{background:#fff;padding:14px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.no-baseline{padding:12px 10px;color:#b54708;background:#fff7ed;border:1px solid #fed7aa;border-radius:6px;margin-top:8px}
"""
