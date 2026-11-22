import math
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple


_KNOWN_NODE_LABELS: Dict[str, str] = {
    "hub_lead_channel_req": "Hub:\nLead×Chan\n×Req",
    "time_ym_wknd": "Time:\nYearMo\n×Wknd",
    "product_room_meal_rate": "Product:\nRoom×Meal\n×Rate",
    "stay_nights_wknd": "Stay:\nNights\n×Wknd",
    "party_adults_child_park": "Party:\nAdult×Chd\n×Park",
    "history_ret_cancel_comp": "History:\nRet×Canc\n×Comp",
    "days_before_arrival_is_missing": "Missing:\nDays\nBefore",
    "nightly_rate_idr_is_missing": "Missing:\nNightly\nRate",
    "num_adults_is_missing": "Missing:\nNum\nAdults",
    "special_requests_is_missing": "Missing:\nSpecial\nReq",
    "booking_group_size": "Booking\nGroup\nSize",
    "arrival_day": "Arrival\nDay",
    "total_spend_idr": "Total\nSpend\n(IDR)",
    "lead_to_stay_ratio": "Lead/Stay\nRatio",
    "rate_vs_room_month": "Rate vs\nRoom-Mo",
    "days_before_arrival": "Days\nBefore\nArrival",
    "nightly_rate_idr": "Nightly\nRate\n(IDR)",
    "prior_completed_bookings": "Prior\nCompleted",
    "prior_cancellations": "Prior\nCancels",
    "is_returning_guest": "Returning\nGuest",
    "special_requests": "Special\nRequests",
}


def _format_node_label(fname: str, is_root: bool = False) -> str:
    """Memformat label fitur agar muat rapi di dalam simpul lingkaran graf."""
    if fname in _KNOWN_NODE_LABELS:
        txt = _KNOWN_NODE_LABELS[fname]
    else:
        clean = (
            fname.replace("booking_channel_", "Ch:\n")
            .replace("meal_package_", "Meal:\n")
            .replace("room_category_", "Room:\n")
            .replace("_is_missing", "\n(Miss)")
            .replace("_", " ")
        )
        lines: List[str] = []
        for part in clean.split("\n"):
            wrapped = textwrap.wrap(part, width=10)
            lines.extend(wrapped if wrapped else [""])
        txt = "\n".join(lines[:3])
    return f"{txt}\n[TAN Root]" if is_root else txt


def _compute_tidy_tan_layout(
    n_features: int, tan_tree_edges: List[Tuple[int, int, float]], tan_root: Optional[int], np: Any
) -> Tuple[Any, Any, Any, Any, float, float]:
    """Menghitung koordinat hierarkis tanpa perpotongan edge untuk pohon TAN Chow-Liu."""
    children: Dict[int, List[int]] = {i: [] for i in range(n_features)}
    in_deg = np.zeros(n_features, dtype=int)
    edge_weights: Dict[Tuple[int, int], float] = {}
    for p_idx, c_idx, w in tan_tree_edges:
        if 0 <= p_idx < n_features and 0 <= c_idx < n_features:
            children[p_idx].append(c_idx)
            in_deg[c_idx] += 1
            edge_weights[(p_idx, c_idx)] = float(w)

    if tan_root is not None and 0 <= int(tan_root) < n_features:
        root_idx = int(tan_root)
    else:
        roots = [i for i in range(n_features) if in_deg[i] == 0]
        root_idx = roots[0] if roots else 0

    subtree_w: Dict[int, int] = {}
    depth_map: Dict[int, int] = {}
    visited = set()

    def _measure(u: int, d: int) -> int:
        visited.add(u)
        depth_map[u] = d
        kids = [v for v in children[u] if v not in visited]
        children[u] = kids
        if not kids:
            subtree_w[u] = 1
            return 1
        total = sum(_measure(v, d + 1) for v in kids)
        kids.sort(key=lambda v: (subtree_w[v], edge_weights.get((u, v), 0.0)))
        subtree_w[u] = max(1, total)
        return subtree_w[u]

    _measure(root_idx, 1)
    for idx in range(n_features):
        if idx not in visited:
            depth_map[idx] = 1
            subtree_w[idx] = 1

    total_leaves = max(1, subtree_w[root_idx])
    max_depth = max(depth_map.values()) if depth_map else 1
    node_r = 0.35 if total_leaves <= 10 else max(0.24, 3.3 / total_leaves)
    slot_step = max(2.0 * node_r + 0.14, 0.84)
    total_span = max(1, total_leaves - 1) * slot_step
    x_coords = np.zeros(n_features, dtype=np.float64)

    def _assign_x(u: int, x_left: float, x_right: float) -> None:
        kids = children[u]
        if not kids:
            x_coords[u] = 0.5 * (x_left + x_right)
            return
        total_w = max(sum(subtree_w[v] for v in kids), 1)
        cur = x_left
        span = x_right - x_left
        for v in kids:
            nxt = cur + span * (subtree_w[v] / total_w)
            _assign_x(v, cur, nxt)
            cur = nxt
        x_coords[u] = 0.5 * (x_coords[kids[0]] + x_coords[kids[-1]])

    _assign_x(root_idx, -0.5 * total_span - 0.5 * slot_step, 0.5 * total_span + 0.5 * slot_step)
    x_coords -= x_coords[root_idx]

    if max_depth <= 1:
        layer_ys = {1: 0.0}
    else:
        ys = np.linspace(1.08, -2.35, max_depth)
        layer_ys = {d + 1: float(ys[d]) for d in range(max_depth)}

    feat_pos = np.column_stack([x_coords, [layer_ys[depth_map[i]] for i in range(n_features)]])
    target_pos = np.array([0.0, 2.28], dtype=np.float64)
    x_pad = node_r + 0.32
    x_lim = np.array([float(np.min(x_coords)) - x_pad, float(np.max(x_coords)) + x_pad])
    return target_pos, feat_pos, x_lim, np.array([-2.90, 2.90]), node_r, 0.46


def render_bayesian_network_graph(
    nb: Any,
    feature_names: Optional[List[str]] = None,
    target_name: str = "is_canceled",
    save_path: str = "output/figures/naive_bayes_graph.png",
) -> str:
    """Memvisualisasikan struktur DAG Naive Bayes standar vs TAN Chow-Liu ke file gambar PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrowPatch

    if nb.use_tan and nb.tan_feature_names_ is not None:
        feature_names = list(nb.tan_feature_names_)
    elif feature_names is None:
        if nb.feature_names_in_ is not None and len(nb.feature_names_in_) == nb.n_features_in_:
            feature_names = list(nb.feature_names_in_)
        else:
            feature_names = [f"X_{i}" for i in range(nb.n_features_in_)]

    n_features = len(feature_names)
    abs_save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_save_path), exist_ok=True)

    primary_purple, target_edge_col = "#5b21b6", "#2e1065"
    light_purple, node_border_purple = "#f3e8ff", "#7c3aed"
    tan_edge_color, tan_root_fill, class_edge_color = "#d95f02", "#ffedd5", "#8b5cf6"
    has_tan = nb.use_tan and len(nb.tan_tree_edges_) > 0

    if has_tan:
        t_pos_tan, f_pos_tan, xlim_tan, ylim_tan, r_tan_node, r_tan_target = _compute_tidy_tan_layout(
            n_features, nb.tan_tree_edges_, nb.tan_root_, np
        )
        tan_width_ratio = float((xlim_tan[1] - xlim_tan[0]) / 5.70)
        fig, axes = plt.subplots(
            1, 2, figsize=(12.2 * (1.0 + tan_width_ratio), 11.2), dpi=200,
            gridspec_kw={"width_ratios": [1.0, max(1.05, tan_width_ratio)], "wspace": 0.06},
        )
        subtitle_tan = (
            f"Tree-Augmented Naive Bayes (TAN - Chow-Liu MWST)\n"
            f"({n_features} Core Features, {len(nb.tan_tree_edges_)} Conditional Dependency Edges)"
        )
        panels = [
            (axes[0], "Standard Naive Bayes Structure\n(Conditional Independence Assumption)", False),
            (axes[1], subtitle_tan, True),
        ]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 11.5), dpi=200)
        panels = [(ax, f"Bayesian Network Structure (Naive Bayes • {n_features} Features)", False)]

    text_fit_records: List[Tuple[Any, Any, Circle, float]] = []
    for ax, title, draw_tan in panels:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14, color="#240b46")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if draw_tan:
            target_pos, feat_pos, node_r, target_r = t_pos_tan, f_pos_tan, r_tan_node, r_tan_target
            ax.set_xlim(float(xlim_tan[0]), float(xlim_tan[1]))
            ax.set_ylim(float(ylim_tan[0]), float(ylim_tan[1]))
        else:
            star_r = max(2.12, n_features * 0.155)
            node_r = min(0.37, max(0.20, 0.43 * (2.0 * math.pi * star_r / max(n_features, 1))))
            target_r = max(0.46, node_r * 1.25)
            target_pos = np.array([0.0, 0.0], dtype=np.float64)
            angles = np.pi / 2.0 - np.linspace(0.0, 2.0 * np.pi, n_features, endpoint=False)
            feat_pos = np.column_stack([star_r * np.cos(angles), star_r * np.sin(angles)])
            lim_box = star_r + node_r + 0.36
            ax.set_xlim(-lim_box, lim_box)
            ax.set_ylim(-lim_box, lim_box)

        target_circle = Circle(tuple(target_pos), radius=target_r, facecolor=primary_purple, edgecolor=target_edge_col, linewidth=2.4, zorder=6)
        ax.add_patch(target_circle)
        feat_circles: List[Circle] = []
        for i in range(n_features):
            is_root = draw_tan and (nb.tan_root_ == i)
            c_patch = Circle(
                tuple(feat_pos[i]),
                radius=node_r * (1.07 if is_root else 1.0),
                facecolor=tan_root_fill if is_root else light_purple,
                edgecolor=tan_edge_color if is_root else node_border_purple,
                linewidth=2.5 if is_root else 1.7,
                zorder=6,
            )
            ax.add_patch(c_patch)
            feat_circles.append(c_patch)

        for i in range(n_features):
            if draw_tan:
                dx = float(feat_pos[i, 0] - target_pos[0])
                if nb.tan_root_ == i:
                    rad_val = 0.0
                elif abs(dx) < 0.25:
                    rad_val = 0.24
                else:
                    rad_val = float(np.clip(-0.07 * dx, -0.24, 0.24))
                conn_style = f"arc3,rad={rad_val:.3f}"
            else:
                conn_style = "arc3,rad=0.0"

            ax.add_patch(
                FancyArrowPatch(
                    posA=tuple(target_pos), posB=tuple(feat_pos[i]), patchA=target_circle, patchB=feat_circles[i],
                    connectionstyle=conn_style, arrowstyle="-|>,head_length=6.5,head_width=4.0",
                    color=class_edge_color, alpha=0.28 if draw_tan else 0.78, linestyle="--" if draw_tan else "-",
                    linewidth=1.15 if draw_tan else 1.7, shrinkA=1.0, shrinkB=2.0, zorder=2,
                )
            )

        if draw_tan and len(nb.tan_tree_edges_) > 0:
            max_w = max([w for _, _, w in nb.tan_tree_edges_] + [1e-6])
            for p_idx, c_idx, weight in nb.tan_tree_edges_:
                if p_idx >= n_features or c_idx >= n_features:
                    continue
                dx_edge = float(feat_pos[c_idx, 0] - feat_pos[p_idx, 0])
                rad_tan = 0.0 if abs(dx_edge) < 0.1 else (0.06 if dx_edge < 0 else -0.06)
                ax.add_patch(
                    FancyArrowPatch(
                        posA=tuple(feat_pos[p_idx]), posB=tuple(feat_pos[c_idx]),
                        patchA=feat_circles[p_idx], patchB=feat_circles[c_idx],
                        connectionstyle=f"arc3,rad={rad_tan:.3f}", arrowstyle="-|>,head_length=8,head_width=5",
                        color=tan_edge_color, alpha=0.95, linewidth=1.6 + 2.6 * (weight / max_w),
                        shrinkA=1.0, shrinkB=2.5, zorder=4,
                    )
                )
                if weight >= 0.015:
                    mid = 0.48 * feat_pos[p_idx] + 0.52 * feat_pos[c_idx]
                    ax.text(
                        mid[0], mid[1], f"{weight:.3f}", fontsize=8.0, fontweight="bold", color="#7c2d12",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.20", fc="#fff7ed", ec=tan_edge_color, alpha=0.95, lw=0.9),
                        zorder=5,
                    )

        t_txt = ax.text(
            target_pos[0], target_pos[1], f"Target\n({target_name})", fontsize=10.0, fontweight="bold",
            color="white", ha="center", va="center", linespacing=1.18, zorder=7,
        )
        text_fit_records.append((ax, t_txt, target_circle, 10.0))
        for i, fname in enumerate(feature_names):
            is_root = draw_tan and (nb.tan_root_ == i)
            base_fs = 8.4 if not is_root else 8.1
            f_txt = ax.text(
                feat_pos[i, 0], feat_pos[i, 1], _format_node_label(fname, is_root=is_root),
                fontsize=base_fs, fontweight="bold", color="#431407" if is_root else "#1e1b4b",
                ha="center", va="center", linespacing=1.12, zorder=7,
            )
            text_fit_records.append((ax, f_txt, feat_circles[i], base_fs))

    legend_handles = [
        mpatches.Patch(facecolor=primary_purple, edgecolor=target_edge_col, label=f"Class Variable ({target_name})"),
        mpatches.Patch(facecolor=light_purple, edgecolor=node_border_purple, label="Predictor Feature ($X_i$)"),
        Line2D([0], [0], color=class_edge_color, lw=2.0, linestyle="--" if has_tan else "-", label="Class-to-Feature Dependency $P(X_i \\mid Y)$"),
    ]
    if has_tan:
        legend_handles.extend([
            mpatches.Patch(facecolor=tan_root_fill, edgecolor=tan_edge_color, label="TAN Chow-Liu Root Feature ($X_{root}$)"),
            Line2D([0], [0], color=tan_edge_color, lw=2.8, label="TAN Tree Edge $P(X_i \\mid X_{\\pi(i)}, Y)$ with $I(X_i; X_j \\mid Y)$"),
        ])

    fig.legend(
        handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 3),
        fontsize=10.0, frameon=True, facecolor="#fafafa", edgecolor="#d4d4d8", bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.10, wspace=0.06)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, txt_artist, circ_patch, base_fs in text_fit_records:
        cx, cy = circ_patch.center
        p_center = ax.transData.transform((cx, cy))
        p_edge = ax.transData.transform((cx + circ_patch.radius, cy))
        max_r_px = 0.85 * float(np.hypot(p_edge[0] - p_center[0], p_edge[1] - p_center[1]))
        bbox = txt_artist.get_window_extent(renderer=renderer)
        half_diag_px = 0.5 * math.hypot(bbox.width, bbox.height)
        if half_diag_px > max_r_px and half_diag_px > 1e-6:
            txt_artist.set_fontsize(max(4.2, base_fs * (max_r_px / half_diag_px)))

    fig.savefig(abs_save_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return abs_save_path
