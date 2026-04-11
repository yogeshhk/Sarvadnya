# Usage (run from ask_floorplans/):
#   streamlit run streamlit_main.py --server.fileWatcherType none

import sys
import os

# Point Python at the src/ sub-package and make relative file I/O (FAISS index
# files) resolve inside src/ where the existing index already lives.
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, _SRC_DIR)
os.chdir(_SRC_DIR)

import json
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from floorplan_system import FloorPlanSystem
from floor_plan_schema import EXAMPLE_FLOOR_PLAN


# ── Room colour palette (pastel, one per room type) ───────────────────────────

ROOM_COLORS = {
    "bedroom":     "#AED6F1",   # pale blue
    "bathroom":    "#A9DFBF",   # pale green
    "kitchen":     "#FAD7A0",   # pale amber
    "living_room": "#F9E79F",   # pale yellow
    "dining_room": "#D7BDE2",   # pale lavender
    "corridor":    "#D5DBDB",   # light grey
    "balcony":     "#ABEBC6",   # mint
    "storage":     "#CCD1D1",   # grey
    "office":      "#AEB6BF",   # blue-grey
    "laundry":     "#F5B7B1",   # pale rose
    "garage":      "#CACFD2",   # silver
    "other":       "#EAECEE",   # off-white
}

ZONE_FILLS = [
    "rgba(52,152,219,0.10)",
    "rgba(155,89,182,0.10)",
    "rgba(231,76,60,0.10)",
    "rgba(46,204,113,0.10)",
]


# ── Layout / geometry helpers ─────────────────────────────────────────────────

def auto_layout(rooms: list) -> list:
    """
    Assign _x, _y, _w, _h to rooms that lack explicit geometry, using a
    shelf-packing algorithm (left-to-right, wrap when row exceeds MAX_WIDTH).
    """
    PADDING = 0.4
    MAX_ROW_WIDTH = 15.0

    x = y = row_h = 0.0
    result = []
    for room in rooms:
        dims = room.get("dimensions", {})
        w = max(float(dims.get("length", 3.0)), 1.0)
        h = max(float(dims.get("width", 2.5)), 1.0)
        if x + w > MAX_ROW_WIDTH and x > 0:
            y += row_h + PADDING
            x = row_h = 0.0
        result.append({**room, "_x": x, "_y": y, "_w": w, "_h": h})
        row_h = max(row_h, h)
        x += w + PADDING
    return result


def room_polygon(room: dict) -> tuple[list, list]:
    """Return (xs, ys) closed polygon for a room."""
    verts = (room.get("geometry") or {}).get("vertices")
    if verts and len(verts) >= 3:
        xs = [v[0] for v in verts] + [verts[0][0]]
        ys = [v[1] for v in verts] + [verts[0][1]]
        return xs, ys
    x, y, w, h = room["_x"], room["_y"], room["_w"], room["_h"]
    return ([x, x + w, x + w, x, x],
            [y, y, y + h, y + h, y])


def centroid(room: dict) -> tuple[float, float]:
    verts = (room.get("geometry") or {}).get("vertices")
    if verts:
        return (sum(v[0] for v in verts) / len(verts),
                sum(v[1] for v in verts) / len(verts))
    return room["_x"] + room["_w"] / 2, room["_y"] + room["_h"] / 2


# ── 2D Floor-plan visualisation ───────────────────────────────────────────────

def visualise_floor_plan(floor_plan: dict) -> go.Figure:
    """
    Render a floor plan as an interactive Plotly figure.

    Layers (bottom → top):
      1. Zone bounding boxes (dashed outline + translucent fill)
      2. Room polygons (solid fill + dark border)
      3. Door connections between adjacent rooms (red dashed line)
      4. Room label annotations (name + area)
    """
    rooms = floor_plan.get("rooms", [])
    if not rooms:
        fig = go.Figure()
        fig.add_annotation(text="No rooms to display", showarrow=False,
                           font=dict(size=14, color="grey"))
        return fig

    # Auto-layout rooms that don't carry polygon vertices
    if any(not r.get("geometry") for r in rooms):
        rooms = auto_layout(rooms)

    room_by_id = {r["id"]: r for r in rooms}
    fig = go.Figure()

    # 1. Zone outlines ──────────────────────────────────────────────────────
    for zi, zone in enumerate(floor_plan.get("zones", [])):
        zone_rooms = [room_by_id[rid] for rid in zone.get("rooms", [])
                      if rid in room_by_id]
        if len(zone_rooms) < 2:
            continue
        all_xs, all_ys = [], []
        for zr in zone_rooms:
            xs, ys = room_polygon(zr)
            all_xs.extend(xs)
            all_ys.extend(ys)
        x0, x1 = min(all_xs) - 0.35, max(all_xs) + 0.35
        y0, y1 = min(all_ys) - 0.35, max(all_ys) + 0.35
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=ZONE_FILLS[zi % len(ZONE_FILLS)],
            line=dict(color="#95A5A6", width=1, dash="dot"),
            layer="below",
        )
        fig.add_annotation(
            x=x0 + 0.12, y=y1 - 0.12,
            text=f"<i>{zone['name'].replace('_', ' ').title()}</i>",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=8, color="#7F8C8D"),
        )

    # 2. Room polygons ──────────────────────────────────────────────────────
    for room in rooms:
        rtype  = room.get("type", "other")
        color  = ROOM_COLORS.get(rtype, "#EAECEE")
        rname  = room.get("name") or rtype.replace("_", " ").title()
        area   = room.get("area", 0.0)
        feats  = room.get("features", {})
        tip    = (
            f"<b>{rname}</b><br>"
            f"Type: {rtype}<br>"
            f"Area: {area:.1f} m²<br>"
            f"Windows: {feats.get('windows', 0)}  Doors: {feats.get('doors', 0)}"
            + ("<br>🌿 Balcony"  if feats.get("balcony")       else "")
            + ("<br>🛁 Ensuite"  if feats.get("ensuite")       else "")
            + ("<br>👟 Walk-in"  if feats.get("walk_in_closet") else "")
        )
        xs, ys = room_polygon(room)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself", fillcolor=color,
            line=dict(color="#2C3E50", width=2),
            mode="lines",
            name=rname,
            text=tip, hoverinfo="text",
            showlegend=False,
        ))

    # 3. Door / access connections ──────────────────────────────────────────
    for adj in floor_plan.get("adjacencies", []):
        if not adj.get("has_door"):
            continue
        r1 = room_by_id.get(adj["room1"])
        r2 = room_by_id.get(adj["room2"])
        if not (r1 and r2):
            continue
        c1x, c1y = centroid(r1)
        c2x, c2y = centroid(r2)
        label = adj.get("description") or f"{adj['room1']} ↔ {adj['room2']}"
        fig.add_trace(go.Scatter(
            x=[c1x, c2x], y=[c1y, c2y],
            mode="lines+markers",
            line=dict(color="#E74C3C", width=1.5, dash="dot"),
            marker=dict(symbol="circle-open", size=6, color="#E74C3C"),
            name=label, hoverinfo="name",
            showlegend=False,
        ))

    # 4. Room label annotations ─────────────────────────────────────────────
    for room in rooms:
        rname = room.get("name") or room.get("type", "").replace("_", " ").title()
        area  = room.get("area", 0.0)
        cx, cy = centroid(room)
        fig.add_annotation(
            x=cx, y=cy,
            text=f"<b>{rname}</b><br>"
                 f"<span style='font-size:9px'>{area:.1f} m²</span>",
            showarrow=False,
            font=dict(size=9, color="#1A252F"),
            bgcolor="rgba(255,255,255,0.78)",
            borderpad=3,
        )

    plan_name  = floor_plan.get("name", "Floor Plan")
    total_area = floor_plan.get("total_area", 0)
    fig.update_layout(
        title=dict(text=f"<b>{plan_name}</b>  —  {total_area:.1f} m²",
                   font=dict(size=14)),
        xaxis=dict(title="metres", scaleanchor="y", scaleratio=1,
                   showgrid=True, gridcolor="#EBEBEB", zeroline=False),
        yaxis=dict(title="metres", showgrid=True, gridcolor="#EBEBEB",
                   zeroline=False),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="closest",
    )
    return fig


# ── Summary metrics ───────────────────────────────────────────────────────────

def show_plan_metrics(plan: dict):
    rooms = plan.get("rooms", [])
    type_counts: dict[str, int] = {}
    for r in rooms:
        t = r.get("type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1

    cols = st.columns(min(len(type_counts) + 1, 6))
    cols[0].metric("Total area", f"{plan.get('total_area', 0):.1f} m²")
    for i, (rtype, cnt) in enumerate(list(type_counts.items())[:5], 1):
        cols[i].metric(rtype.replace("_", " ").title(), cnt)


# ── Cached system init ────────────────────────────────────────────────────────

@st.cache_resource
def get_system() -> FloorPlanSystem:
    return FloorPlanSystem(vector_store_type="faiss")


# ── App layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Ask Floorplans", page_icon="🏠", layout="wide")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Export it and restart the app.")
    st.stop()

system = get_system()

# Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 Ask Floorplans")
    st.markdown("AI-powered floor plan generation, search, and modification.")
    st.divider()

    if st.button("📥 Load example 2BHK plan", use_container_width=True):
        try:
            system.store_floor_plan(EXAMPLE_FLOOR_PLAN)
            st.success("Loaded example plan.")
            st.rerun()
        except Exception as e:
            st.warning(str(e))

    st.divider()
    st.markdown("**Stored plans**")
    all_plans = system.list_floor_plans()
    if all_plans:
        for p in all_plans:
            st.markdown(
                f"- **{p['name']}**  "
                f"{p['bedroom_count']}BR · {p['bathroom_count']}BA · {p['total_area']:.0f} m²"
            )
    else:
        st.caption("No plans yet. Generate one or load the example.")

# Main tabs ───────────────────────────────────────────────────────────────────
tab_gen, tab_query, tab_modify, tab_browse = st.tabs(
    ["✨ Generate", "💬 Query", "🔧 Modify", "📋 Browse"]
)

# ── Generate ──────────────────────────────────────────────────────────────────
with tab_gen:
    st.subheader("Generate a Floor Plan")
    st.markdown(
        "Describe the unit in plain English. The AI will create a structured floor plan "
        "and visualise it below."
    )
    description = st.text_area(
        "Description",
        placeholder=(
            "e.g. 3 bedroom apartment with 2 bathrooms, open-plan kitchen and "
            "living area, master ensuite, balcony, about 1 200 sq ft"
        ),
        height=90,
    )
    if st.button("Generate", type="primary") and description.strip():
        with st.spinner("Generating floor plan…"):
            try:
                fp = system.generate_floor_plan(description)
                st.session_state["last_generated"] = fp
            except Exception as e:
                st.error(f"Generation failed: {e}")

    if "last_generated" in st.session_state:
        fp = st.session_state["last_generated"]
        st.success(
            f"Generated: **{fp.get('name', 'Floor Plan')}** — "
            f"{fp.get('total_area', 0):.1f} m²"
        )
        st.plotly_chart(visualise_floor_plan(fp), use_container_width=True)
        show_plan_metrics(fp)
        with st.expander("View raw JSON"):
            st.json(fp)

# ── Query ─────────────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask About Stored Plans")
    st.markdown("Ask anything about the floor plans currently in the system.")

    if "q_messages" not in st.session_state:
        st.session_state.q_messages = []

    for msg in st.session_state.q_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("e.g. Which plans have a balcony? What is the largest plan?"):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = system.answer_question(q)
            st.markdown(answer)
        st.session_state.q_messages.append({"role": "user",      "content": q})
        st.session_state.q_messages.append({"role": "assistant", "content": answer})

# ── Modify ────────────────────────────────────────────────────────────────────
with tab_modify:
    st.subheader("Modify a Floor Plan")

    all_plans = system.list_floor_plans()
    if not all_plans:
        st.info("No plans stored yet. Generate or load one first.")
    else:
        plan_labels = {p["name"]: p["id"] for p in all_plans}
        selected_name = st.selectbox("Select plan to modify", list(plan_labels.keys()))
        selected_id   = plan_labels[selected_name]
        current_plan  = system.get_floor_plan(selected_id)

        if current_plan:
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("**Before**")
                st.plotly_chart(
                    visualise_floor_plan(current_plan),
                    use_container_width=True, key="before_viz"
                )

            command = st.text_input(
                "Modification command",
                placeholder=(
                    "e.g. Add a balcony to the living room  /  "
                    "Make the kitchen larger  /  Remove the storage room"
                ),
            )
            if st.button("Apply", type="primary") and command.strip():
                with st.spinner("Modifying…"):
                    try:
                        modified = system.modify_floor_plan(selected_id, command)
                        st.session_state["modified_plan"] = modified
                    except Exception as e:
                        st.error(f"Modification failed: {e}")

            if "modified_plan" in st.session_state:
                with col_after:
                    st.markdown("**After**")
                    st.plotly_chart(
                        visualise_floor_plan(st.session_state["modified_plan"]),
                        use_container_width=True, key="after_viz"
                    )
                with st.expander("Modified plan JSON"):
                    st.json(st.session_state["modified_plan"])

# ── Browse ────────────────────────────────────────────────────────────────────
with tab_browse:
    st.subheader("Browse All Plans")

    all_plans = system.list_floor_plans()
    if not all_plans:
        st.info("No plans stored yet.")
    else:
        for p in all_plans:
            full = system.get_floor_plan(p["id"])
            if not full:
                continue
            with st.expander(
                f"**{p['name']}** — "
                f"{p['bedroom_count']} BR · {p['bathroom_count']} BA · {p['total_area']:.0f} m²",
                expanded=False,
            ):
                st.plotly_chart(
                    visualise_floor_plan(full),
                    use_container_width=True,
                    key=f"browse_{p['id']}",
                )
                show_plan_metrics(full)
