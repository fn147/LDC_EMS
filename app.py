import streamlit as st
import plotly.graph_objects as go
import json, math, requests, hashlib
from collections import defaultdict
from datetime import datetime

# --- GitHub Gist persistence ---
_GIST_ID    = st.secrets["GIST_ID"]
_GH_TOKEN   = st.secrets["GH_TOKEN"]
_GIST_URL   = f"https://api.github.com/gists/{_GIST_ID}"
_GH_HEADERS = {
    "Authorization": f"token {_GH_TOKEN}",
    "Accept": "application/vnd.github+json",
}
_TASKS_FILENAME = "tasks.json"
_DONE_FILENAME  = "done_tasks.json"
_USERS_FILENAME = "users.json"

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

NOW = lambda: datetime.now().strftime("%Y-%m-%d %H:%M")


def _gist_read_all(filename: str) -> dict:
    resp = requests.get(_GIST_URL, headers=_GH_HEADERS, timeout=10)
    resp.raise_for_status()
    raw = resp.json()["files"].get(filename, {}).get("content", "{}")
    return json.loads(raw)


def _gist_write_all(filename: str, data: dict) -> None:
    payload = {"files": {filename: {"content": json.dumps(data, indent=2)}}}
    resp = requests.patch(_GIST_URL, headers=_GH_HEADERS,
                          json=payload, timeout=10)
    resp.raise_for_status()


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    return _gist_read_all(_USERS_FILENAME)


def register_user(username: str, password: str) -> bool:
    users = load_users()
    if username in users:
        return False
    users[username] = _hash(password)
    _gist_write_all(_USERS_FILENAME, users)
    return True


def check_password(username: str, password: str) -> bool:
    users = load_users()
    return users.get(username) == _hash(password)


def load_tasks(user_email: str) -> list:
    return _gist_read_all(_TASKS_FILENAME).get(user_email, [])


def save_tasks(user_email: str, tasks: list) -> None:
    all_data = _gist_read_all(_TASKS_FILENAME)
    all_data[user_email] = tasks
    _gist_write_all(_TASKS_FILENAME, all_data)


def load_done(user_email: str) -> list:
    return _gist_read_all(_DONE_FILENAME).get(user_email, [])


def save_done(user_email: str, done: list) -> None:
    all_data = _gist_read_all(_DONE_FILENAME)
    all_data[user_email] = done
    _gist_write_all(_DONE_FILENAME, all_data)


def migrate(tasks):
    for t in tasks:
        t.pop("description", None)
        t.setdefault("created_at", NOW())
        # Normalise subtodos: old string entries → {text, done}
        raw = t.get("subtodos", [])
        normalised = []
        for s in raw:
            if isinstance(s, str):
                normalised.append({"text": s, "done": False})
            elif isinstance(s, dict):
                s.setdefault("done", False)
                normalised.append(s)
        t["subtodos"] = normalised
    return tasks


def display_positions(tasks):
    groups = defaultdict(list)
    for i, t in enumerate(tasks):
        groups[(t["urgency"], t["importance"])].append(i)
    pos = {}
    for (urg, imp), indices in groups.items():
        if len(indices) == 1:
            pos[indices[0]] = (urg, imp)
        else:
            r = 0.28
            for k, idx in enumerate(indices):
                angle = 2 * math.pi * k / len(indices) - math.pi / 2
                pos[idx] = (round(urg + r * math.cos(angle), 3),
                            round(imp + r * math.sin(angle), 3))
    return pos


st.set_page_config(page_title="Task Priority Matrix", layout="wide")

# --- Auth ---
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if not st.session_state.logged_in_user:
    st.title("Task Priority Matrix")
    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log in"):
            if check_password(username.strip(), password):
                st.session_state.logged_in_user = username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with tab_register:
        new_user = st.text_input("Choose a username", key="reg_user")
        new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        new_pass2 = st.text_input("Confirm password", type="password", key="reg_pass2")
        if st.button("Create account"):
            if not new_user.strip():
                st.error("Username cannot be empty.")
            elif new_pass != new_pass2:
                st.error("Passwords do not match.")
            elif len(new_pass) < 4:
                st.error("Password must be at least 4 characters.")
            elif register_user(new_user.strip(), new_pass):
                st.success("Account created! Switch to the Log in tab.")
            else:
                st.error("Username already taken.")
    st.stop()

USER_EMAIL = st.session_state.logged_in_user

st.title("Task Priority Matrix")
st.caption("Click a dot to edit, move, complete, or delete it.")

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

tasks = migrate(load_tasks(USER_EMAIL))
done  = load_done(USER_EMAIL)

# --- Sidebar: add tasks ---
with st.sidebar:
    st.caption(f"Logged in as **{USER_EMAIL}**")
    if st.button("Log out"):
        st.session_state.logged_in_user = None
        st.rerun()
    st.divider()

    st.header("Add Task")
    task_name  = st.text_input("Task or Project Name")
    importance = st.slider("Importance", 1, 10, 5)
    urgency    = st.slider("Urgency",    1, 10, 5)

    if st.button("Add Task", use_container_width=True):
        if task_name.strip():
            tasks.append({
                "name":       task_name.strip(),
                "importance": importance,
                "urgency":    urgency,
                "subtodos":   [],
                "created_at": NOW(),
            })
            save_tasks(USER_EMAIL, tasks)
            st.success(f'Added "{task_name.strip()}"')
            st.rerun()
        else:
            st.warning("Please enter a task name.")

    if done:
        st.divider()
        st.markdown(f"**Completed tasks: {len(done)}**")


# --- Build plotly figure ---
def build_fig(tasks, selected_idx):
    fig = go.Figure()

    for x0, y0, x1, y1, col in [
        (5.5, 5.5, 10.5, 10.5, "red"),
        (0.5, 5.5, 5.5,  10.5, "orange"),
        (5.5, 0.5, 10.5, 5.5,  "steelblue"),
        (0.5, 0.5, 5.5,  5.5,  "green"),
    ]:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=col, opacity=0.08, line_width=0)

    fig.add_shape(type="line", x0=5.5, x1=5.5, y0=0.5, y1=10.5,
                  line=dict(color="#555", dash="dash", width=1))
    fig.add_shape(type="line", x0=0.5, x1=10.5, y0=5.5, y1=5.5,
                  line=dict(color="#555", dash="dash", width=1))

    lkw = dict(showarrow=False, opacity=0.55)
    fig.add_annotation(x=8, y=9.2, text="Do First<br>(urgent & important)",         font=dict(size=11, color="red"),       **lkw)
    fig.add_annotation(x=3, y=9.2, text="Schedule<br>(important, not urgent)",      font=dict(size=11, color="orange"),    **lkw)
    fig.add_annotation(x=8, y=1.8, text="Delegate<br>(urgent, not important)",      font=dict(size=11, color="steelblue"), **lkw)
    fig.add_annotation(x=3, y=1.8, text="Eliminate<br>(not urgent, not important)", font=dict(size=11, color="green"),     **lkw)

    pos = display_positions(tasks)
    for i, task in enumerate(tasks):
        color  = COLORS[i % len(COLORS)]
        sel    = (i == selected_idx)
        dx, dy = pos[i]
        subs   = task.get("subtodos", [])
        n_done = sum(1 for s in subs if s["done"])
        sub_line = f"Subtasks: {n_done}/{len(subs)}" if subs else "No subtasks"
        fig.add_trace(go.Scatter(
            x=[dx], y=[dy],
            mode="markers",
            marker=dict(size=22 if sel else 17, color=color,
                        line=dict(color="white" if sel else color, width=3 if sel else 1.5)),
            customdata=[i],
            name=task["name"],
            hovertemplate=(
                f"<b>{task['name']}</b><br>"
                f"Importance: {task['importance']}  |  Urgency: {task['urgency']}<br>"
                f"{sub_line}<br>"
                f"Added: {task.get('created_at','—')}<extra></extra>"
            ),
        ))
        fig.add_annotation(
            x=dx, y=dy,
            text=task["name"],
            showarrow=False,
            xshift=14, yshift=10,
            font=dict(size=10, color=color),
            xanchor="left",
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(title="Urgency  (how fast to do it →)", range=[0.5, 10.5],
                   tickvals=list(range(1, 11)), gridcolor="#333"),
        yaxis=dict(title="Importance  (how critical it is ↑)", range=[0.5, 10.5],
                   tickvals=list(range(1, 11)), gridcolor="#333"),
        height=580, showlegend=False,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        margin=dict(l=60, r=20, t=20, b=60),
        clickmode="event+select",
    )
    return fig


fig   = build_fig(tasks, st.session_state.selected_idx)
event = st.plotly_chart(fig, on_select="rerun", use_container_width=True)

# Detect click
if event and event.selection and event.selection.points:
    pt  = event.selection.points[0]
    raw = pt.get("customdata")
    clicked = int(raw[0]) if isinstance(raw, list) else int(raw)
    if clicked != st.session_state.selected_idx:
        st.session_state.selected_idx = clicked
        st.rerun()

selected_idx = st.session_state.selected_idx

# --- Edit panel ---
if selected_idx is not None and selected_idx < len(tasks):
    task = tasks[selected_idx]

    st.divider()

    col_title, col_done, col_del = st.columns([0.75, 0.13, 0.12])
    with col_title:
        st.markdown(f"### {task['name']}")
        st.caption(f"Added: {task.get('created_at', '—')}")
    with col_done:
        if st.button("✔ Done", key="btn_done", use_container_width=True):
            task["completed_at"] = NOW()
            done.append(task)
            save_done(USER_EMAIL, done)
            tasks.pop(selected_idx)
            save_tasks(USER_EMAIL, tasks)
            st.session_state.selected_idx = None
            st.rerun()
    with col_del:
        if st.button("🗑 Delete", key="btn_delete", type="primary", use_container_width=True):
            tasks.pop(selected_idx)
            save_tasks(USER_EMAIL, tasks)
            st.session_state.selected_idx = None
            st.rerun()

    st.markdown("**Move task**")
    col_u, col_i, col_btn = st.columns([2, 2, 1])
    with col_u:
        new_urg = st.number_input("Urgency", min_value=1, max_value=10,
                                  value=task["urgency"], step=1, key=f"ni_urg_{selected_idx}")
    with col_i:
        new_imp = st.number_input("Importance", min_value=1, max_value=10,
                                  value=task["importance"], step=1, key=f"ni_imp_{selected_idx}")
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Move", key="btn_move", use_container_width=True):
            task["urgency"]    = int(new_urg)
            task["importance"] = int(new_imp)
            save_tasks(USER_EMAIL, tasks)
            st.rerun()

    # --- Subtasks ---
    subs   = task["subtodos"]
    n_done = sum(1 for s in subs if s["done"])
    st.markdown(f"**Subtasks** &nbsp; <small>{n_done}/{len(subs)} done</small>" if subs
                else "**Subtasks**", unsafe_allow_html=True)

    for si, sub in enumerate(subs):
        col_chk, col_txt, col_x = st.columns([0.06, 0.82, 0.12])
        with col_chk:
            checked = st.checkbox("", value=sub["done"],
                                  key=f"sub_chk_{selected_idx}_{si}",
                                  label_visibility="collapsed")
            if checked != sub["done"]:
                task["subtodos"][si]["done"] = checked
                save_tasks(USER_EMAIL, tasks)
                st.rerun()
        with col_txt:
            if sub["done"]:
                st.markdown(f"~~{sub['text']}~~")
            else:
                st.markdown(sub["text"])
        with col_x:
            if st.button("✕", key=f"del_sub_{selected_idx}_{si}", help="Remove subtask"):
                task["subtodos"].pop(si)
                save_tasks(USER_EMAIL, tasks)
                st.rerun()

    col_new, col_add = st.columns([0.82, 0.18])
    with col_new:
        new_sub = st.text_input("new subtask", placeholder="Add a subtask…",
                                key=f"new_sub_{selected_idx}",
                                label_visibility="collapsed")
    with col_add:
        if st.button("+ Add", key=f"add_sub_{selected_idx}", use_container_width=True):
            if new_sub.strip():
                task["subtodos"].append({"text": new_sub.strip(), "done": False})
                save_tasks(USER_EMAIL, tasks)
                st.rerun()

elif not tasks:
    st.info("No tasks yet. Add one using the sidebar.")

# --- Completed tasks ---
if done:
    st.divider()
    with st.expander(f"Completed Tasks ({len(done)})", expanded=False):
        for j, d in enumerate(reversed(done)):
            real_j = len(done) - 1 - j
            col_info, col_btn = st.columns([0.88, 0.12])
            with col_info:
                subs   = d.get("subtodos", [])
                n_done = sum(1 for s in subs if s.get("done"))
                sub_summary = f" &nbsp;·&nbsp; {n_done}/{len(subs)} subtasks done" if subs else ""
                st.markdown(
                    f"**{d['name']}** &nbsp;·&nbsp; "
                    f"Importance {d['importance']} / Urgency {d['urgency']}{sub_summary}<br>"
                    f"<small>Added: {d.get('created_at','—')} &nbsp;|&nbsp; "
                    f"Done: {d.get('completed_at','—')}</small>",
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("🗑", key=f"del_done_{real_j}", help="Delete this record"):
                    done.pop(real_j)
                    save_done(USER_EMAIL, done)
                    st.rerun()
            st.markdown("---")
