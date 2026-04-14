import streamlit as st
import time
import pandas as pd
import pydeck as pdk
import sys
import os
from stable_baselines3 import PPO

# Force Python to recognize the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.rl_training.environment import UrbanTrafficEnv

st.set_page_config(page_title="Urban Pulse Command Center", page_icon="🚦", layout="wide")
st.title("🚦 Urban Pulse: District-Level Traffic Orchestration")
st.markdown("Watch the PPO Agent manage a massive 20-intersection (4x5) grid in Central London.")

# --- PROGRAMMATIC GEOMETRY (4x5 GRID) ---
# Mathematically mapping 20 nodes across Soho / Covent Garden
NODE_COORDS = []
start_lat, start_lon = 51.516, -0.140
for row in range(4):
    for col in range(5):
        NODE_COORDS.append({
            "lat": start_lat - (row * 0.003), # Move South
            "lon": start_lon + (col * 0.005)  # Move East
        })

@st.cache_resource
def load_ai_system():
    env = UrbanTrafficEnv(num_intersections=20)
    # Load the new massive 20-node brain
    model = PPO.load("models/ppo_traffic/ppo_agent_20_nodes", device="cpu")
    return env, model

env, model = load_ai_system()

if "running" not in st.session_state:
    st.session_state.running = False

# --- UI LAYOUT ---
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("▶️ Start District Simulation", use_container_width=True):
        st.session_state.running = True
with col2:
    status_text = st.empty()

map_col, chart_col = st.columns([2, 1])
with map_col:
    map_placeholder = st.empty()
with chart_col:
    chart_placeholder = st.empty()
    
# Tucking the raw data away so the UI stays clean
raw_data_expander = st.expander("View Raw Queue Data (All 20 Nodes)")
metrics_container = raw_data_expander.empty()

# --- THE LIVE INFERENCE LOOP ---
if st.session_state.running:
    obs, _ = env.reset()
    reward_history = []
    
    status_text.info("Simulation Running... AI is balancing 64 independent traffic lanes.")
    
    for step in range(1, 51):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_history.append(reward)
        
        # 1. UPDATE REWARD CHART
        chart_data = pd.DataFrame(reward_history, columns=["District Reward"])
        chart_placeholder.line_chart(chart_data, height=400)
        
        # 2. TRANSLATION LAYER (20 NODES)
        map_data = []
        raw_text = ""
        
        for i in range(20):
            idx = i * 4
            n, s, e, w = obs[idx:idx+4]
            total_queue = n + s + e + w
            
            color = [0, 255, 255, 200] if action[i] == 0 else [255, 0, 255, 200]
            
            map_data.append({
                "lat": NODE_COORDS[i]["lat"],
                "lon": NODE_COORDS[i]["lon"],
                "color": color,
                "radius": max(10, total_queue * 3) # Scaled down slightly for dense map
            })
            
            light_str = "N/S Green" if action[i] == 0 else "E/W Green"
            raw_text += f"**Node {i+1:02d}** ({light_str}) | Queues -> N:{int(n):02d} S:{int(s):02d} E:{int(e):02d} W:{int(w):02d}\n\n"
            
        metrics_container.markdown(raw_text)
        df_map = pd.DataFrame(map_data)
        
        # 3. RENDER THE PYDECK MAP
        # Pulled the camera back to view the whole 20-node district
        view_state = pdk.ViewState(latitude=51.511, longitude=-0.130, zoom=14.0, pitch=45)
        
        heat_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[lon, lat]',
            get_radius='radius * 3', 
            get_fill_color='[255, 50, 50, 100]',
            pickable=False
        )
        
        node_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[lon, lat]',
            get_radius=20,
            get_fill_color='color',
            pickable=False
        )
        
        r = pdk.Deck(layers=[heat_layer, node_layer], initial_view_state=view_state, map_style='dark')
        map_placeholder.pydeck_chart(r)
        
        time.sleep(1.0) 
        
        if terminated or truncated:
            status_text.error("🚨 Gridlock Detected! Simulation Terminated.")
            break
            
    st.session_state.running = False
    status_text.success(f"🏁 Simulation Complete! Total District Reward: {sum(reward_history):.2f}")