"""Interactive simulation only. No live sensors or navigation instruction."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from li_stick_next.engine import SafetyEngine,Sample,Detection
st.set_page_config(page_title="Li-Stick | Hanan Aref",page_icon="🦯",layout="wide")
st.title("Li-Stick · Assistive Perception Lab")
st.caption("SIMULATED INPUTS. Not for unsupervised real-world navigation.")
with st.sidebar:
    distance=st.slider("Ultrasonic distance (cm)",0,350,80)
    speed=st.slider("Speed (m/s)",0.0,2.5,.6,.1)
    label=st.selectbox("Object class",["none","pole","stairs","curb","car","person","overhead_obstacle","pothole"])
    direction=st.selectbox("Position in image",["left","ahead","right"])
    camera=st.checkbox("Camera healthy",True)
    ultrasonic=st.checkbox("Ultrasonic healthy",True)
    fall=st.selectbox("Fall-event demo",["normal","suspected","impact","confirmed"])
if st.button("Analyze simulated scenario",type="primary"):
    e=SafetyEngine()
    zones={"left":(.05,.25),"ahead":(.40,.60),"right":(.75,.95)}
    x1,x2=zones[direction]
    boxes=() if label=="none" else (Detection(label,.83,x1,.25,x2,.70),)
    if fall!="normal":
        e.process(Sample(1,imu_valid=True,vertical_accel_mps2=-6))
        if fall in ("impact","confirmed"):e.process(Sample(2,imu_valid=True,impact_g=3.2))
    alerts=e.process(Sample(3,detections=boxes,ultrasonic_cm=float(distance),speed_mps=speed,
        camera_valid=camera,ultrasonic_valid=ultrasonic,imu_valid=True,
        orientation_change_deg=65 if fall=="confirmed" else 0,
        stillness_seconds=2.4 if fall=="confirmed" else 0))
    st.session_state["alerts"]=alerts
    st.session_state["fall_phase"]=e.phase.value
a,b,c=st.columns(3)
a.metric("Distance (simulated)",f"{distance} cm")
b.metric("Walking speed (simulated)",f"{speed:.1f} m/s")
c.metric("Fall phase",st.session_state.get("fall_phase","normal"))
st.subheader("Alert priority")
for alert in st.session_state.get("alerts",[]):
    fn=st.error if alert.severity.value=="critical" else st.warning if alert.severity.value=="warning" else st.info
    fn(f"{alert.severity.value.upper()} | {alert.message} | {alert.direction}")
if not st.session_state.get("alerts"):
    st.info("No alert in this simulation. This does NOT prove the path is clear.")
