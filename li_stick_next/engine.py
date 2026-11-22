"""Safety policy prototype; candidate thresholds, not certified navigation logic."""
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

class Severity(str, Enum):
    INFO="info"
    CAUTION="caution"
    WARNING="warning"
    CRITICAL="critical"

class FallPhase(str, Enum):
    NORMAL="normal"
    SUSPECTED="suspected"
    IMPACT="impact"
    CONFIRMED="confirmed"

@dataclass(frozen=True)
class Detection:
    label:str
    confidence:float
    x1:float
    y1:float
    x2:float
    y2:float

@dataclass(frozen=True)
class Sample:
    timestamp:float
    ultrasonic_cm:float|None=None
    detections:tuple[Detection,...]=()
    speed_mps:float|None=None
    vertical_accel_mps2:float|None=None
    impact_g:float|None=None
    orientation_change_deg:float|None=None
    stillness_seconds:float|None=None
    imu_valid:bool=False
    camera_valid:bool=True
    ultrasonic_valid:bool=True

@dataclass(frozen=True)
class Alert:
    severity:Severity
    message:str
    direction:str
    reason:str

@dataclass
class SafetyEngine:
    last_timestamp:float|None=None
    phase:FallPhase=FallPhase.NORMAL
    phase_started:float=0
    last_alert:dict[str,float]=field(default_factory=dict)
    cooldown_seconds:float=1.5
    fall_timeout_s:float=3
    impact_limit_g:float=2.7
    fall_orientation_deg:float=50
    min_stillness_s:float=2
    ultrasonic_min_cm:float=20
    ultrasonic_max_cm:float=350
    freshness_gap_s:float=1

    def _emit(self,a:Alert,t:float,out:list[Alert],bypass:bool=False):
        key=a.reason+":"+a.direction
        if bypass or t-self.last_alert.get(key,float("-inf"))>=self.cooldown_seconds:
            out.append(a)
            self.last_alert[key]=t

    def _fall(self,s:Sample,out:list[Alert]):
        t=s.timestamp
        if not s.imu_valid:
            if self.phase!=FallPhase.NORMAL:
                self._emit(Alert(Severity.WARNING,"IMU unavailable","unknown","fall_imu_lost"),t,out)
            self.phase=FallPhase.NORMAL
            return
        impact=s.impact_g is not None and isfinite(s.impact_g) and s.impact_g>=self.impact_limit_g
        rotation=s.orientation_change_deg is not None and isfinite(s.orientation_change_deg) and s.orientation_change_deg>=self.fall_orientation_deg
        descent=s.vertical_accel_mps2 is not None and isfinite(s.vertical_accel_mps2) and s.vertical_accel_mps2< -4
        still=s.stillness_seconds is not None and isfinite(s.stillness_seconds) and s.stillness_seconds>=self.min_stillness_s
        if self.phase not in (FallPhase.NORMAL,FallPhase.CONFIRMED) and t-self.phase_started>self.fall_timeout_s:
            self.phase=FallPhase.NORMAL
        if self.phase==FallPhase.NORMAL and (descent or rotation):
            self.phase,self.phase_started=FallPhase.SUSPECTED,t
            self._emit(Alert(Severity.CAUTION,"Possible fall movement","unknown","fall_suspected"),t,out)
        if self.phase==FallPhase.SUSPECTED and impact:
            self.phase,self.phase_started=FallPhase.IMPACT,t
            self._emit(Alert(Severity.WARNING,"Possible impact; checking movement","unknown","fall_impact"),t,out)
        if self.phase==FallPhase.IMPACT and still and rotation:
            self.phase,self.phase_started=FallPhase.CONFIRMED,t
            self._emit(Alert(Severity.CRITICAL,"Possible fall: check user","unknown","fall_confirmed"),t,out,True)

    def process(self,s:Sample)->list[Alert]:
        t=s.timestamp
        if not isfinite(t) or self.last_timestamp is not None and t<=self.last_timestamp:
            raise ValueError("Strictly increasing finite timestamps required")
        out=[]
        if self.last_timestamp is not None and t-self.last_timestamp>self.freshness_gap_s:
            self._emit(Alert(Severity.WARNING,"Sensor stream delayed","unknown","stale_stream"),t,out)
        self.last_timestamp=t
        if not s.camera_valid:
            self._emit(Alert(Severity.WARNING,"Camera offline","unknown","camera_fault"),t,out)
        if not s.ultrasonic_valid:
            self._emit(Alert(Severity.WARNING,"Ultrasonic offline","unknown","sonar_fault"),t,out)
        self._fall(s,out)
        speed=s.speed_mps if s.speed_mps is not None and isfinite(s.speed_mps) and 0<=s.speed_mps<=4 else None
        if s.speed_mps is not None and speed is None:
            self._emit(Alert(Severity.WARNING,"Invalid speed reading","unknown","speed_fault"),t,out)
        warning=100+160*(speed or 0)
        critical=45+65*(speed or 0)
        d=s.ultrasonic_cm
        if s.ultrasonic_valid:
            if d is None or not isfinite(d) or not 0<=d<=self.ultrasonic_max_cm:
                self._emit(Alert(Severity.WARNING,"Invalid ultrasonic distance","unknown","sonar_invalid"),t,out)
            elif d<self.ultrasonic_min_cm:
                self._emit(Alert(Severity.CRITICAL,"Very close or blind-zone obstacle: stop","ahead","blind_zone"),t,out)
            elif d<=critical:
                self._emit(Alert(Severity.CRITICAL,f"Obstacle approx {d:.0f} cm ahead","ahead","sonar_critical"),t,out)
            elif d<=warning:
                self._emit(Alert(Severity.WARNING,f"Obstacle approx {d:.0f} cm ahead","ahead","sonar_warning"),t,out)
        if s.camera_valid:
            labels={"stairs","curb","pole","bollard","barrier","tree_trunk","person","bicycle","car","motorcycle","bus","truck","chair","pothole","drop_off","overhead_obstacle"}
            for d in s.detections:
                vals=(d.confidence,d.x1,d.y1,d.x2,d.y2)
                if not all(isfinite(v) for v in vals) or not (0<=d.confidence<=1 and 0<=d.x1<d.x2<=1 and 0<=d.y1<d.y2<=1):
                    continue
                if d.label not in labels or d.confidence<.35:continue
                cx=(d.x1+d.x2)/2
                direction="left" if cx<.36 else "right" if cx>.64 else "ahead"
                severity=Severity.WARNING if d.label in {"stairs","curb","pothole","drop_off","overhead_obstacle"} else Severity.CAUTION
                self._emit(Alert(severity,f"{d.label} detected {direction}; verify path",direction,"vision_"+d.label),t,out)
        if not s.camera_valid and not s.ultrasonic_valid:
            self._emit(Alert(Severity.CRITICAL,"Both sensors offline; stop and use primary aid","unknown","all_sensors_offline"),t,out)
        return sorted(out,key=lambda a:list(Severity).index(a.severity),reverse=True)
