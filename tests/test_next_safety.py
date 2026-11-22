import unittest
from li_stick_next.engine import SafetyEngine, Sample, Detection, Severity, FallPhase
from li_stick_next.transport import parse_telemetry
from li_stick_next.adapters import convert_upstream_detections
class SafetyTests(unittest.TestCase):
    def test_blind_zone(self):
        alerts=SafetyEngine().process(Sample(1,ultrasonic_cm=12))
        self.assertTrue(any(a.severity==Severity.CRITICAL for a in alerts))
    def test_speed_changes_warning_range(self):
        s=Sample(1,ultrasonic_cm=170,speed_mps=1.5)
        self.assertTrue(any(a.reason=="sonar_warning" for a in SafetyEngine().process(s)))
    def test_left_direction(self):
        a=SafetyEngine().process(Sample(1,detections=(Detection("pole",.8,.05,.2,.20,.4),),ultrasonic_valid=False))
        self.assertTrue(any(x.direction=="left" for x in a))
    def test_invalid_time(self):
        e=SafetyEngine()
        e.process(Sample(1))
        with self.assertRaises(ValueError):e.process(Sample(1))
    def test_invalid_json(self):
        with self.assertRaises(ValueError):parse_telemetry(b'{"type":"hello"}')
    def test_serial(self):
        x=parse_telemetry(b'{"type":"telemetry","seq":1,"ultrasonic_cm":45,"ultrasonic_valid":true,"imu_valid":false}')
        self.assertEqual(x.ultrasonic_cm,45)
    def test_invalid_box(self):
        class X:
            class_name="pole";confidence=.9;x1=10;x2=2;y1=3;y2=20
        self.assertEqual(convert_upstream_detections([X()],100,100),())
if __name__=="__main__":unittest.main()
