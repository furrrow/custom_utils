"""Offline regression checks for the ROS-clock controller without a wall tick watchdog."""
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT = Path(__file__).resolve().parents[1]

def load_definitions(path, names, namespace):
    """Execute real pure definitions without importing ROS, CUDA or model code."""
    import ast
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(selected) == len(names)
    source = 'from __future__ import annotations\n' + '\n'.join((ast.unparse(node) for node in selected))
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace

def test_controller_clock_jump_clears_motion_on_next_controller_callback():
    ns = load_definitions(ROOT / 'custom_utils/kinematic_controller_naive.py', {'PDControllerNode'}, {'Node': object})
    node = ns['PDControllerNode'].__new__(ns['PDControllerNode'])
    (cleared, stopped) = ([], [])
    node.waypoint = SimpleNamespace(clear=lambda : cleared.append(True))
    node.publish_zero = lambda **kw: stopped.append(True)
    node.simulation_episode_control = True
    node.ready_for_motion = True
    node._clock_reset_pending = False
    node._on_clock_jump(None)
    assert not stopped
    node._apply_clock_reset()
    assert not node.ready_for_motion
    assert cleared == [True]
    assert stopped == [True]

@pytest.mark.parametrize('trace_enabled', [False, True])
def test_slow_behavior_ticks_do_not_inject_controller_watchdog_zeros(trace_enabled):
    import json
    import os
    import numpy as np
    wall = SimpleNamespace(now=100.0)

    def twist():
        return SimpleNamespace(linear=SimpleNamespace(x=0.0), angular=SimpleNamespace(z=0.0))
    ns = load_definitions(ROOT / 'custom_utils/kinematic_controller_naive.py', {'PDControllerNode', 'pd_controller', 'clip_angle'}, dict(Node=object, np=np, json=json, os=os, Twist=twist, time=SimpleNamespace(monotonic=lambda : wall.now)))
    node = ns['PDControllerNode'].__new__(ns['PDControllerNode'])
    (output, logs) = ([], [])
    node._command_trace_enabled = trace_enabled
    node._command_trace_sequence = 0
    node._clock_reset_pending = False
    node.get_clock = lambda : SimpleNamespace(ros_time_is_active=True, now=lambda : SimpleNamespace(nanoseconds=1000000000))
    node.get_logger = lambda : SimpleNamespace(info=logs.append, error=logs.append)
    node.current_episode_id = 'test'
    node.waypoint = SimpleNamespace(is_valid=lambda **kw: True, get=lambda : np.array([0.3, 0.0]), last_time_received=0.9, timeout=5.0)
    node.simulation_episode_control = True
    node.ready_for_motion = True
    node.reached_goal = False
    (node.max_v, node.max_w, node.dt) = (1.5, 1.2, 0.75)
    node.vel_msg = twist()
    node.vel_out = SimpleNamespace(publish=lambda msg: output.append((msg.linear.x, msg.angular.z)))
    node.timer_callback()
    wall.now = 100.8
    node.timer_callback()
    assert output[0] == pytest.approx((0.4, 0.0))
    assert output[1] == pytest.approx((0.4, 0.0))
    assert not hasattr(node, '_check_control_liveness')
    traces = [json.loads(line.split('COMMAND_TRACE ', 1)[1]) for line in logs if line.startswith('COMMAND_TRACE ')]
    if trace_enabled:
        assert [r['reason'] for r in traces] == ['control_tick_valid_waypoint', 'control_tick_valid_waypoint']
        assert 'control_tick_wall_age_s' not in traces[-1]
        assert traces[-1]['linear_x'] == pytest.approx(0.4)
        assert traces[-1]['waypoint_receipt_ros_age_s'] == pytest.approx(0.1)
    else:
        assert traces == []

def test_controller_constructor_has_only_the_behavior_clock_timer(tmp_path):
    import numpy as np
    import yaml
    path = ROOT / 'custom_utils/kinematic_controller_naive.py'
    ros_clock = SimpleNamespace(now=lambda : SimpleNamespace(nanoseconds=1000000000), create_jump_callback=lambda *a, **k: object())
    timers = []

    class FakeNode:

        def __init__(self, name):
            pass

        def get_clock(self):
            return ros_clock

        def create_timer(self, period, callback, clock):
            timers.append((period, callback, clock))
            return object()

        def create_subscription(self, *a, **k):
            return object()

        def create_publisher(self, *a, **k):
            return SimpleNamespace(publish=lambda msg: None)

        def get_logger(self):
            return SimpleNamespace(info=lambda msg: None)
    ns = load_definitions(path, {'ROSData', 'PDControllerNode'}, dict(Node=FakeNode, np=np, yaml=yaml, os=__import__('os'), time=SimpleNamespace(monotonic=lambda : 100.0), Twist=lambda : SimpleNamespace(), Float32MultiArray=object, Bool=object, String=object, QoSProfile=lambda **kw: kw, QoSReliabilityPolicy=SimpleNamespace(RELIABLE=1), QoSHistoryPolicy=SimpleNamespace(KEEP_LAST=1), JumpThreshold=lambda **kw: kw, Duration=lambda **kw: kw))
    config = tmp_path / 'robot.yaml'
    config.write_text(__import__('json').dumps(dict(controller_rate=10, sim_nova=dict(max_v=1.5,max_w=1.2,image_topic='/image',waypoint_topic='/waypoint',steered_waypoint_topic='/steered',reached_goal_topic='/goal',vel_topic='/cmd_vel'))))
    node = ns['PDControllerNode'](str(config), 'sim_nova', default_dt=0.75)
    assert timers[0][0] == 0.1
    assert timers[0][2] is ros_clock
    assert len(timers) == 1
    assert node.dt == 0.75
    node.waypoint.set([1.0, 0.0])
    assert node.waypoint.last_time_received == 1.0
