"""Five demo poses from SMPL24 geometry; calibrate thresholds on real camera data."""
import json
from pathlib import Path
import numpy as np

DEFAULTS = {
    'confirm_predictions': 3,
    'arm_raise_deg': 55.0, 'arm_rest_deg': 40.0,
    'raised_wrist_min_torso': -0.40, 'rest_wrist_max_torso': -0.30,
    'standing_knee_min_deg': 135.0, 'standing_extension_min': 0.85,
    'squat_knee_max_deg': 135.0, 'squat_extension_max': 0.82,
}

def angle(a, b):
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale < 1e-8:
        raise ValueError('degenerate limb')
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / scale, -1., 1.))))

class PoseRecognizer:
    def __init__(self, config_path=None):
        self.config = dict(DEFAULTS)
        if config_path is not None:
            custom = json.loads(Path(config_path).read_text())
            if set(custom) - set(DEFAULTS):
                raise ValueError('unknown pose threshold names')
            self.config.update(custom)
        c = self.config
        if (int(c['confirm_predictions']) != c['confirm_predictions']
                or c['confirm_predictions'] < 1
                or not 0 <= c['arm_rest_deg'] < c['arm_raise_deg'] <= 180
                or not 0 < c['squat_knee_max_deg'] <= c['standing_knee_min_deg'] <= 180
                or not 0 < c['squat_extension_max'] < c['standing_extension_min'] <= 1
                or not all(np.isfinite(v) for v in c.values())):
            raise ValueError('invalid pose thresholds')
        self.reset()

    def reset(self):
        self.candidate = None
        self.count = 0
        self.label = 'UNKNOWN'

    def update(self, joints, body_at_frame_edge=False, body_severely_clipped=False):
        features, reason, raw = {}, 'INVALID_JOINTS', 'UNKNOWN'
        j = np.asarray(joints, dtype=float)
        try:
            if j.shape != (24, 3) or not np.isfinite(j).all():
                raise ValueError('invalid joints')
            up = j[12] - j[0]
            torso = float(np.linalg.norm(up))
            if not .10 < torso < 1.2:
                raise ValueError('invalid torso')
            up /= torso
            for side, shoulder, elbow, wrist, hip, knee, ankle in (
                    ('left', 16, 18, 20, 1, 4, 7),
                    ('right', 17, 19, 21, 2, 5, 8)):
                features[side+'_arm_lift_deg'] = angle(j[elbow]-j[shoulder], -up)
                features[side+'_wrist_torso'] = float(np.dot(j[wrist]-j[shoulder], up)/torso)
                features[side+'_elbow_deg'] = angle(j[shoulder]-j[elbow], j[wrist]-j[elbow])
                features[side+'_knee_deg'] = angle(j[hip]-j[knee], j[ankle]-j[knee])
            leg_length = np.mean([np.linalg.norm(j[h]-j[k])+np.linalg.norm(j[k]-j[a])
                                  for h,k,a in ((1,4,7),(2,5,8))])
            if leg_length < .2:
                raise ValueError('invalid legs')
            features['leg_extension_ratio'] = float(np.linalg.norm((j[1]+j[2]-j[7]-j[8])*.5)/leg_length)
            c = self.config
            lk, rk = features['left_knee_deg'], features['right_knee_deg']
            extension = features['leg_extension_ratio']
            squat = max(lk, rk) <= c['squat_knee_max_deg'] and extension <= c['squat_extension_max']
            standing = min(lk, rk) >= c['standing_knee_min_deg'] and extension >= c['standing_extension_min']
            raised = {s: features[s+'_arm_lift_deg'] >= c['arm_raise_deg']
                         and features[s+'_wrist_torso'] >= c['raised_wrist_min_torso']
                      for s in ('left', 'right')}
            resting = {s: features[s+'_arm_lift_deg'] <= c['arm_rest_deg']
                          and features[s+'_wrist_torso'] <= c['rest_wrist_max_torso']
                       for s in ('left', 'right')}
            if body_severely_clipped:
                reason = 'STEP_BACK_FULL_BODY'
            elif squat:
                raw, reason = 'SQUAT', 'BOTH_KNEES_BENT'
            elif raised['left'] and raised['right']:
                raw, reason = 'BOTH_ARMS_UP', 'BOTH_ARMS_LIFT'
            elif raised['left'] and resting['right']:
                raw, reason = 'LEFT_ARM_UP', 'LEFT_ARM_LIFT'
            elif raised['right'] and resting['left']:
                raw, reason = 'RIGHT_ARM_UP', 'RIGHT_ARM_LIFT'
            elif standing and resting['left'] and resting['right']:
                raw, reason = 'STANDING', 'LEGS_STRAIGHT_ARMS_DOWN'
            elif not standing:
                reason = 'LEGS_TRANSITION'
            else:
                reason = 'ARMS_TRANSITION'
        except ValueError:
            raw, reason = 'UNKNOWN', 'INVALID_JOINTS'
        if raw == 'UNKNOWN':
            self.reset()
        else:
            self.count = self.count+1 if raw == self.candidate else 1
            self.candidate = raw
            self.label = raw if self.count >= self.config['confirm_predictions'] else 'CONFIRMING'
        return {'label': self.label, 'candidate': raw, 'consecutive': self.count,
                'required_predictions': int(self.config['confirm_predictions']),
                'reason': reason, 'features': features,
                'method': 'SMPL24 geometry v2; snapshot-calibrated demo thresholds; consecutive confirmation',
                'side_convention': 'subject anatomical left/right; not screen left/right',
                'warning': 'BODY_NEAR_FRAME_EDGE' if body_at_frame_edge else '',
                'visibility_check': 'severe crop heuristic only; no measured joint confidence'}

ArmRaiseRecognizer = PoseRecognizer
