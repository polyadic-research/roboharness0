"""Current-target progress from public evidence; no simulator truth or motor authority."""
from __future__ import annotations

import math
import unicodedata

from rpent.harness.lifecycle import POLICY_PROGRESS_PHASES


def validate_progress_evidence(value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError('progress_evidence must be an object')
    required = {'target_id', 'phase', 'observation_step', 'observed_change'}
    if not required <= value.keys() or value.keys() - required - {'target_distance_m', 'phase_complete'}:
        raise ValueError('progress_evidence requires stable target, phase, exact step and observed change')
    if any(not isinstance(value[k], str) or not value[k].strip() or len(value[k]) > 1000
           for k in ['target_id', 'observed_change']):
        raise ValueError('progress_evidence target_id and observed_change must be non-empty strings')
    if value['phase'] not in POLICY_PROGRESS_PHASES:
        raise ValueError('progress_evidence phase must be a declared policy phase')
    if type(value['observation_step']) is not int or value['observation_step'] < 0:
        raise ValueError('progress_evidence observation_step must be a nonnegative integer')
    if 'phase_complete' in value and type(value['phase_complete']) is not bool:
        raise ValueError('progress_evidence phase_complete must be a boolean')
    distance = value.get('target_distance_m')
    if distance is not None and (isinstance(distance, bool) or not isinstance(distance, (int, float))
                                 or not math.isfinite(distance) or distance < 0):
        raise ValueError('progress_evidence target_distance_m must be finite and nonnegative')


def progress_evidence_schema():
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'target_id': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
            'phase': {'type': 'string', 'enum': list(POLICY_PROGRESS_PHASES)},
            'observation_step': {'type': 'integer', 'minimum': 0},
            'observed_change': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
            'target_distance_m': {'type': 'number', 'minimum': 0},
            'phase_complete': {'type': 'boolean'},
        },
        'required': ['target_id', 'phase', 'observation_step', 'observed_change'],
    }


def review_policy_progress(ledger, value, *, observation_step, physical, distance_threshold):
    """Return (new current-goal progress, newly completed phase).

    The control ledger already scopes this map by stable operation/subtask.
    Neither changing wording nor a new target's first distance establishes progress.
    A completion is an evidence-backed local phase claim, never native success.
    """
    if value is None or not physical or value['observation_step'] != observation_step:
        return False, False
    target = ''.join(c for c in unicodedata.normalize('NFKC', value['target_id']).casefold() if c.isalnum())
    if not target:
        return False, False
    key = (target, value['phase'])
    records = ledger.setdefault('goal_progress', {})
    record = records.setdefault(key, {'step': -1, 'distance': None, 'completed': False})
    if value['observation_step'] <= record['step'] or record['completed']:
        return False, False
    record['step'] = value['observation_step']
    boundary = value.get('phase_complete') is True
    if boundary:
        record['completed'] = True
    distance = value.get('target_distance_m')
    closer = False
    if distance is not None:
        previous = record['distance']
        closer = previous is not None and previous-distance >= distance_threshold
        record['distance'] = distance if previous is None else min(previous, distance)
    return closer or boundary, boundary
