# class SubjectMeta:
#     def __init__(self, subject_id=None, date_completed=None, number_of_learning=None):
#         self.number_of_learning = number_of_learning
#         self.date_completed = date_completed
#         self.subject_id = subject_id


from typing import Any, List, Optional, Dict

from analysis.movement_data import MovementData
from analysis.rotation_data import RotationData


class Subject:
    def __init__(self,
                 meta: Optional[Any] = None,
                 movement_sequence: Optional[Dict[int, List[MovementData]]] = None,
                 rotation_sequence: Optional[Dict[int, List[RotationData]]] = None,
                 name: Optional[str] = None,
                 timeout_trials: Optional[List[int]] = None):
        """
        A class representing a subject in a study.

        :param meta: Subject's metadata.
        :param movement_sequence: A list of subject's movement sequences.
        :param rotation_sequence: A list of subject's rotation sequences.
        :param name: The name of the subject.
        :param timeout_trials: A list of trial numbers where the subject timed out.
        """
        self.name = name
        self.meta = meta
        self.movement_sequence = movement_sequence
        self.rotation_sequence = rotation_sequence
        self.timeout_trials = timeout_trials
