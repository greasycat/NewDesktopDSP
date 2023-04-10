# class SubjectMeta:
#     def __init__(self, subject_id=None, date_completed=None, number_of_learning=None):
#         self.number_of_learning = number_of_learning
#         self.date_completed = date_completed
#         self.subject_id = subject_id


class Subject:
    def __init__(self, meta=None, movement_sequence=None, rotation_sequence=None, name=None, timeout_trials=None):
        self.name = name
        self.meta = meta
        self.movement_sequence = movement_sequence
        self.rotation_sequence = rotation_sequence
        self.timeout_trials = timeout_trials
        

