class TrialConfiguration:
    def __init__(self, row_generator):
        self.configuration = {}
        for row in row_generator:
            trial_name = row["TrialName"]
            self.configuration[trial_name] = row

    def get_source_destination_pair_by_name(self, name):
        return self.configuration[name]["Source"], self.configuration[name]["Destination"]

    def get_true_angle(self, name):
        return float(self.configuration[name]["TrueAngle"])

    pass
