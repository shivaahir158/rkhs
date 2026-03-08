class PriorityTemplate:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def score(self, feat_dict, theta):
        return sum(theta[name] * feat_dict[name] for name in self.feature_names)