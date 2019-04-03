class Constants:
    def __init__(self, alpha, gamma, const, no_steps):
        self.alpha = alpha
        self.gamma = gamma
        self.const = const
        self.no_steps = no_steps

    def __str__(self):
        return 'alpha=' + str(self.alpha) + 'gamma=' + str(self.gamma) + 'c=' + str(self.const)
