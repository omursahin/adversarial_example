class AdaptiveParameterControl:
    def __init__(self, problem):
        self.problem = problem
        self.threshold = problem.conf.APC_THRESHOLD
        self.start_time = problem.conf.APC_START_TIME
        self.pixel_start_value = problem.conf.APC_PIXEL_START_VALUE
        self.pixel_end_value = problem.conf.APC_PIXEL_END_VALUE
        self.noise_start_value = problem.conf.APC_NOISE_START_VALUE
        self.noise_end_value = problem.conf.APC_NOISE_END_VALUE

    def get_dpc_value(self, start, end):
        passed = self.problem.percentage_used_budget()

        if passed < self.start_time:
            return start

        if passed >= self.threshold:
            return end

        scale = (passed - self.start_time) / (self.threshold - self.start_time)
        delta = end - start

        return start + delta * scale

