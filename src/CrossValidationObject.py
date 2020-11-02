class CrossValidationObject():
    def __init__(self, x_train, x_val, x_test, y_train, y_val, y_test):
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = x_train, x_val, x_test, y_train, y_val, y_test
