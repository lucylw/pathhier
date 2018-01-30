

# class for representing a dictionary where the values are incremental
# integers as new keys are added
class IncrementDict:
    def __init__(self):
        self.content = dict()
        self.index = 0

    def __repr__(self):
        return self.content.__repr__()

    def get(self, item):
        """
        If item is not in the dictionary, add it, then return the corresponding dict value
        :param item:
        :return:
        """
        if item not in self.content:
            self.content[item] = self.index
            self.index += 1
        return self.content[item]