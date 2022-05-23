class Light(object):
    def __init__(self, position, intensity):
        '''
        @param
            `position` --np.array --shape=(3,)
            `intensity` --np.array --shape=(3,)
        '''
        self.position = position
        self.intensity = intensity
        