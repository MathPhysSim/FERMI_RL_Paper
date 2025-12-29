import numpy as np


class SimTangoConnection:
    # Seed laser control simulator

    def __init__(self, **kwargs):

        self.system = 'LocalSimulator'

        # actuators: 2 piezo motors with 2 degree of freedom each
        self.actuators_num = 2
        self.actuators_attr_num = 2

        # sensors: 2 Coupled Charge Device
        self.sensors_num = 2

        # state: piezo-motor inputs
        self.state_size = self.actuators_num * self.actuators_attr_num

        # action: variation of piezo-motor inputs
        self.action_size = self.actuators_num * self.actuators_attr_num

        # state with highest intensity
        if 'target_state' in kwargs:
            self.target_state = kwargs.get('target_state')
        else:
            self.target_state = 131072 * np.ones(self.state_size)
            # piezo-motor range: integers in [0, 262144]

        # response matrix
        self.rm = self.get_respmatrix()

        # state definition
        self.state = self.target_state.copy()

        # position on CCD when the highest intensity is reached
        self.target_position = self.get_position()

        # position definition
        self.position = self.target_position.copy()

        # highest intensity
        self.target_intensity = self.get_intensity()

        # intensity definition
        self.intensity = self.target_intensity.copy()




    def get_respmatrix(self):
        # response matrix
        # position on CCDs =  response matrix * piezo-motor inputs
        rm = np.array([[-1.5570540161682593E-5, +3.2428289038152253E-7, +0.0000000000000000E-0, +0.0000000000000000E-0],
                       [+1.7061003855705661E-6, +1.3362442319301898E-5, +0.0000000000000000E-0, +0.0000000000000000E-0],
                       [+3.6504472405940234E-5, -2.7883739555787350E-8, +2.1631117516360490E-5, -1.0906340491205139E-6],
                       [-2.9017613830940000E-6, -2.6667704592363296E-5, -5.2804805443334150E-7,
                        +8.0338913621924470E-6]])
        return rm

    def get_position(self):
        # return current position on CCD
        position = self.rm.dot(self.state)
        return position

    def set_state(self, state):
        # set state in simulator
        self.state = state

    def get_state(self):
        # return current state
        state = self.state
        return state

    def get_intensity(self):
        # return current intensity
        # intensity calculated on laser spot position on CCDs

        position = self.get_position()

        # initialization of the intensity on each CCD
        screen_intensity = np.zeros(self.sensors_num)
        # acquisition of the intensity on each CCD
        for i in range(self.sensors_num):
            # laser spot position on CCD_i
            screen_position = position[self.sensors_num*i:self.sensors_num*i+2]
            # target position on CCD_i
            target_position = self.target_position[self.sensors_num*i:self.sensors_num*i+2]
            # current position error with respect to target position on CCD_i
            difference = screen_position - target_position
            # absolute value of the distance between the 2 positions
            distance = np.sqrt(np.power(difference, 2))

            # You can adapt this if condition
            if any(distance > 0.1):
                screen_intensity[i] = 0.0
                # if the spot is more distant than 0.1 then the spot is not in the CCD
            else:
                # a gaussian represent the intensity on CCD_i

                # screen_intensity[i] = 1 - np.sqrt(np.sum(distance))
                den = 2*np.power(0.04, 2)
                screen_intensity[i] = np.exp(-np.sum(np.power(difference, 2))/den)
        # NOTICE:
        # here you can play with 0.1 in any(distance > 0.1) and 0.07 in den = 2*np.power(0.07, 2)

        # intensity is given by the product of the CCD intensities
        intensity = np.prod(screen_intensity)
        self.intensity = intensity
        return intensity


if __name__ == '__main__':
    tng = SimTangoConnection()
