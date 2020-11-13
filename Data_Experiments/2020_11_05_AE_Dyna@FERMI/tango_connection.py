import json
import time
import numpy as np
import PyTango as tango

class TangoConnection:

    def __init__(self, conf_file, **kwargs):

        # load json configuration file
        with open(conf_file) as f:
            self.conf_data = json.load(f)

        self.system = self.conf_data['system']

        # get actuators data
        conf_actuators = self.conf_data['actuators']
        self.actuators_data= self.get_confdata(conf_actuators)
        self.actuators_device_num = self.actuators_data[0]
        self.actuators_device_list = self.actuators_data[1]
        self.actuators_device_attr_num = self.actuators_data[2]
        self.actuators_device_attr_list = self.actuators_data[3]

        self.actuators_size = np.sum(self.actuators_device_attr_num)
        self.state_size = self.actuators_size.copy()
        self.action_size = self.actuators_size.copy()
        self.state = np.zeros(self.state_size)

        # get sensors data
        conf_sensors = self.conf_data['sensors']
        self.sensors_data = self.get_confdata(conf_sensors)
        self.sensors_device_num = self.sensors_data[0]
        self.sensors_device_list = self.sensors_data[1]
        self.sensors_device_attr_num = self.sensors_data[2]
        self.sensors_device_attr_list = self.sensors_data[3]

        self.sensors_size = np.sum(self.sensors_device_attr_num)
        self.intensity = np.zeros(1)

        # get spectrometer data
        conf_spectrometer = self.conf_data['spectrometer']
        self.spectrometer_data = self.get_confdata(conf_spectrometer)
        self.spectrometer_device_num = self.spectrometer_data[0]
        self.spectrometer_device_list = self.spectrometer_data[1]
        self.spectrometer_device_attr_num = self.spectrometer_data[2]
        self.spectrometer_device_attr_list = self.spectrometer_data[3]

        # get security data
        conf_security = self.conf_data['security']
        self.security_data = self.get_confdata(conf_security)
        self.security_device_num = self.security_data[0]
        self.security_device_list = self.security_data[1]
        self.security_device_attr_num = self.security_data[2]
        self.security_device_attr_list = self.security_data[3]
        self.security_threshold = 100.

        if 'num_samples' in kwargs:
            self.num_samples = kwargs.get('num_samples')
        else:
            self.num_samples = 25  # 11  # 25  # 51  # 25

        # self.pause = 0.5 + 0.02*self.num_samples
        self.pause = 0.5 + 0.02*self.num_samples + 0.25 
        # self.pause = 0.5 + 0.02*self.num_samples + 1 

        if 'target_state' in kwargs:
            self.target_actuators = kwargs.get('target_state')
        else:
            self.target_actuators = 131072 * np.ones(self.actuators_size)

        if self.system == 'sequencer':
            self.set_state(self.target_actuators)
            self.target_position = self.get_position()

        # read initial values for actuators and sensors
        self.init_state = self.get_state()
        self.init_intensity = self.get_intensity()

        self.state = self.init_state.copy()
        self.intensity = self.init_intensity.copy()


    def get_confdata(self, conf_dev):
        dev_list, dev_attr_num, dev_attr_list = [], [], []
        dev_num = len(conf_dev)
        for j in range(dev_num):
            dev_data = conf_dev[j]
            dev_name = dev_data['host'] + dev_data['address']
            dev = tango.DeviceProxy(dev_name)
            dev_attr = dev_data['attributes']

            dev_list.append(dev)
            dev_attr_num.append(len(dev_attr))
            dev_attr_list.append(dev_attr)
        return [dev_num, dev_list, dev_attr_num, dev_attr_list]

    def get_position(self):
        position = np.zeros(self.sensors_size)
        for i in range(self.sensors_device_num):
            dev = self.sensors_device_list[i]
            for j in range(self.sensors_device_attr_num[i]):
                idx = self.sensors_device_num * i + j
                attr_name = self.sensors_device_attr_list[i][j]
                position[idx] = dev.read_attribute(attr_name).value

        return position

    def set_state(self, state):
        self.check_charge()
        self.set_actuators(state)
        self.state = state


    def get_state(self):
        self.check_charge()
        state = self.get_actuators()
        self.state = state
        return state

    def set_actuators(self, actuators_val):
        
        for i in range(self.actuators_device_num):
            dev = self.actuators_device_list[i]
            for j in range(self.actuators_device_attr_num[i]):
                idx = self.actuators_device_num * i + j
                attr_name = self.actuators_device_attr_list[i][j]
                attr_val = actuators_val[idx]
                dev.write_attribute(attr_name, attr_val)

        time.sleep(self.pause)
        pass

    def get_actuators(self):
        attr_val = np.zeros(self.actuators_size)
        for i in range(self.actuators_device_num):
            dev = self.actuators_device_list[i]
            for j in range(self.actuators_device_attr_num[i]):
                idx = self.actuators_device_num * i + j
                attr_name = self.actuators_device_attr_list[i][j]
                attr_val[idx] = dev.read_attribute(attr_name).value
        return attr_val

    def get_sensors(self):
        attr_val = []
        
        if self.system in ['fel', 'fel1', 'fel2']:
        #if self.system == 'fel' or self.system == 'fel1' or self.system == 'fel2':
            attr_val = np.zeros(self.sensors_size)
            attr_val_seq = np.zeros((self.sensors_size, self.num_samples))
            for i in range(self.sensors_device_num):
                dev = self.sensors_device_list[i]
                for j in range(self.sensors_device_attr_num[i]):
                    idx = self.sensors_device_num * i + j
                    attr_name = self.sensors_device_attr_list[i][j]
                    attr_val_seq[idx] = dev.command_inout(attr_name, [0, int(self.num_samples)])
                    attr_val[idx] = np.median(attr_val_seq[idx])

        elif self.system == 'sequencer':
            position = self.get_position()
            screen_intensity = np.zeros(self.sensors_device_num)
            for i in range(self.sensors_device_num):
                screen_position = position[self.sensors_device_num * i:self.sensors_device_num * i + 2]
                target_position = self.target_position[self.sensors_device_num * i:self.sensors_device_num * i + 2]
                difference = screen_position - target_position
                distance = np.sqrt(np.power(difference, 2))
                if any(distance > 0.1):
                    screen_intensity[i] = 0.0
                else:
                    den = 2 * np.power(0.04, 2)
                    screen_intensity[i] = np.exp(-np.sum(np.power(difference, 2)) / den)
            attr_val = screen_intensity
        #'''    
        elif self.system == 'eos':
            attr_val = np.zeros(self.sensors_size)
            attr_val_seq = np.zeros((self.sensors_size, self.num_samples))
            idx = 0
            for i in range(self.sensors_device_num):
                dev = self.sensors_device_list[i]
                for j in range(self.sensors_device_attr_num[i]):
                    # idx = self.sensors_device_num * i + j
                    attr_name = self.sensors_device_attr_list[i][j]
                    attr_val_seq[idx] = dev.command_inout(attr_name, [0, int(self.num_samples)])
                    attr_val[idx] = np.median(attr_val_seq[idx])
                    idx += 1
        #'''
        return attr_val

    def get_intensity(self):
        self.check_charge()
        attr_val = self.get_sensors()
        intensity = np.prod(attr_val)
        self.intensity = intensity
        return intensity

    def get_image(self):
        self.check_charge()
        attr_val = []
        for i in range(self.spectrometer_device_num):
            dev = self.spectrometer_device_list[i]
            for j in range(self.spectrometer_device_attr_num[i]):
                # idx = self.spectrometer_device_num * i + j
                attr_name = self.spectrometer_device_attr_list[i][j]
                attr_val.append(dev.read_attribute(attr_name).value)
        return attr_val[0]

    def get_security_check(self):
        attr_val = []
        for i in range(self.security_device_num):
            dev = self.security_device_list[i]
            for j in range(self.spectrometer_device_attr_num[i]):
                # idx = self.security_device_num * i + j
                attr_name = self.security_device_attr_list[i][j]
                attr_val.append(dev.read_attribute(attr_name).value)
        return attr_val[0]
    
    def check_charge(self):
        if self.system in ['fel', 'fel1', 'fel2']:
        #if self.system == 'fel' or self.system == 'fel1' or self.system == 'fel2':
        #if self.system in ['eos', 'fel']:
            # print('\nSECURITY CHECK\n')
            flag = 0
            charge = self.get_security_check()
            #while charge < 100.:
            while charge < self.security_threshold:
                flag = 1
                print('\nwait...\n')
                time.sleep(5)
                charge = self.get_security_check()
            
            if flag:
                print('FEL is coming back!\nWait 1 minute more...\n')
                time.sleep(60)
		


if __name__ == '__main__':

    # sequencer
    # system = 'sequencer'
    # path = '/home/niky/PycharmProjects/FERMI/devel/sequencer_new/configuration/'
    
    # fel
    '''
    # system = 'eos'
    system = 'fel2'
    path = '/home/niky/FERMI/2020_10_06/configuration/'
    conf_file = 'conf_'+system+'.json'
    
    filename = path+conf_file

    tng = TangoConnection(conf_file=filename)
    '''
    
