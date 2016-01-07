"""Provide the device class for charge transformers."""

# Imports
import sys
import time
import traceback
import itertools
import numpy as np

# RedPitaya imports
from rpyc import connect
from PyRedPitaya.pc import RedPitaya
from PyRedPitaya.instrument import TriggerSource

# PyTango imports
from PyTango import DevState, server
from PyTango.server import Device, DeviceMeta
from PyTango.server import command, attribute, device_property

# Constants

F0 = 125e6  # Hz
BASE_LENGTH = 2**14  # Points
DECIMATIONS = [1, 8, 64, 1024, 8192, 65536]


# Helper
def find_settings(time_range, time_position):
    decimation = F0 * time_range / BASE_LENGTH
    try:
        decimation = next(x for x in DECIMATIONS if x >= decimation)
    except StopIteration:
        raise ValueError("Can't achieve requested time range")
    frequency = F0 / decimation
    new_range = BASE_LENGTH / frequency
    new_position = float(time_position * new_range) / time_range
    delay_length = int(round(new_position * frequency))
    new_position = delay_length / frequency
    return decimation, frequency, new_range, new_position, delay_length


# Red Pitaya charge transformer device
class RpScope(Device):
    """Tango Device for a Red Pitaya charge transformer.

    Device states description:
        - **DevState.RUNNING**: the board is acquiring
        - **DevState.FAULT**: any error
    """
    __metaclass__ = DeviceMeta

    # Constants
    SOURCES = ['CHA', 'CHB', 'EXT']
    EDGES = ['PE', 'NE']
    TRIGGERS = [
        TriggerSource.chA_posedge,
        TriggerSource.chA_negedge,
        TriggerSource.chB_posedge,
        TriggerSource.chB_negedge,
        TriggerSource.ext_posedge,
        TriggerSource.ext_negedge]
    TRIGGER_DICT = dict(zip(itertools.product(SOURCES, EDGES), TRIGGERS))

    # Settings
    port = 18861
    acquisition_timeout = 2.0

    # Device properties

    Host = device_property(
        dtype=str,
        doc='Host name of the RedPitaya.')

    TriggerSource = device_property(
        dtype=str,
        default_value='CHA',
        doc="Can be CHA for channel A, CHB for channel B or EXT for external.")

    TriggerEdge = device_property(
        dtype=str,
        default_value='PE',
        doc="Can be PE for positive edge or NE for negative edge.")

    TriggerLevel = device_property(
        dtype=float,
        default_value=0.0,
        doc="Trigger level in volts.")

    TimeRange = device_property(
        dtype=float,
        default_value=1e-3,
        doc="Minimum time range for the waveforms.")

    TimePosition = device_property(
        dtype=float,
        default_value=0.0,
        doc="Minimum time position for the waveforms")

    Gain = device_property(
        dtype=float,
        default_value=1.0)

    Offset = device_property(
        dtype=float,
        default_value=0.0)

    # Waveform attributes

    Waveform1 = attribute(
        dtype=(float,),
        max_dim_x=10**6,
        label='Waveform 1',
        doc="Waveform on channel 1.")

    def read_Waveform1(self):
        return self.waveform1

    def is_Waveform1_allowed(self, attr):
        return self.connected

    Waveform2 = attribute(
        dtype=(float,),
        max_dim_x=10**6,
        label='Waveform 2',
        doc="Waveform on channel 2.")

    def read_Waveform2(self):
        return self.waveform2

    def is_Waveform2_allowed(self, attr):
        return self.connected

    TimeBase = attribute(
        dtype=(float,),
        max_dim_x=10**6,
        label='Time base',
        doc="Time base for the waveforms")

    def read_TimeBase(self):
        return self.timebase

    def is_TimeBase_allowed(self, attr):
        return self.connected

    # State methods

    @property
    def connected(self):
        return self.get_state() != DevState.FAULT

    def register_exception(self, exc, context=None, method=None):
        if context is not None:
            status = 'Exception while {0}: {1!r}'.format(context, exc)
        elif method is not None:
            status = "Exception in '{0}': {1!r}".format(method.__name__, exc)
        else:
            status = 'Exception: {1!r}'.format(exc)
        self.error_stream(status)
        self.debug_stream(traceback.format_exc().replace('%', '%%'))
        self.set_status(status)
        self.set_state(DevState.FAULT)

    # Initialization and deletion

    def get_device_properties(self):
        super(RpScope, self).get_device_properties()
        if self.TriggerSource not in self.SOURCES:
            msg = "'{0}' is not a valid trigger source"
            raise ValueError(msg.format(self.TriggerSource))
        if self.TriggerEdge not in self.EDGES:
            msg = "'{0}' is not a valid trigger edge"
            raise ValueError(msg.format(self.TriggerEdge))

    def init_device(self):
        """Initialize the device."""
        # Initialize variables
        self.deadline = None
        self.interface = None
        self.connection = None
        self.waveform1 = None
        self.waveform2 = None
        # Initialize settings
        self.trigger = None
        self.decimation = None
        self.frequency = None
        self.time_range = None
        self.time_position = None
        self.delay_length = None
        # Initialize hardware
        try:
            context = 'parsing properties'
            super(RpScope, self).init_device()
            context = 'connecting to the red pitaya'
            self.connection = connect(self.Host, self.port)
            context = 'instantiating interface'
            self.interface = RedPitaya(self.connection).scope
            context = 'setting trigger source and edge'
            self.trigger = self.interface.trigger_source = \
                self.TRIGGER_DICT[self.TriggerSource, self.TriggerEdge]
            context = 'setting trigger level'
            self.interface.threshold_ch1 = self.volts_to_int(self.TriggerLevel)
            self.interface.threshold_ch2 = self.volts_to_int(self.TriggerLevel)
            context = 'finding settings'
            settings = find_settings(self.TimeRange, self.TimePosition)
            self.decimation, self.frequency, self.time_range, \
                self.time_position, self.delay_length = settings
            context = 'setting trigger delay'
            self.interface.trigger_delay = BASE_LENGTH - self.delay_length
            context = 'setting decimation'
            self.interface.decimation = self.decimation
            context = 'building time base'
            self.time_base = self.build_time_base()
            context = 'running first acquisition'
            self.check_acquisition()
        # Fault state
        except Exception as exc:
            return self.register_exception(exc, context)
        # Running state
        else:
            self.set_state(DevState.RUNNING)
            self.set_status("Red pitaya connected, configured and running.")

    def build_time_base(self):
        start = -self.time_position
        step = self.frequency
        stop = BASE_LENGTH * step + start
        return np.arange(start, step, stop)

    def prepare_acquisition(self):
        self.interface.reset_writestate_machine()
        self.interface.trigger_source = self.trigger
        self.interface.arm_trigger()
        time.sleep(self.time_position)
        self.deadline = time.time() + self.acquisition_timeout

    def check_acquisition(self):
        if self.deadline is None:
            self.prepare_acquisition()
        if time.time() > self.deadline:
            raise RuntimeError("Acquisition timed out")
        if self.interface.trigger_source != 0:
            return False
        self.get_waveforms()
        self.prepare_acquisition()
        return True

    def get_waveforms(self):
        # Acquire waveforms
        raw1, raw2 = self.interface.rawdata_ch1, self.interface.rawdata_ch2
        # Fix PyRedPitaya bug
        raw1[raw1 >= 2**13] -= 2**14
        raw2[raw2 >= 2**13] -= 2**14
        # Roll and convert waveforms
        offset = self.delay_length - int(self.interface.write_pointer_trigger)
        self.waveform1 = self.int_to_volts(np.roll(raw1, offset))
        self.waveform2 = self.int_to_volts(np.roll(raw2, offset))

    def int_to_volts(self, value):
        normalized = value / float(2**14)
        return normalized * self.Gain + self.Offset

    def volts_to_int(self, value):
        normalized = (value - self.Offset) / self.Gain
        return int(normalized * 2**14)

    def delete_device(self):
        """Delete device."""
        try:
            if self.connection:
                self.connection.close()
        except Exception as exc:
            self.register_exception(exc, 'closing connection')
        finally:
            self.connection = None
            self.interface = None

    # Commands

    @command
    def PolledRun(self):
        """Poll this command for continuous acquisition."""
        try:
            self.check_acquisition()
        except Exception as exc:
            self.register_exception(exc, 'checking acquisition')

    def is_PolledRun_allowed(self):
        return self.get_state() == DevState.RUNNING

    @command
    def RunSingle(self):
        """Run a single acquisition."""
        try:
            self.prepare_acquisition()
            while not self.check_acquisition():
                pass
        except Exception as exc:
            self.register_exception(exc, 'running acquisition')

    def is_RunSingle_allowed(self):
        return self.get_state() == DevState.RUNNING

    # Run class method

    @classmethod
    def run_server(cls, args=None, **kwargs):
        """Run the class as a device server.
        It is based on the PyTango.server.run method.

        The difference is that the device class
        and server name are automatically given.

        Args:
            args (iterable): args as given in the PyTango.server.run method
                             without the server name. If None, the sys.argv
                             list is used
            kwargs: the other keywords argument are as given
                    in the PyTango.server.run method.
        """
        if args is None:
            args = sys.argv[1:]
        args = [cls.__name__] + list(args)
        return server.run((cls,), args, **kwargs)

# Aliases
RP_SCOPE_SERVER = RpScope.__name__
run = RpScope.run_server


# Main execution
if __name__ == "__main__":
    run()
