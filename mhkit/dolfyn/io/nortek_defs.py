import numpy as np

nan = np.nan


class _VarAtts:
    """
    A data variable attributes class.

    Parameters
    ----------
    dims : (list, optional)
      The dimensions of the array other than the 'time'
      dimension. By default the time dimension is appended to the
      end. To specify a point to place it, place 'n' in that
      location.
    dtype : (type, optional)
      The data type of the array to create (default: float32).
    group : (string, optional)
      The data group to which this variable should be a part
      (default: 'main').
    view_type : (type, optional)
      Specify a numpy view to cast the array into.
    default_val : (numeric, optional)
      The value to initialize with (default: use an empty array).
    offset : (numeric, optional)
      The offset, 'b', by which to adjust the data when converting to
      scientific units.
    factor : (numeric, optional)
      The factor, 'm', by which to adjust the data when converting to
      scientific units.
    title_name : (string, optional)
      The name of the variable.
    units : (string, optional)
      The units of this variable.
    dim_names : (list, optional)
      A list of names for each dimension of the array.
    """

    def __init__(
        self,
        dims=[],
        dtype=None,
        group="data_vars",
        view_type=None,
        default_val=None,
        offset=0,
        factor=1,
        title_name=None,
        units="1",
        dim_names=None,
        long_name="",
        standard_name="",
    ):
        self.dims = list(dims)
        if dtype is None:
            dtype = np.float32
        self.dtype = dtype
        self.group = group
        self.view_type = view_type
        self.default_val = default_val
        self.offset = offset
        self.factor = factor
        self.title_name = title_name
        self.units = units
        self.dim_names = dim_names
        self.long_name = long_name
        self.standard_name = standard_name

    def shape(self, **kwargs):
        a = list(self.dims)
        hit = False
        for ky in kwargs:
            if ky in self.dims:
                hit = True
                a[a.index(ky)] = kwargs[ky]
        if hit:
            return a
        else:
            return self.dims + [kwargs["n"]]

    def _empty_array(self, **kwargs):
        out = np.zeros(self.shape(**kwargs), dtype=self.dtype)
        try:
            out[:] = np.NaN
        except:
            pass
        if self.view_type is not None:
            out = out.view(self.view_type)
        return out

    def sci_func(self, data):
        """
        Scale the data to scientific units.

        Parameters
        ----------
        data : :class:`<numpy.ndarray>`
          The data to scale.

        Returns
        -------
        retval : {None, data}
          If this funciton modifies the data in place it returns None,
          otherwise it returns the new data object.
        """

        if self.offset != 0:
            data += self.offset
        if self.factor != 1:
            data *= self.factor
            return data


vec_data = {
    "AnaIn2LSB": _VarAtts(
        dims=[],
        dtype=np.uint8,
        group="sys",
    ),
    "Count": _VarAtts(
        dims=[],
        dtype=np.uint8,
        group="sys",
        units="1",
    ),
    "PressureMSB": _VarAtts(
        dims=[],
        dtype=np.uint8,
        group="data_vars",
    ),
    "AnaIn2MSB": _VarAtts(
        dims=[],
        dtype=np.uint8,
        group="sys",
    ),
    "PressureLSW": _VarAtts(
        dims=[],
        dtype=np.uint16,
        group="data_vars",
    ),
    "AnaIn1": _VarAtts(
        dims=[],
        dtype=np.uint16,
        group="sys",
    ),
    "vel": _VarAtts(
        dims=[3],
        dtype=np.float32,
        group="data_vars",
        factor=0.001,
        default_val=nan,
        units="m s-1",
        long_name="Water Velocity",
    ),
    "amp": _VarAtts(
        dims=[3],
        dtype=np.uint8,
        group="data_vars",
        units="1",
        long_name="Acoustic Signal Amplitude",
        standard_name="signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water",
    ),
    "corr": _VarAtts(
        dims=[3],
        dtype=np.uint8,
        group="data_vars",
        units="%",
        long_name="Acoustic Signal Correlation",
    ),
}

vec_sysdata = {
    "time": _VarAtts(
        dims=[],
        dtype=np.float64,
        group="coords",
        default_val=nan,
        units="seconds since 1970-01-01 00:00:00 UTC",
        long_name="Time",
        standard_name="time",
    ),
    "batt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="V",
        long_name="Battery Voltage",
    ),
    "c_sound": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="m s-1",
        long_name="Speed of Sound",
        standard_name="speed_of_sound_in_sea_water",
    ),
    "heading": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Heading",
        standard_name="platform_orientation",
    ),
    "pitch": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Pitch",
        standard_name="platform_pitch",
    ),
    "roll": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Roll",
        standard_name="platform_roll",
    ),
    "temp": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.01,
        units="degree_C",
        long_name="Temperature",
        standard_name="sea_water_temperature",
    ),
    "error": _VarAtts(
        dims=[],
        dtype=np.uint8,
        group="data_vars",
        default_val=nan,
        long_name="Error Code",
    ),
    "status": _VarAtts(
        dims=[],
        dtype=np.uint8,
        group="data_vars",
        default_val=nan,
        long_name="Status Code",
    ),
    "AnaIn": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="sys",
        default_val=nan,
    ),
    "orientation_down": _VarAtts(
        dims=[],
        dtype=bool,
        group="data_vars",
        default_val=nan,
        long_name="Orientation of ADV Communication Cable",
    ),
}

awac_profile = {
    "time": _VarAtts(
        dims=[],
        dtype=np.float64,
        group="coords",
        units="seconds since 1970-01-01 00:00:00 UTC",
        long_name="Time",
        standard_name="time",
    ),
    "error": _VarAtts(
        dims=[],
        dtype=np.uint16,
        group="data_vars",
        long_name="Error Code",
    ),
    "AnaIn1": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="sys",
        default_val=nan,
        units="n/a",
    ),
    "batt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="V",
        long_name="Battery Voltage",
    ),
    "c_sound": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="m s-1",
        long_name="Speed of Sound",
        standard_name="speed_of_sound_in_sea_water",
    ),
    "heading": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Heading",
        standard_name="platform_orientation",
    ),
    "pitch": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Pitch",
        standard_name="platform_pitch",
    ),
    "roll": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Roll",
        standard_name="platform_roll",
    ),
    "pressure": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="dbar",
        long_name="Pressure",
        standard_name="sea_water_pressure",
    ),
    "status": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        long_name="Status Code",
    ),
    "temp": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.01,
        units="degree_C",
        long_name="Temperature",
        standard_name="sea_water_temperature",
    ),
    "vel": _VarAtts(
        dims=[3, "nbins", "n"],  # how to change this for different # of beams?
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="m s-1",
        long_name="Water Velocity",
    ),
    "amp": _VarAtts(
        dims=[3, "nbins", "n"],
        dtype=np.uint8,
        group="data_vars",
        units="1",
        long_name="Acoustic Signal Amplitude",
        standard_name="signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water",
    ),
}

waves_hdrdata = {
    "time_alt": _VarAtts(
        dims=[],
        dtype=np.float64,
        group="coords",
        default_val=nan,
        units="seconds since 1970-01-01 00:00:00 UTC",
        long_name="Time",
        standard_name="time",
    ),
    "batt_alt": _VarAtts(
        dims=[],
        dtype=np.uint16,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="V",
        long_name="Battery Voltage",
    ),
    "c_sound_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="m s-1",
        long_name="Speed of Sound",
        standard_name="speed_of_sound_in_sea_water",
    ),
    "heading_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Heading",
        standard_name="platform_orientation",
    ),
    "pitch_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Pitch",
        standard_name="platform_pitch",
    ),
    "roll_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.1,
        units="degree",
        long_name="Roll",
        standard_name="platform_roll",
    ),
    "pressure1_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="dbar",
        long_name="Pressure Min",
        standard_name="sea_water_pressure",
    ),
    "pressure2_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="dbar",
        long_name="Pressure Max",
        standard_name="sea_water_pressure",
    ),
    "temp_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.01,
        units="degree_C",
        long_name="Temperature",
        standard_name="sea_water_temperature",
    ),
}

waves_data = {
    "pressure_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="dbar",
        long_name="Pressure",
        standard_name="sea_water_pressure",
    ),
    "dist1_alt": _VarAtts(
        dims=[],
        dtype=np.uint16,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="m",
        long_name="AST distance1 on vertical beam",
        standard_name="altimeter_range",
    ),
    "dist2_alt": _VarAtts(
        dims=[],
        dtype=np.uint16,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="m",
        long_name="AST distance2 on vertical beam",
        standard_name="altimeter_range",
    ),
    "AnaIn1_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="sys",
        default_val=nan,
        units="n/a",
    ),
    "vel_alt": _VarAtts(
        dims=[4, "n"],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        factor=0.001,
        units="m s-1",
        long_name="Water Velocity",
    ),
    "amp_alt": _VarAtts(
        dims=[4, "n"],
        dtype=np.uint8,
        group="data_vars",
        default_val=nan,
        units="1",
        long_name="Acoustic Signal Amplitude",
        standard_name="signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water",
    ),
    "quality_alt": _VarAtts(
        dims=[],
        dtype=np.float32,
        group="data_vars",
        default_val=nan,
        units="1",
        long_name="Altimeter Quality Indicator",
    ),
}
