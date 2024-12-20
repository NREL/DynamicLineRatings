import os
import argparse
from datetime import datetime
from dlr import paths, helpers, physics, linerating

class DictArg(argparse.Action):
    """Class to enable inputting dictionary-structured arguments through the CLI."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        optional_str: str,
    ) -> None:
        """Save arguments in the namespace."""
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value
    
def process_cli_args(args):
    """Checks for valid CLI arguments and converts numeric arguments to floats."""

    valid_wind_sources = ["wtk"]
    valid_irradiance_pairs = ["nsrdb-ghi", "nsrdb-clearsky_ghi"]
    valid_conductor_params = ["temperature", "emissivity", "absorptivity"]
    valid_forecast_margin_params = [
        "windspeed", "wind_conductor_angle", "temperature", "pressure", "irradiance"
    ]
    args_dict = args.__dict__
    for key, value in args_dict.items():
        if isinstance(value, str):
            if value.replace('.', '').isnumeric():
                value = float(value)
                if key == "temperature":
                    value += physics.C2K
                args_dict[key] = float(value)
            else:
                match key:
                    case "windspeed" | "wind_conductor_angle" | "temperature" | "pressure":
                        if value not in valid_wind_sources:
                            raise NotImplementedError(
                                f"The acceptable data sources for {key} are "
                                f"{valid_wind_sources}. Handling for '{value}' "
                                "is not implemented."
                            )
                    case "irradiance":
                        if value not in valid_irradiance_pairs:
                            raise NotImplementedError(
                                f"The acceptable source-type pairs for {key} are "
                                f"{valid_irradiance_pairs}. Handling for '{value}' is "
                                "not implemented."
                            )
        else:
            match key:
                case "conductor_params":
                    for param, param_value in args_dict[key].items():
                        if param not in valid_conductor_params:
                            raise NotImplementedError(
                                f"The acceptable conductor parameters are "
                                f"{valid_conductor_params}. Handling for '{param}' is "
                                "not implemented."
                            )
                        assert param_value.replace('.', '').isnumeric(), \
                            f"Provided value '{param_value}' for conductor {param} is non-numeric."
                    args_dict[key] = {k: float(v) for k, v in args_dict[key].items()}
                case "forecast_margin":
                    for param, param_value in args_dict[key].items():
                        if param not in valid_forecast_margin_params:
                            raise NotImplementedError(
                                f"The acceptable conductor parameters are "
                                f"{valid_forecast_margin_params}. Handling for '{param}' "
                                "is not implemented."
                            )
                        assert param_value.replace('.', '').isnumeric(), \
                            f"Provided value '{param_value}' for {param} forecast margin is non-numeric."
                        args_dict[key] = {k: float(v) for k, v in args_dict[key].items()}
    return args_dict

def run(
    line_idx_start: int,
    line_idx_end: int,
    years: int | list[int] = list(range(2007, 2014)),
    windspeed: float | str = 0.61,
    pressure: float | str = 101325,
    temp_ambient_air: float | str = 40 + physics.C2K,
    wind_conductor_angle: float | str = 90,
    solar_ghi: float | str = 1000,
    conductor_params: dict = {},
    forecast_margin: dict = {}
):
    """Calculate hourly ratings for a set of lines as a function of weather and conductor parameters.

    Args:
        line_idx_start: Starting index of subset of lines to process, based on data source specified
            in 'paths.lines'
        line_idx_end: Ending index of subset of lines to process, based on data source specified in
            'paths.lines'
        years: int or list representing year(s) of historic weather data
        windspeed (numeric, str): Static windspeed [m/s] or data source for variable windspeed 
            (e.g., 'wtk')
        pressure (numeric, str): Static air pressure [Pa] or data source for variable pressure 
            (e.g., 'wtk')
        temp_ambient_air (numeric, str): Static air temperature [K] or data source for variable temp
            (e.g., 'wtk')
        wind_conductor_angle (numeric): Static angle between wind direction and line segment [°] or
            data source for variable wind direction (e.g., 'wtk')
        solar_ghi (numeric): Static solar global horizontal irradiance [W m^-2] or '-'-delimited
            pair of variable irradiance data source and irradiance type (e.g., 'nsrdb-ghi')
        conductor_params (dict[str, numeric]): Dictionary to override default values for
            conductor temperature (75°C), absorptivity (0.8), and emissivity (0.8)
        forecast_margin (dict[str, numeric]): Additive adjustments to apply to each
            weather parameter
    """
    now = datetime.now()
    timestamp = f"{now.year}{str(now.month).zfill(2)}{str(now.day).zfill(2)}"
    out_dirpath = os.path.join(paths.outputs, timestamp)
    os.makedirs(out_dirpath, exist_ok=True)

    line_idx_range = slice(line_idx_start, line_idx_end + 1)
    dflines = helpers.read_lines(paths.lines, line_idx_range)

    temp_conductor = conductor_params.get("temperature", 75) + physics.C2K
    dflines = helpers.lookup_diameter_resistance(
        dflines=dflines,
        temp_conductor_kelvin=temp_conductor
    )

    emissivity_conductor = conductor_params.get("emissivity", 0.8)
    absorptivity_conductor = conductor_params.get("absorptivity", 0.8)
    for iline, line in dflines.iterrows():
        hourly_ratings = linerating.calc_ratings(
            line=line,
            years=years,
            windspeed=windspeed,
            pressure=pressure,
            temp_ambient_air=temp_ambient_air,
            wind_conductor_angle=wind_conductor_angle,
            solar_ghi=solar_ghi,
            temp_conductor=temp_conductor,
            emissivity_conductor=emissivity_conductor,
            absorptivity_conductor=absorptivity_conductor,
            forecast_margin=forecast_margin
        )
        hourly_ratings.to_hdf(
            os.path.join(out_dirpath, f"line_ratings_{line_idx_start}_{line_idx_end}.h5"),
            key=f"line_{iline}"
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--start", required=False, nargs="?", type=int, help=""
    )
    parser.add_argument(
        "-e", "--end", required=False, nargs="?", type=int, help=""
    )
    parser.add_argument(
        "-y", "--years", required=False, nargs="+", default=list(range(2007,2014)),
        type=int, help="Years to derive weather data for (options: 2007 through 2013)"
    )
    parser.add_argument(
        "-w", "--windspeed", required=False, nargs="?", default=0.61,
        help="Static windspeed (in m/s) or data source for variable windspeed (options: ['wtk'])"
    )
    parser.add_argument(
        "-d", "--wind_conductor_angle", required=False, nargs="?", default=90,
        help="Static angle between wind and conductor (in degrees) or data source for "
        "variable wind direction (options: ['wtk'])"
    )
    parser.add_argument(
        "-t", "--temperature", required=False, nargs="?", default=40,
        help="Static ambient air temperature (in Celsius) or data source for "
        "variable air temp (options: ['wtk'])"
    )
    parser.add_argument(
        "-p", "--pressure", required=False, nargs="?", default=101325,
        help="Static air pressure (in Pa) or data source for variable pressure "
        "(options: ['wtk'])"
    )
    parser.add_argument(
        "-i", "--irradiance", required=False, nargs="?", default=1000,
        help="Static solar irradiance (in W/m**2) or '-'-delimited pair of data source for "
        "variable irradiance and irradiance type (options: ['nsrdb-ghi', 'nsrdb-clearsky_ghi'])"
    )
    parser.add_argument(
        "-c", "--conductor_params", required=False, nargs="*", action=DictArg, default={},
        help="Accepted parameters are 'temperature' (in C), 'emissivity', and 'absorptivity'"
    )
    parser.add_argument(
        "-m", "--forecast_margin", required=False, nargs="*", action=DictArg, default={},
        help="Accepted parameters are 'windspeed' (m/s), 'wind_conductor_angle' (degrees), "
        "'temperature' (C), 'pressure' (Pa), and 'irradiance' (W/m**2)"
    )    

    args = process_cli_args(parser.parse_args())
    run(
        line_idx_start=args["start"],
        line_idx_end=args["end"],
        years=args["years"],
        windspeed=args["windspeed"],
        pressure=args["pressure"],
        temp_ambient_air=args["temperature"],
        wind_conductor_angle=args["wind_conductor_angle"],
        solar_ghi=args["irradiance"],
        conductor_params=args["conductor_params"],
        forecast_margin=args["forecast_margin"],
    )