"""gustaf/gustaf/utils/tictoc.py.

Timer that tics, tocs and logs.
"""
from time import perf_counter as now

from gustaf._base import GustafBase


class Tic(GustafBase):
    """
    Timer class for easier time measurement.
    """

    __slots__ = ("_title", "_names", "_laps", "_logger")

    # add short cut to perf_counter in case you just want pure "now"
    now = now

    def __init__(self, title="untitled", log_level="debug"):
        """
        Configures the timer and logger. Marks the starting point.

        Parameters
        ----------
        title: str
          title for this measurement.
        log_level: str
          Valid options are {"info", "debug", "warning"}
        """
        # select logger based on logger level
        self._logger = eval(f"self._log{str(log_level).lower()[0]}")

        # initialize names
        self._names = []

        # give name
        self._title = str(title)

        # start
        self._laps = [now()]

    def toc(self, name=None, log=False):
        """
        Adds now to the measurements.

        Parameters
        ----------
        name: str
          name of this specific measurement.
          By default it assign a number to this lap.
        log: bool
          If True, will log lapsed time.

        Returns
        -------
        None
        """
        # measure now
        self._laps.append(now())

        self._names.append(
            f"lap-{len(self._laps) - 2}" if name is None else str(name)
        )

        if log:
            self._logger("time lapsed: ", self._laps[-2] - self._laps[-1])

    def summary(self, print_=False):
        """
        Prints measurement summany.

        Parameters
        ----------
        print_: bool


        Returns
        -------
        records: tuple
          Records starting from starting point. tuple of names and cummulative
          lapsed time.
        """
        message = [f"\n+++ {self._title} - time logs +++\n"]

        name_width = int(max([len(n) for n in self._names]))

        # extract info
        start = self._laps[0]
        cummulative_raw = [lap - start for lap in self._laps[1:]]
        cummulative = [f"{c:.10f}" for c in cummulative_raw]
        diff = [
            f"{l1 - l0:.10f}"
            for l0, l1 in zip(self._laps[:-1], self._laps[1:])
        ]
        diff_width = int(max([len(d) for d in diff]))
        cumm_width = int(max([len(c) for c in cummulative]))

        message.append(
            f"{'names'.ljust(name_width)} | "
            f"{'diff'.rjust(diff_width)} | "
            f"{'cummulative'.rjust(cumm_width)}\n"
        )

        for n, d, c in zip(self._names, diff, cummulative):
            this_line = (
                f"{n.ljust(name_width)} | "
                f"{d.rjust(diff_width)} | "
                f"{c.rjust(cumm_width)}\n"
            )
            message.append(this_line)

        self._logger(*message)
        if print_:
            print(*message)

        return self._names.copy(), cummulative
