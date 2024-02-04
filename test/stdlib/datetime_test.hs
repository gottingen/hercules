from datetime import *
from unittest import TestCase
from operator import lt, le, gt, ge, eq, ne, truediv, floordiv, mod


class TestTimeDelta(Static[TestCase]):
    def test_constructor(self):
        eq = self.assertEqual
        td = timedelta

        # Check keyword args to constructor
        eq(
            td(),
            td(
                weeks=0,
                days=0,
                hours=0,
                minutes=0,
                seconds=0,
                milliseconds=0,
                microseconds=0,
            ),
        )
        eq(td(1), td(days=1))
        eq(td(0, 1), td(seconds=1))
        eq(td(0, 0, 1), td(microseconds=1))
        eq(td(weeks=1), td(days=7))
        eq(td(days=1), td(hours=24))
        eq(td(hours=1), td(minutes=60))
        eq(td(minutes=1), td(seconds=60))
        eq(td(seconds=1), td(milliseconds=1000))
        eq(td(milliseconds=1), td(microseconds=1000))

        # Check float args to constructor
        eq(td(weeks=1.0 / 7), td(days=1))
        eq(td(days=1.0 / 24), td(hours=1))
        eq(td(hours=1.0 / 60), td(minutes=1))
        eq(td(minutes=1.0 / 60), td(seconds=1))
        eq(td(seconds=0.001), td(milliseconds=1))
        eq(td(milliseconds=0.001), td(microseconds=1))

    def test_computations(self):
        eq = self.assertEqual
        td = timedelta

        a = td(7)  # One week
        b = td(0, 60)  # One minute
        c = td(0, 0, 1000)  # One millisecond
        eq(a + b + c, td(7, 60, 1000))
        eq(a - b, td(6, 24 * 3600 - 60))
        # eq(b.__rsub__(a), td(6, 24*3600 - 60))
        eq(-a, td(-7))
        eq(+a, td(7))
        eq(-b, td(-1, 24 * 3600 - 60))
        eq(-c, td(-1, 24 * 3600 - 1, 999000))
        eq(abs(a), a)
        eq(abs(-a), a)
        eq(td(6, 24 * 3600), a)
        eq(td(0, 0, 60 * 1000000), b)
        eq(a * 10, td(70))
        eq(a * 10, 10 * a)
        eq(a * 10, 10 * a)
        eq(b * 10, td(0, 600))
        eq(10 * b, td(0, 600))
        eq(b * 10, td(0, 600))
        eq(c * 10, td(0, 0, 10000))
        eq(10 * c, td(0, 0, 10000))
        eq(c * 10, td(0, 0, 10000))
        eq(a * -1, -a)
        eq(b * -2, -b - b)
        eq(c * -2, -c + -c)
        eq(b * (60 * 24), (b * 60) * 24)
        eq(b * (60 * 24), (60 * b) * 24)
        eq(c * 1000, td(0, 1))
        eq(1000 * c, td(0, 1))
        eq(a // 7, td(1))
        eq(b // 10, td(0, 6))
        eq(c // 1000, td(0, 0, 1))
        eq(a // 10, td(0, 7 * 24 * 360))
        eq(a // 3600000, td(0, 0, 7 * 24 * 1000))
        eq(a / 0.5, td(14))
        eq(b / 0.5, td(0, 120))
        eq(a / 7, td(1))
        eq(b / 10, td(0, 6))
        eq(c / 1000, td(0, 0, 1))
        eq(a / 10, td(0, 7 * 24 * 360))
        eq(a / 3600000, td(0, 0, 7 * 24 * 1000))

        # Multiplication by float
        us = td(microseconds=1)
        eq((3 * us) * 0.5, 2 * us)
        eq((5 * us) * 0.5, 2 * us)
        eq(0.5 * (3 * us), 2 * us)
        eq(0.5 * (5 * us), 2 * us)
        eq((-3 * us) * 0.5, -2 * us)
        eq((-5 * us) * 0.5, -2 * us)

        # TODO: check Python's Issue #23521 and possibly
        # incorporate the same fix here. For now a couple
        # tests are disabled.

        # Issue #23521
        eq(td(seconds=1) * 0.123456, td(microseconds=123456))
        # eq(td(seconds=1) * 0.6112295, td(microseconds=611229))

        # Division by int and float
        eq((3 * us) / 2, 2 * us)
        eq((5 * us) / 2, 2 * us)
        eq((-3 * us) / 2.0, -2 * us)
        eq((-5 * us) / 2.0, -2 * us)
        eq((3 * us) / -2, -2 * us)
        eq((5 * us) / -2, -2 * us)
        eq((3 * us) / -2.0, -2 * us)
        eq((5 * us) / -2.0, -2 * us)
        for i in range(-10, 10):
            eq((i * us / 3) // us, round(i / 3))
        for i in range(-10, 10):
            eq((i * us / -3) // us, round(i / -3))

        # Issue #23521
        # eq(td(seconds=1) / (1 / 0.6112295), td(microseconds=611229))

        # Issue #11576
        eq(td(999999999, 86399, 999999) - td(999999999, 86399, 999998), td(0, 0, 1))
        eq(td(999999999, 1, 1) - td(999999999, 1, 0), td(0, 0, 1))

    """
    def test_disallowed_special(self):
        a = timedelta(42)
        NAN = 0. / 0.
        self.assertRaises(ValueError, a.__mul__, NAN)
        self.assertRaises(ValueError, a.__truediv__, NAN)
    """

    def test_basic_attributes(self):
        days, seconds, us = 1, 7, 31
        td = timedelta(days, seconds, us)
        self.assertEqual(td.days, days)
        self.assertEqual(td.seconds, seconds)
        self.assertEqual(td.microseconds, us)

    def test_total_seconds(self):
        td = timedelta(days=365)
        self.assertEqual(td.total_seconds(), 31536000.0)
        for total_seconds in [123456.789012, -123456.789012, 0.123456, 0, 1e6]:
            td = timedelta(seconds=total_seconds)
            self.assertEqual(td.total_seconds(), total_seconds)
        # Issue8644: Test that td.total_seconds() has the same
        # accuracy as td / timedelta(seconds=1).
        for ms in [-1, -2, -123]:
            td = timedelta(microseconds=ms)
            self.assertEqual(td.total_seconds(), td / timedelta(seconds=1))

    def test_carries(self):
        t1 = timedelta(
            days=100,
            weeks=-7,
            hours=-24 * (100 - 49),
            minutes=-3,
            seconds=12,
            microseconds=(3 * 60 - 12) * 1e6 + 1,
        )
        t2 = timedelta(microseconds=1)
        self.assertEqual(t1, t2)

    def test_hash_equality(self):
        t1 = timedelta(
            days=100,
            weeks=-7,
            hours=-24 * (100 - 49),
            minutes=-3,
            seconds=12,
            microseconds=(3 * 60 - 12) * 1000000,
        )
        t2 = timedelta()
        self.assertEqual(hash(t1), hash(t2))

        t1 += timedelta(weeks=7)
        t2 += timedelta(days=7 * 7)
        self.assertEqual(t1, t2)
        self.assertEqual(hash(t1), hash(t2))

        d = {t1: 1}
        d[t2] = 2
        self.assertEqual(len(d), 1)
        self.assertEqual(d[t1], 2)

    def test_compare(self):
        t1 = timedelta(2, 3, 4)
        t2 = timedelta(2, 3, 4)
        self.assertEqual(t1, t2)
        self.assertTrue(t1 <= t2)
        self.assertTrue(t1 >= t2)
        self.assertFalse(t1 != t2)
        self.assertFalse(t1 < t2)
        self.assertFalse(t1 > t2)

        for args in (3, 3, 3), (2, 4, 4), (2, 3, 5):
            t2 = timedelta(*args)  # this is larger than t1
            self.assertTrue(t1 < t2)
            self.assertTrue(t2 > t1)
            self.assertTrue(t1 <= t2)
            self.assertTrue(t2 >= t1)
            self.assertTrue(t1 != t2)
            self.assertTrue(t2 != t1)
            self.assertFalse(t1 == t2)
            self.assertFalse(t2 == t1)
            self.assertFalse(t1 > t2)
            self.assertFalse(t2 < t1)
            self.assertFalse(t1 >= t2)
            self.assertFalse(t2 <= t1)

    def test_str(self):
        td = timedelta
        eq = self.assertEqual

        eq(str(td(1)), "1 day, 0:00:00")
        eq(str(td(-1)), "-1 day, 0:00:00")
        eq(str(td(2)), "2 days, 0:00:00")
        eq(str(td(-2)), "-2 days, 0:00:00")

        eq(str(td(hours=12, minutes=58, seconds=59)), "12:58:59")
        eq(str(td(hours=2, minutes=3, seconds=4)), "2:03:04")
        eq(str(td(weeks=-30, hours=23, minutes=12, seconds=34)), "-210 days, 23:12:34")

        eq(str(td(milliseconds=1)), "0:00:00.001000")
        eq(str(td(microseconds=3)), "0:00:00.000003")

        # Hercules's timedelta has a smaller range than Python's
        # since it uses a pure microseconds representation, so
        # below case is not supported.
        """
        eq(str(td(days=999999999, hours=23, minutes=59, seconds=59,
                   microseconds=999999)),
           "999999999 days, 23:59:59.999999")
        """

    def test_repr(self):
        td = timedelta
        self.assertEqual(repr(td(1)), "timedelta(days=1)")
        self.assertEqual(repr(td(10, 2)), "timedelta(days=10, seconds=2)")
        self.assertEqual(
            repr(td(-10, 2, 400000)),
            "timedelta(days=-10, seconds=2, microseconds=400000)",
        )
        self.assertEqual(repr(td(seconds=60)), "timedelta(seconds=60)")
        self.assertEqual(repr(td()), "timedelta(0)")
        self.assertEqual(repr(td(microseconds=100)), "timedelta(microseconds=100)")
        self.assertEqual(
            repr(td(days=1, microseconds=100)), "timedelta(days=1, microseconds=100)"
        )
        self.assertEqual(
            repr(td(seconds=1, microseconds=100)),
            "timedelta(seconds=1, microseconds=100)",
        )

    def test_resolution_info(self):
        self.assertTrue(timedelta.max > timedelta.min)
        #self.assertEqual(timedelta.min, timedelta(-999999999))
        #self.assertEqual(timedelta.max, timedelta(999999999, 24*3600-1, 1e6-1))
        self.assertEqual(timedelta.resolution, timedelta(0, 0, 1))

    """
    def test_overflow(self):
        tiny = timedelta(microseconds=1)  # timedelta.resolution

        td = timedelta.min + tiny
        td -= tiny  # no problem
        self.assertRaises(OverflowError, td.__sub__, tiny)
        self.assertRaises(OverflowError, td.__add__, -tiny)

        td = timedelta.max - tiny
        td += tiny  # no problem
        self.assertRaises(OverflowError, td.__add__, tiny)
        self.assertRaises(OverflowError, td.__sub__, -tiny)

        self.assertRaises(OverflowError, lambda: -timedelta.max)

        day = timedelta(1)
        self.assertRaises(OverflowError, day.__mul__, 10**9)
        self.assertRaises(OverflowError, day.__mul__, 1e9)
        self.assertRaises(OverflowError, day.__truediv__, 1e-20)
        self.assertRaises(OverflowError, day.__truediv__, 1e-10)
        self.assertRaises(OverflowError, day.__truediv__, 9e-10)

    def _test_overflow_special(self):
        day = timedelta(1)
        INF = 1. / 0.
        self.assertRaises(OverflowError, day.__mul__, INF)
        self.assertRaises(OverflowError, day.__mul__, -INF)
    """

    def test_microsecond_rounding(self):
        td = timedelta
        eq = self.assertEqual

        # Single-field rounding.
        eq(td(milliseconds=0.4 / 1000), td(0))  # rounds to 0
        eq(td(milliseconds=-0.4 / 1000), td(0))  # rounds to 0
        eq(td(milliseconds=0.5 / 1000), td(microseconds=0))
        eq(td(milliseconds=-0.5 / 1000), td(microseconds=-0))
        eq(td(milliseconds=0.6 / 1000), td(microseconds=1))
        eq(td(milliseconds=-0.6 / 1000), td(microseconds=-1))
        eq(td(milliseconds=1.5 / 1000), td(microseconds=2))
        eq(td(milliseconds=-1.5 / 1000), td(microseconds=-2))
        eq(td(seconds=0.5 / 10 ** 6), td(microseconds=0))
        eq(td(seconds=-0.5 / 10 ** 6), td(microseconds=-0))
        eq(td(seconds=1 / 2 ** 7), td(microseconds=7812))
        eq(td(seconds=-1 / 2 ** 7), td(microseconds=-7812))

        # Rounding due to contributions from more than one field.
        us_per_hour = 3600e6
        us_per_day = us_per_hour * 24
        eq(td(days=0.4 / us_per_day), td(0))
        eq(td(hours=0.2 / us_per_hour), td(0))
        eq(td(days=0.4 / us_per_day, hours=0.2 / us_per_hour), td(microseconds=1))

        eq(td(days=-0.4 / us_per_day), td(0))
        eq(td(hours=-0.2 / us_per_hour), td(0))
        eq(td(days=-0.4 / us_per_day, hours=-0.2 / us_per_hour), td(microseconds=-1))

        # Test for a patch in Issue 8860
        eq(td(microseconds=0.5), 0.5 * td(microseconds=1.0))
        resolution = td(microseconds=1)  # td.resolution
        eq(td(microseconds=0.5) // resolution, 0.5 * resolution // resolution)

    def test_massive_normalization(self):
        td = timedelta(microseconds=-1)
        self.assertEqual(
            (td.days, td.seconds, td.microseconds), (-1, 24 * 3600 - 1, 999999)
        )

    def test_bool(self):
        self.assertTrue(timedelta(1))
        self.assertTrue(timedelta(0, 1))
        self.assertTrue(timedelta(0, 0, 1))
        self.assertTrue(timedelta(microseconds=1))
        self.assertFalse(timedelta(0))

    def test_division(self):
        t = timedelta(hours=1, minutes=24, seconds=19)
        second = timedelta(seconds=1)
        self.assertEqual(t / second, 5059.0)
        self.assertEqual(t // second, 5059)

        t = timedelta(minutes=2, seconds=30)
        minute = timedelta(minutes=1)
        self.assertEqual(t / minute, 2.5)
        self.assertEqual(t // minute, 2)

        zerotd = timedelta(0)
        # self.assertRaises(ZeroDivisionError, truediv, t, zerotd)
        # self.assertRaises(ZeroDivisionError, floordiv, t, zerotd)

    def test_remainder(self):
        t = timedelta(minutes=2, seconds=30)
        minute = timedelta(minutes=1)
        r = t % minute
        self.assertEqual(r, timedelta(seconds=30))

        t = timedelta(minutes=-2, seconds=30)
        r = t % minute
        self.assertEqual(r, timedelta(seconds=30))

        zerotd = timedelta(0)
        # self.assertRaises(ZeroDivisionError, mod, t, zerotd)

    def test_divmod(self):
        t = timedelta(minutes=2, seconds=30)
        minute = timedelta(minutes=1)
        q, r = divmod(t, minute)
        self.assertEqual(q, 2)
        self.assertEqual(r, timedelta(seconds=30))

        t = timedelta(minutes=-2, seconds=30)
        q, r = divmod(t, minute)
        self.assertEqual(q, -2)
        self.assertEqual(r, timedelta(seconds=30))

        zerotd = timedelta(0)
        # self.assertRaises(ZeroDivisionError, divmod, t, zerotd)


class TestDateOnly(Static[TestCase]):
    def test_delta_non_days_ignored(self):
        dt = date(2000, 1, 2)
        delta = timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=5)
        days = timedelta(delta.days)
        self.assertEqual(days, timedelta(1))

        dt2 = dt + delta
        self.assertEqual(dt2, dt + days)

        dt2 = delta + dt
        self.assertEqual(dt2, dt + days)

        dt2 = dt - delta
        self.assertEqual(dt2, dt - days)

        delta = -delta
        days = timedelta(delta.days)
        self.assertEqual(days, timedelta(-2))

        dt2 = dt + delta
        self.assertEqual(dt2, dt + days)

        dt2 = delta + dt
        self.assertEqual(dt2, dt + days)

        dt2 = dt - delta
        self.assertEqual(dt2, dt - days)


class TestDate(Static[TestCase]):
    theclass: type

    def test_basic_attributes(self):
        dt = self.theclass(2002, 3, 1)
        self.assertEqual(dt.year, 2002)
        self.assertEqual(dt.month, 3)
        self.assertEqual(dt.day, 1)

    def test_ordinal_conversions(self):
        # Check some fixed values.
        for y, m, d, n in [
            (1, 1, 1, 1),  # calendar origin
            (1, 12, 31, 365),
            (2, 1, 1, 366),
            # first example from "Calendrical Calculations"
            (1945, 11, 12, 710347),
        ]:
            dt = self.theclass(y, m, d)
            self.assertEqual(n, dt.toordinal())
            fromord = self.theclass.fromordinal(n)
            self.assertEqual(dt, fromord)
            if hasattr(fromord, "hour"):
                # if we're checking something fancier than a date, verify
                # the extra fields have been zeroed out
                self.assertEqual(fromord.hour, 0)
                self.assertEqual(fromord.minute, 0)
                self.assertEqual(fromord.second, 0)
                self.assertEqual(fromord.microsecond, 0)

        # Check first and last days of year spottily across the whole
        # range of years supported.
        for year in range(MINYEAR, MAXYEAR + 1, 7):
            # Verify (year, 1, 1) -> ordinal -> y, m, d is identity.
            d = self.theclass(year, 1, 1)
            n = d.toordinal()
            d2 = self.theclass.fromordinal(n)
            self.assertEqual(d, d2)
            # Verify that moving back a day gets to the end of year-1.
            if year > 1:
                d = self.theclass.fromordinal(n - 1)
                d2 = self.theclass(year - 1, 12, 31)
                self.assertEqual(d, d2)
                self.assertEqual(d2.toordinal(), n - 1)

        # Test every day in a leap-year and a non-leap year.
        dim = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        for year, isleap in (2000, True), (2002, False):
            n = self.theclass(year, 1, 1).toordinal()
            for month, maxday in zip(range(1, 13), dim):
                if month == 2 and isleap:
                    maxday += 1
                for day in range(1, maxday + 1):
                    d = self.theclass(year, month, day)
                    self.assertEqual(d.toordinal(), n)
                    self.assertEqual(d, self.theclass.fromordinal(n))
                    n += 1

    """
    def test_extreme_ordinals(self):
        a = self.theclass.min
        a = self.theclass(a.year, a.month, a.day)  # get rid of time parts
        aord = a.toordinal()
        b = a.fromordinal(aord)
        self.assertEqual(a, b)

        self.assertRaises(ValueError, lambda: a.fromordinal(aord - 1))

        b = a + timedelta(days=1)
        self.assertEqual(b.toordinal(), aord + 1)
        self.assertEqual(b, self.theclass.fromordinal(aord + 1))

        a = self.theclass.max
        a = self.theclass(a.year, a.month, a.day)  # get rid of time parts
        aord = a.toordinal()
        b = a.fromordinal(aord)
        self.assertEqual(a, b)

        self.assertRaises(ValueError, lambda: a.fromordinal(aord + 1))

        b = a - timedelta(days=1)
        self.assertEqual(b.toordinal(), aord - 1)
        self.assertEqual(b, self.theclass.fromordinal(aord - 1))
    """

    def test_bad_constructor_arguments(self):
        # bad years
        self.theclass(MINYEAR, 1, 1)  # no exception
        self.theclass(MAXYEAR, 1, 1)  # no exception

        def make(theclass, a, b, c):
            return self.theclass(a, b, c)

        self.assertRaises(ValueError, make(self.theclass, ...), MINYEAR - 1, 1, 1)
        self.assertRaises(ValueError, make(self.theclass, ...), MAXYEAR + 1, 1, 1)
        # bad months
        self.theclass(2000, 1, 1)  # no exception
        self.theclass(2000, 12, 1)  # no exception
        self.assertRaises(ValueError, make(self.theclass, ...), 2000, 0, 1)
        self.assertRaises(ValueError, make(self.theclass, ...), 2000, 13, 1)
        # bad days
        self.theclass(2000, 2, 29)  # no exception
        self.theclass(2004, 2, 29)  # no exception
        self.theclass(2400, 2, 29)  # no exception
        self.assertRaises(ValueError, make(self.theclass, ...), 2000, 2, 30)
        self.assertRaises(ValueError, make(self.theclass, ...), 2001, 2, 29)
        self.assertRaises(ValueError, make(self.theclass, ...), 2100, 2, 29)
        self.assertRaises(ValueError, make(self.theclass, ...), 1900, 2, 29)
        self.assertRaises(ValueError, make(self.theclass, ...), 2000, 1, 0)
        self.assertRaises(ValueError, make(self.theclass, ...), 2000, 1, 32)

    def test_hash_equality(self):
        d = self.theclass(2000, 12, 31)
        # same thing
        e = self.theclass(2000, 12, 31)
        self.assertEqual(d, e)
        self.assertEqual(hash(d), hash(e))

        dic = {d: 1}
        dic[e] = 2
        self.assertEqual(len(dic), 1)
        self.assertEqual(dic[d], 2)
        self.assertEqual(dic[e], 2)

        d = self.theclass(2001, 1, 1)
        # same thing
        e = self.theclass(2001, 1, 1)
        self.assertEqual(d, e)
        self.assertEqual(hash(d), hash(e))

        dic = {d: 1}
        dic[e] = 2
        self.assertEqual(len(dic), 1)
        self.assertEqual(dic[d], 2)
        self.assertEqual(dic[e], 2)

    def test_computations(self):
        a = self.theclass(2002, 1, 31)
        b = self.theclass(1956, 1, 31)
        c = self.theclass(2001, 2, 1)

        diff = a - b
        self.assertEqual(diff.days, 46 * 365 + len(range(1956, 2002, 4)))
        self.assertEqual(diff.seconds, 0)
        self.assertEqual(diff.microseconds, 0)

        day = timedelta(1)
        week = timedelta(7)
        a = self.theclass(2002, 3, 2)
        self.assertEqual(a + day, self.theclass(2002, 3, 3))
        self.assertEqual(day + a, self.theclass(2002, 3, 3))
        self.assertEqual(a - day, self.theclass(2002, 3, 1))
        self.assertEqual(-day + a, self.theclass(2002, 3, 1))
        self.assertEqual(a + week, self.theclass(2002, 3, 9))
        self.assertEqual(a - week, self.theclass(2002, 2, 23))
        self.assertEqual(a + 52 * week, self.theclass(2003, 3, 1))
        self.assertEqual(a - 52 * week, self.theclass(2001, 3, 3))
        self.assertEqual((a + week) - a, week)
        self.assertEqual((a + day) - a, day)
        self.assertEqual((a - week) - a, -week)
        self.assertEqual((a - day) - a, -day)
        self.assertEqual(a - (a + week), -week)
        self.assertEqual(a - (a + day), -day)
        self.assertEqual(a - (a - week), week)
        self.assertEqual(a - (a - day), day)
        self.assertEqual(c - (c - day), day)

        """
        # Add/sub ints or floats should be illegal
        for i in 1, 1.0:
            self.assertRaises(TypeError, lambda: a+i)
            self.assertRaises(TypeError, lambda: a-i)
            self.assertRaises(TypeError, lambda: i+a)
            self.assertRaises(TypeError, lambda: i-a)

        # delta - date is senseless.
        self.assertRaises(TypeError, lambda: day - a)
        # mixing date and (delta or date) via * or // is senseless
        self.assertRaises(TypeError, lambda: day * a)
        self.assertRaises(TypeError, lambda: a * day)
        self.assertRaises(TypeError, lambda: day // a)
        self.assertRaises(TypeError, lambda: a // day)
        self.assertRaises(TypeError, lambda: a * a)
        self.assertRaises(TypeError, lambda: a // a)
        # date + date is senseless
        self.assertRaises(TypeError, lambda: a + a)
        """

    """
    def test_overflow(self):
        tiny = self.theclass.resolution

        for delta in [tiny, timedelta(1), timedelta(2)]:
            dt = self.theclass.min + delta
            dt -= delta  # no problem
            self.assertRaises(OverflowError, dt.__sub__, delta)
            self.assertRaises(OverflowError, dt.__add__, -delta)

            dt = self.theclass.max - delta
            dt += delta  # no problem
            self.assertRaises(OverflowError, dt.__add__, delta)
            self.assertRaises(OverflowError, dt.__sub__, -delta)
    """

    def test_fromtimestamp(self):
        import time

        # Try an arbitrary fixed value.
        year, month, day = 1999, 9, 19
        ts = time.mktime((year, month, day, 0, 0, 0, 0, 0, -1))
        d = self.theclass.fromtimestamp(ts)
        self.assertEqual(d.year, year)
        self.assertEqual(d.month, month)
        self.assertEqual(d.day, day)

    def test_today(self):
        import time

        # We claim that today() is like fromtimestamp(time.time()), so
        # prove it.
        today = self.theclass.today()
        todayagain = today

        for dummy in range(3):
            today = self.theclass.today()
            ts = time.time()
            todayagain = self.theclass.fromtimestamp(ts)
            if today == todayagain:
                break
            # There are several legit reasons that could fail:
            # 1. It recently became midnight, between the today() and the
            #    time() calls.
            # 2. The platform time() has such fine resolution that we'll
            #    never get the same value twice.
            # 3. The platform time() has poor resolution, and we just
            #    happened to call today() right before a resolution quantum
            #    boundary.
            # 4. The system clock got fiddled between calls.
            # In any case, wait a little while and try again.
            time.sleep(0.1)

        # It worked or it didn't.  If it didn't, assume it's reason #2, and
        # let the test pass if they're within half a second of each other.
        if today != todayagain:
            self.assertAlmostEqual(todayagain, today, delta=timedelta(seconds=0.5))

    def test_weekday(self):
        for i in range(7):
            # March 4, 2002 is a Monday
            self.assertEqual(self.theclass(2002, 3, 4 + i).weekday(), i)
            self.assertEqual(self.theclass(2002, 3, 4 + i).isoweekday(), i + 1)
            # January 2, 1956 is a Monday
            self.assertEqual(self.theclass(1956, 1, 2 + i).weekday(), i)
            self.assertEqual(self.theclass(1956, 1, 2 + i).isoweekday(), i + 1)

    def test_isocalendar(self):
        # Check examples from
        # http://www.phys.uu.nl/~vgent/calendar/isocalendar.htm
        week_mondays = [
            ((2003, 12, 22), (2003, 52, 1)),
            ((2003, 12, 29), (2004, 1, 1)),
            ((2004, 1, 5), (2004, 2, 1)),
            ((2009, 12, 21), (2009, 52, 1)),
            ((2009, 12, 28), (2009, 53, 1)),
            ((2010, 1, 4), (2010, 1, 1)),
        ]

        test_cases = []
        for cal_date, iso_date in week_mondays:
            base_date = self.theclass(*cal_date)
            # Adds one test case for every day of the specified weeks
            for i in range(7):
                new_date = base_date + timedelta(i)
                new_iso = iso_date[0:2] + (iso_date[2] + i,)
                test_cases.append((new_date, new_iso))

        for d, exp_iso in test_cases:
            self.assertEqual(d.isocalendar(), exp_iso)

            # Check that the tuple contents are accessible by field name
            t = d.isocalendar()
            self.assertEqual((t.year, t.week, t.weekday), exp_iso)

    def test_iso_long_years(self):
        # Calculate long ISO years and compare to table from
        # http://www.phys.uu.nl/~vgent/calendar/isocalendar.htm
        ISO_LONG_YEARS_TABLE = """
              4   32   60   88
              9   37   65   93
             15   43   71   99
             20   48   76
             26   54   82
            105  133  161  189
            111  139  167  195
            116  144  172
            122  150  178
            128  156  184
            201  229  257  285
            207  235  263  291
            212  240  268  296
            218  246  274
            224  252  280
            303  331  359  387
            308  336  364  392
            314  342  370  398
            320  348  376
            325  353  381
        """
        iso_long_years = sorted(map(int, ISO_LONG_YEARS_TABLE.split()))
        L = []
        for i in range(400):
            d = self.theclass(2000 + i, 12, 31).isocalendar()
            d1 = self.theclass(1600 + i, 12, 31).isocalendar()
            self.assertEqual((d.week, d.weekday), (d1.week, d1.weekday))
            if d.week == 53:
                L.append(i)
        self.assertEqual(L, iso_long_years)

    def test_isoformat(self):
        t = self.theclass(2, 3, 2)
        self.assertEqual(t.isoformat(), "0002-03-02")

    def test_ctime(self):
        t = self.theclass(2002, 3, 2)
        self.assertEqual(t.ctime(), "Sat Mar  2 00:00:00 2002")

    def test_resolution_info(self):
        self.assertTrue(self.theclass.max > self.theclass.min)

    """
    def test_extreme_timedelta(self):
        big = self.theclass.max - self.theclass.min
        # 3652058 days, 23 hours, 59 minutes, 59 seconds, 999999 microseconds
        n = (big.days*24*3600 + big.seconds)*1000000 + big.microseconds
        # n == 315537897599999999 ~= 2**58.13
        justasbig = timedelta(0, 0, n)
        self.assertEqual(big, justasbig)
        self.assertEqual(self.theclass.min + big, self.theclass.max)
        self.assertEqual(self.theclass.max - big, self.theclass.min)
    """

    def test_timetuple(self):
        from time import struct_time

        for i in range(7):
            # January 2, 1956 is a Monday (0)
            d = self.theclass(1956, 1, 2 + i)
            t = d.timetuple()
            self.assertEqual(t, struct_time(1956, 1, 2 + i, 0, 0, 0, i, 2 + i, -1))
            # February 1, 1956 is a Wednesday (2)
            d = self.theclass(1956, 2, 1 + i)
            t = d.timetuple()
            self.assertEqual(
                t, struct_time(1956, 2, 1 + i, 0, 0, 0, (2 + i) % 7, 32 + i, -1)
            )
            # March 1, 1956 is a Thursday (3), and is the 31+29+1 = 61st day
            # of the year.
            d = self.theclass(1956, 3, 1 + i)
            t = d.timetuple()
            self.assertEqual(
                t, struct_time(1956, 3, 1 + i, 0, 0, 0, (3 + i) % 7, 61 + i, -1)
            )
            self.assertEqual(t.tm_year, 1956)
            self.assertEqual(t.tm_mon, 3)
            self.assertEqual(t.tm_mday, 1 + i)
            self.assertEqual(t.tm_hour, 0)
            self.assertEqual(t.tm_min, 0)
            self.assertEqual(t.tm_sec, 0)
            self.assertEqual(t.tm_wday, (3 + i) % 7)
            self.assertEqual(t.tm_yday, 61 + i)
            self.assertEqual(t.tm_isdst, -1)

    def test_compare(self):
        t1 = self.theclass(2, 3, 4)
        t2 = self.theclass(2, 3, 4)
        self.assertEqual(t1, t2)
        self.assertTrue(t1 <= t2)
        self.assertTrue(t1 >= t2)
        self.assertFalse(t1 != t2)
        self.assertFalse(t1 < t2)
        self.assertFalse(t1 > t2)

        for args in (3, 3, 3), (2, 4, 4), (2, 3, 5):
            t2 = self.theclass(*args)  # this is larger than t1
            self.assertTrue(t1 < t2)
            self.assertTrue(t2 > t1)
            self.assertTrue(t1 <= t2)
            self.assertTrue(t2 >= t1)
            self.assertTrue(t1 != t2)
            self.assertTrue(t2 != t1)
            self.assertFalse(t1 == t2)
            self.assertFalse(t2 == t1)
            self.assertFalse(t1 > t2)
            self.assertFalse(t2 < t1)
            self.assertFalse(t1 >= t2)
            self.assertFalse(t2 <= t1)

        """
        for badarg in OTHERSTUFF:
            self.assertEqual(t1 == badarg, False)
            self.assertEqual(t1 != badarg, True)
            self.assertEqual(badarg == t1, False)
            self.assertEqual(badarg != t1, True)

            self.assertRaises(TypeError, lambda: t1 < badarg)
            self.assertRaises(TypeError, lambda: t1 > badarg)
            self.assertRaises(TypeError, lambda: t1 >= badarg)
            self.assertRaises(TypeError, lambda: badarg <= t1)
            self.assertRaises(TypeError, lambda: badarg < t1)
            self.assertRaises(TypeError, lambda: badarg > t1)
            self.assertRaises(TypeError, lambda: badarg >= t1)
        """

    def test_bool(self):
        # All dates are considered true.
        # self.assertTrue(self.theclass.min)
        # self.assertTrue(self.theclass.max)
        self.assertTrue(self.theclass(1, 1, 1))

    def test_replace(self):
        cls = self.theclass
        args = (1, 2, 3)
        base = cls(*args)
        self.assertEqual(base, base.replace())
        self.assertEqual(base.replace(year=2), cls(2, 2, 3))
        self.assertEqual(base.replace(month=3), cls(1, 3, 3))
        self.assertEqual(base.replace(day=4), cls(1, 2, 4))

        # Out of bounds.
        base = cls(2000, 2, 29)
        self.assertRaises(ValueError, base.replace, year=2001)

    def test_fromisoformat(self):
        # Test that isoformat() is reversible
        base_dates = [
            (1, 1, 1),
            (1000, 2, 14),
            (1900, 1, 1),
            (2000, 2, 29),
            (2004, 11, 12),
            (2004, 4, 3),
            (2017, 5, 30),
        ]

        for dt_tuple in base_dates:
            dt = self.theclass(*dt_tuple)
            dt_str = dt.isoformat()
            dt_rt = self.theclass.fromisoformat(dt.isoformat())
            self.assertEqual(dt, dt_rt)

    def test_fromisoformat_fails(self):
        # Test that fromisoformat() fails on invalid values
        bad_strs = [
            "",  # Empty string
            "\ud800",  # bpo-34454: Surrogate code point
            "009-03-04",  # Not 10 characters
            "123456789",  # Not a date
            "200a-12-04",  # Invalid character in year
            "2009-1a-04",  # Invalid character in month
            "2009-12-0a",  # Invalid character in day
            "2009-01-32",  # Invalid day
            "2009-02-29",  # Invalid leap day
            "20090228",  # Valid ISO8601 output not from isoformat()
            "2009\ud80002\ud80028",  # Separators are surrogate codepoints
        ]

        for bad_str in bad_strs:
            self.assertRaises(ValueError, self.theclass.fromisoformat, bad_str)

    def test_fromisocalendar(self):
        # For each test case, assert that fromisocalendar is the
        # inverse of the isocalendar function
        dates = [
            (2016, 4, 3),
            (2005, 1, 2),  # (2004, 53, 7)
            (2008, 12, 30),  # (2009, 1, 2)
            (2010, 1, 2),  # (2009, 53, 6)
            (2009, 12, 31),  # (2009, 53, 4)
            (1900, 1, 1),  # Unusual non-leap year (year % 100 == 0)
            (1900, 12, 31),
            (2000, 1, 1),  # Unusual leap year (year % 400 == 0)
            (2000, 12, 31),
            (2004, 1, 1),  # Leap year
            (2004, 12, 31),
            (1, 1, 1),
            (9999, 12, 31),
            (MINYEAR, 1, 1),
            (MAXYEAR, 12, 31),
        ]

        for datecomps in dates:
            dobj = self.theclass(*datecomps)
            isocal = dobj.isocalendar()

            d_roundtrip = self.theclass.fromisocalendar(*isocal)

            self.assertEqual(dobj, d_roundtrip)

    def test_fromisocalendar_value_errors(self):
        isocals = [
            (2019, 0, 1),
            (2019, -1, 1),
            (2019, 54, 1),
            (2019, 1, 0),
            (2019, 1, -1),
            (2019, 1, 8),
            (2019, 53, 1),
            (10000, 1, 1),
            (0, 1, 1),
            (9999999, 1, 1),
            (2 << 32, 1, 1),
            (2019, 2 << 32, 1),
            (2019, 1, 2 << 32),
        ]

        for isocal in isocals:
            self.assertRaises(ValueError, self.theclass.fromisocalendar, *isocal)


class TestDateTime(Static[TestDate[theclass]]):
    theclass: type

    def test_basic_attributes(self):
        dt = self.theclass(2002, 3, 1, 12, 0)
        self.assertEqual(dt.year, 2002)
        self.assertEqual(dt.month, 3)
        self.assertEqual(dt.day, 1)
        self.assertEqual(dt.hour, 12)
        self.assertEqual(dt.minute, 0)
        self.assertEqual(dt.second, 0)
        self.assertEqual(dt.microsecond, 0)

    def test_basic_attributes_nonzero(self):
        # Make sure all attributes are non-zero so bugs in
        # bit-shifting access show up.
        dt = self.theclass(2002, 3, 1, 12, 59, 59, 8000)
        self.assertEqual(dt.year, 2002)
        self.assertEqual(dt.month, 3)
        self.assertEqual(dt.day, 1)
        self.assertEqual(dt.hour, 12)
        self.assertEqual(dt.minute, 59)
        self.assertEqual(dt.second, 59)
        self.assertEqual(dt.microsecond, 8000)

    def test_isoformat(self):
        t = self.theclass(1, 2, 3, 4, 5, 1, 123)
        self.assertEqual(t.isoformat(), "0001-02-03T04:05:01.000123")
        self.assertEqual(t.isoformat("T"), "0001-02-03T04:05:01.000123")
        self.assertEqual(t.isoformat(" "), "0001-02-03 04:05:01.000123")
        self.assertEqual(t.isoformat("\x00"), "0001-02-03\x0004:05:01.000123")
        # bpo-34482: Check that surrogates are handled properly.
        self.assertEqual(t.isoformat("\ud800"), "0001-02-03\ud80004:05:01.000123")
        self.assertEqual(t.isoformat(timespec="hours"), "0001-02-03T04")
        self.assertEqual(t.isoformat(timespec="minutes"), "0001-02-03T04:05")
        self.assertEqual(t.isoformat(timespec="seconds"), "0001-02-03T04:05:01")
        self.assertEqual(
            t.isoformat(timespec="milliseconds"), "0001-02-03T04:05:01.000"
        )
        self.assertEqual(
            t.isoformat(timespec="microseconds"), "0001-02-03T04:05:01.000123"
        )
        self.assertEqual(t.isoformat(timespec="auto"), "0001-02-03T04:05:01.000123")
        self.assertEqual(t.isoformat(sep=" ", timespec="minutes"), "0001-02-03 04:05")
        # str is ISO format with the separator forced to a blank.
        self.assertEqual(str(t), "0001-02-03 04:05:01.000123")

        # t = self.theclass(1, 2, 3, 4, 5, 1, 999500, tzinfo=timezone.utc)
        # self.assertEqual(t.isoformat(timespec='milliseconds'), "0001-02-03T04:05:01.999+00:00")

        t = self.theclass(1, 2, 3, 4, 5, 1, 999500)
        self.assertEqual(
            t.isoformat(timespec="milliseconds"), "0001-02-03T04:05:01.999"
        )

        t = self.theclass(1, 2, 3, 4, 5, 1)
        self.assertEqual(t.isoformat(timespec="auto"), "0001-02-03T04:05:01")
        self.assertEqual(
            t.isoformat(timespec="milliseconds"), "0001-02-03T04:05:01.000"
        )
        self.assertEqual(
            t.isoformat(timespec="microseconds"), "0001-02-03T04:05:01.000000"
        )

        t = self.theclass(2, 3, 2)
        self.assertEqual(t.isoformat(), "0002-03-02T00:00:00")
        self.assertEqual(t.isoformat("T"), "0002-03-02T00:00:00")
        self.assertEqual(t.isoformat(" "), "0002-03-02 00:00:00")
        # str is ISO format with the separator forced to a blank.
        self.assertEqual(str(t), "0002-03-02 00:00:00")
        # ISO format with timezone
        # tz = FixedOffset(timedelta(seconds=16), 'XXX')
        # t = self.theclass(2, 3, 2, tzinfo=tz)
        # self.assertEqual(t.isoformat(), "0002-03-02T00:00:00+00:00:16")

    """
    def test_isoformat_timezone(self):
        tzoffsets = [
            ('05:00', timedelta(hours=5)),
            ('02:00', timedelta(hours=2)),
            ('06:27', timedelta(hours=6, minutes=27)),
            ('12:32:30', timedelta(hours=12, minutes=32, seconds=30)),
            ('02:04:09.123456', timedelta(hours=2, minutes=4, seconds=9, microseconds=123456))
        ]

        tzinfos = [
            ('', None),
            ('+00:00', timezone.utc),
            ('+00:00', timezone(timedelta(0))),
        ]

        tzinfos += [
            (prefix + expected, timezone(sign * td))
            for expected, td in tzoffsets
            for prefix, sign in [('-', -1), ('+', 1)]
        ]

        dt_base = self.theclass(2016, 4, 1, 12, 37, 9)
        exp_base = '2016-04-01T12:37:09'

        for exp_tz, tzi in tzinfos:
            dt = dt_base.replace(tzinfo=tzi)
            exp = exp_base + exp_tz
            with self.subTest(tzi=tzi):
                assert dt.isoformat() == exp
    """

    """
    def test_more_ctime(self):
        # Test fields that TestDate doesn't touch.
        import time

        t = self.theclass(2002, 3, 2, 18, 3, 5, 123)
        self.assertEqual(t.ctime(), "Sat Mar  2 18:03:05 2002")
        # Oops!  The next line fails on Win2K under MSVC 6, so it's commented
        # out.  The difference is that t.ctime() produces " 2" for the day,
        # but platform ctime() produces "02" for the day.  According to
        # C99, t.ctime() is correct here.
        # self.assertEqual(t.ctime(), time.ctime(time.mktime(t.timetuple())))

        # So test a case where that difference doesn't matter.
        t = self.theclass(2002, 3, 22, 18, 3, 5, 123)
        self.assertEqual(t.ctime(), time.ctime(time.mktime(t.timetuple())))
    """

    def test_tz_independent_comparing(self):
        dt1 = self.theclass(2002, 3, 1, 9, 0, 0)
        dt2 = self.theclass(2002, 3, 1, 10, 0, 0)
        dt3 = self.theclass(2002, 3, 1, 9, 0, 0)
        self.assertEqual(dt1, dt3)
        self.assertTrue(dt2 > dt3)

        # Make sure comparison doesn't forget microseconds, and isn't done
        # via comparing a float timestamp (an IEEE double doesn't have enough
        # precision to span microsecond resolution across years 1 through 9999,
        # so comparing via timestamp necessarily calls some distinct values
        # equal).
        dt1 = self.theclass(MAXYEAR, 12, 31, 23, 59, 59, 999998)
        us = timedelta(microseconds=1)
        dt2 = dt1 + us
        self.assertEqual(dt2 - dt1, us)
        self.assertTrue(dt1 < dt2)

    def test_bad_constructor_arguments(self):
        # bad years
        self.theclass(MINYEAR, 1, 1)  # no exception
        self.theclass(MAXYEAR, 1, 1)  # no exception

        make_dt1 = lambda a, b, c: self.theclass(a, b, c)
        make_dt2 = lambda a, b, c, d: self.theclass(a, b, c, d)
        make_dt3 = lambda a, b, c, d, e: self.theclass(a, b, c, d, e)
        make_dt4 = lambda a, b, c, d, e, f: self.theclass(a, b, c, d, e, f)
        make_dt5 = lambda a, b, c, d, e, f, g: self.theclass(a, b, c, d, e, f, g)

        self.assertRaises(ValueError, make_dt1, MINYEAR - 1, 1, 1)
        self.assertRaises(ValueError, make_dt1, MAXYEAR + 1, 1, 1)
        # bad months
        self.theclass(2000, 1, 1)  # no exception
        self.theclass(2000, 12, 1)  # no exception
        self.assertRaises(ValueError, make_dt1, 2000, 0, 1)
        self.assertRaises(ValueError, make_dt1, 2000, 13, 1)
        # bad days
        self.theclass(2000, 2, 29)  # no exception
        self.theclass(2004, 2, 29)  # no exception
        self.theclass(2400, 2, 29)  # no exception
        self.assertRaises(ValueError, make_dt1, 2000, 2, 30)
        self.assertRaises(ValueError, make_dt1, 2001, 2, 29)
        self.assertRaises(ValueError, make_dt1, 2100, 2, 29)
        self.assertRaises(ValueError, make_dt1, 1900, 2, 29)
        self.assertRaises(ValueError, make_dt1, 2000, 1, 0)
        self.assertRaises(ValueError, make_dt1, 2000, 1, 32)
        # bad hours
        self.theclass(2000, 1, 31, 0)  # no exception
        self.theclass(2000, 1, 31, 23)  # no exception
        self.assertRaises(ValueError, make_dt2, 2000, 1, 31, -1)
        self.assertRaises(ValueError, make_dt2, 2000, 1, 31, 24)
        # bad minutes
        self.theclass(2000, 1, 31, 23, 0)  # no exception
        self.theclass(2000, 1, 31, 23, 59)  # no exception
        self.assertRaises(ValueError, make_dt3, 2000, 1, 31, 23, -1)
        self.assertRaises(ValueError, make_dt3, 2000, 1, 31, 23, 60)
        # bad seconds
        self.theclass(2000, 1, 31, 23, 59, 0)  # no exception
        self.theclass(2000, 1, 31, 23, 59, 59)  # no exception
        self.assertRaises(ValueError, make_dt4, 2000, 1, 31, 23, 59, -1)
        self.assertRaises(ValueError, make_dt4, 2000, 1, 31, 23, 59, 60)
        # bad microseconds
        self.theclass(2000, 1, 31, 23, 59, 59, 0)  # no exception
        self.theclass(2000, 1, 31, 23, 59, 59, 999999)  # no exception
        self.assertRaises(ValueError, make_dt5, 2000, 1, 31, 23, 59, 59, -1)
        self.assertRaises(ValueError, make_dt5, 2000, 1, 31, 23, 59, 59, 1000000)
        """
        # bad fold
        self.assertRaises(ValueError, self.theclass,
                          2000, 1, 31, fold=-1)
        self.assertRaises(ValueError, self.theclass,
                          2000, 1, 31, fold=2)
        # Positional fold:
        self.assertRaises(TypeError, self.theclass,
                          2000, 1, 31, 23, 59, 59, 0, None, 1)
        """

    def test_hash_equality(self):
        d = self.theclass(2000, 12, 31, 23, 30, 17)
        e = self.theclass(2000, 12, 31, 23, 30, 17)
        self.assertEqual(d, e)
        self.assertEqual(hash(d), hash(e))

        dic = {d: 1}
        dic[e] = 2
        self.assertEqual(len(dic), 1)
        self.assertEqual(dic[d], 2)
        self.assertEqual(dic[e], 2)

        d = self.theclass(2001, 1, 1, 0, 5, 17)
        e = self.theclass(2001, 1, 1, 0, 5, 17)
        self.assertEqual(d, e)
        self.assertEqual(hash(d), hash(e))

        dic = {d: 1}
        dic[e] = 2
        self.assertEqual(len(dic), 1)
        self.assertEqual(dic[d], 2)
        self.assertEqual(dic[e], 2)

    def test_computations(self):
        a = self.theclass(2002, 1, 31)
        b = self.theclass(1956, 1, 31)
        diff = a - b
        self.assertEqual(diff.days, 46 * 365 + len(range(1956, 2002, 4)))
        self.assertEqual(diff.seconds, 0)
        self.assertEqual(diff.microseconds, 0)
        a = self.theclass(2002, 3, 2, 17, 6)
        millisec = timedelta(0, 0, 1000)
        hour = timedelta(0, 3600)
        day = timedelta(1)
        week = timedelta(7)
        self.assertEqual(a + hour, self.theclass(2002, 3, 2, 18, 6))
        self.assertEqual(hour + a, self.theclass(2002, 3, 2, 18, 6))
        self.assertEqual(a + 10 * hour, self.theclass(2002, 3, 3, 3, 6))
        self.assertEqual(a - hour, self.theclass(2002, 3, 2, 16, 6))
        self.assertEqual(-hour + a, self.theclass(2002, 3, 2, 16, 6))
        self.assertEqual(a - hour, a + -hour)
        self.assertEqual(a - 20 * hour, self.theclass(2002, 3, 1, 21, 6))
        self.assertEqual(a + day, self.theclass(2002, 3, 3, 17, 6))
        self.assertEqual(a - day, self.theclass(2002, 3, 1, 17, 6))
        self.assertEqual(a + week, self.theclass(2002, 3, 9, 17, 6))
        self.assertEqual(a - week, self.theclass(2002, 2, 23, 17, 6))
        self.assertEqual(a + 52 * week, self.theclass(2003, 3, 1, 17, 6))
        self.assertEqual(a - 52 * week, self.theclass(2001, 3, 3, 17, 6))
        self.assertEqual((a + week) - a, week)
        self.assertEqual((a + day) - a, day)
        self.assertEqual((a + hour) - a, hour)
        self.assertEqual((a + millisec) - a, millisec)
        self.assertEqual((a - week) - a, -week)
        self.assertEqual((a - day) - a, -day)
        self.assertEqual((a - hour) - a, -hour)
        self.assertEqual((a - millisec) - a, -millisec)
        self.assertEqual(a - (a + week), -week)
        self.assertEqual(a - (a + day), -day)
        self.assertEqual(a - (a + hour), -hour)
        self.assertEqual(a - (a + millisec), -millisec)
        self.assertEqual(a - (a - week), week)
        self.assertEqual(a - (a - day), day)
        self.assertEqual(a - (a - hour), hour)
        self.assertEqual(a - (a - millisec), millisec)
        self.assertEqual(
            a + (week + day + hour + millisec),
            self.theclass(2002, 3, 10, 18, 6, 0, 1000),
        )
        self.assertEqual(
            a + (week + day + hour + millisec), (((a + week) + day) + hour) + millisec
        )
        self.assertEqual(
            a - (week + day + hour + millisec),
            self.theclass(2002, 2, 22, 16, 5, 59, 999000),
        )
        self.assertEqual(
            a - (week + day + hour + millisec), (((a - week) - day) - hour) - millisec
        )
        """
        # Add/sub ints or floats should be illegal
        for i in 1, 1.0:
            self.assertRaises(TypeError, lambda: a+i)
            self.assertRaises(TypeError, lambda: a-i)
            self.assertRaises(TypeError, lambda: i+a)
            self.assertRaises(TypeError, lambda: i-a)

        # delta - datetime is senseless.
        self.assertRaises(TypeError, lambda: day - a)
        # mixing datetime and (delta or datetime) via * or // is senseless
        self.assertRaises(TypeError, lambda: day * a)
        self.assertRaises(TypeError, lambda: a * day)
        self.assertRaises(TypeError, lambda: day // a)
        self.assertRaises(TypeError, lambda: a // day)
        self.assertRaises(TypeError, lambda: a * a)
        self.assertRaises(TypeError, lambda: a // a)
        # datetime + datetime is senseless
        self.assertRaises(TypeError, lambda: a + a)
        """

    def test_more_compare(self):
        # The test_compare() inherited from TestDate covers the error cases.
        # We just want to test lexicographic ordering on the members datetime
        # has that date lacks.
        args = (2000, 11, 29, 20, 58, 16, 999998)
        newargsx = [
            (2001, 11, 29, 20, 58, 16, 999998),
            (2000, 12, 29, 20, 58, 16, 999998),
            (2000, 11, 30, 20, 58, 16, 999998),
            (2000, 11, 29, 21, 58, 16, 999998),
            (2000, 11, 29, 20, 59, 16, 999998),
            (2000, 11, 29, 20, 58, 17, 999998),
            (2000, 11, 29, 20, 58, 16, 999999),
        ]
        t1 = self.theclass(*args)
        t2 = self.theclass(*args)
        self.assertEqual(t1, t2)
        self.assertTrue(t1 <= t2)
        self.assertTrue(t1 >= t2)
        self.assertFalse(t1 != t2)
        self.assertFalse(t1 < t2)
        self.assertFalse(t1 > t2)

        for i in range(len(args)):
            newargs = newargsx[i]
            t2 = self.theclass(*newargs)  # this is larger than t1
            self.assertTrue(t1 < t2)
            self.assertTrue(t2 > t1)
            self.assertTrue(t1 <= t2)
            self.assertTrue(t2 >= t1)
            self.assertTrue(t1 != t2)
            self.assertTrue(t2 != t1)
            self.assertFalse(t1 == t2)
            self.assertFalse(t2 == t1)
            self.assertFalse(t1 > t2)
            self.assertFalse(t2 < t1)
            self.assertFalse(t1 >= t2)
            self.assertFalse(t2 <= t1)

    # A helper for timestamp constructor tests.
    def verify_field_equality(self, expected, got):
        self.assertEqual(expected.tm_year, got.year)
        self.assertEqual(expected.tm_mon, got.month)
        self.assertEqual(expected.tm_mday, got.day)
        self.assertEqual(expected.tm_hour, got.hour)
        self.assertEqual(expected.tm_min, got.minute)
        self.assertEqual(expected.tm_sec, got.second)

    def test_fromtimestamp(self):
        import time

        ts = time.time()
        expected = time.localtime(int(ts))
        got = self.theclass.fromtimestamp(ts)
        self.verify_field_equality(expected, got)

    """
    def test_utcfromtimestamp(self):
        import time

        ts = time.time()
        expected = time.gmtime(int(ts))
        got = self.theclass.utcfromtimestamp(ts)
        self.verify_field_equality(expected, got)
    """

    def test_timestamp_naive(self):
        """
        t = self.theclass(1970, 1, 1)
        self.assertEqual(t.timestamp(), 18000.0)
        t = self.theclass(1970, 1, 1, 1, 2, 3, 4)
        self.assertEqual(t.timestamp(),
                         18000.0 + 3600 + 2*60 + 3 + 4*1e-6)
        """

        t = self.theclass(1970, 1, 1)
        self.assertEqual(t.timestamp(), 0)
        t = self.theclass(1970, 1, 1, 1, 2, 3, 4)
        self.assertEqual(t.timestamp(), 3600 + 2 * 60 + 3 + 4 * 1e-6)

        """
        # Missing hour
        t0 = self.theclass(2012, 3, 11, 2, 30)
        t1 = t0.replace(fold=1)
        self.assertEqual(self.theclass.fromtimestamp(t1.timestamp()),
                         t0 - timedelta(hours=1))
        self.assertEqual(self.theclass.fromtimestamp(t0.timestamp()),
                         t1 + timedelta(hours=1))
        # Ambiguous hour defaults to DST
        t = self.theclass(2012, 11, 4, 1, 30)
        self.assertEqual(self.theclass.fromtimestamp(t.timestamp()), t)

        # Timestamp may raise an overflow error on some platforms
        # XXX: Do we care to support the first and last year?
        for t in [self.theclass(2,1,1), self.theclass(9998,12,12)]:
            s = None
            try:
                s = t.timestamp()
            except OverflowError:
                continue

            self.assertEqual(self.theclass.fromtimestamp(s), t)
        """

    """
    def test_timestamp_aware(self):
        t = self.theclass(1970, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(t.timestamp(), 0.0)
        t = self.theclass(1970, 1, 1, 1, 2, 3, 4, tzinfo=timezone.utc)
        self.assertEqual(t.timestamp(),
                         3600 + 2*60 + 3 + 4*1e-6)
        t = self.theclass(1970, 1, 1, 1, 2, 3, 4,
                          tzinfo=timezone(timedelta(hours=-5), 'EST'))
        self.assertEqual(t.timestamp(),
                         18000 + 3600 + 2*60 + 3 + 4*1e-6)
    """
    """
    def test_microsecond_rounding(self):
        for fts in (self.theclass.fromtimestamp,
                    self.theclass.utcfromtimestamp):
            zero = fts(0)
            self.assertEqual(zero.second, 0)
            self.assertEqual(zero.microsecond, 0)
            one = fts(1e-6)
            minus_one = fts(-1e-6)

            self.assertEqual(minus_one.second, 59)
            self.assertEqual(minus_one.microsecond, 999999)

            t = fts(-1e-8)
            self.assertEqual(t, zero)
            t = fts(-9e-7)
            self.assertEqual(t, minus_one)
            t = fts(-1e-7)
            self.assertEqual(t, zero)
            t = fts(-1/2**7)
            self.assertEqual(t.second, 59)
            self.assertEqual(t.microsecond, 992188)

            t = fts(1e-7)
            self.assertEqual(t, zero)
            t = fts(9e-7)
            self.assertEqual(t, one)
            t = fts(0.99999949)
            self.assertEqual(t.second, 0)
            self.assertEqual(t.microsecond, 999999)
            t = fts(0.9999999)
            self.assertEqual(t.second, 1)
            self.assertEqual(t.microsecond, 0)
            t = fts(1/2**7)
            self.assertEqual(t.second, 0)
            self.assertEqual(t.microsecond, 7812)
    """
    """
    def test_timestamp_limits(self):
        # minimum timestamp
        min_dt = self.theclass.min.replace(tzinfo=timezone.utc)
        min_ts = min_dt.timestamp()
        try:
            # date 0001-01-01 00:00:00+00:00: timestamp=-62135596800
            self.assertEqual(self.theclass.fromtimestamp(min_ts, tz=timezone.utc),
                             min_dt)
        except (OverflowError, OSError) as exc:
            # the date 0001-01-01 doesn't fit into 32-bit time_t,
            # or platform doesn't support such very old date
            self.skipTest(str(exc))

        # maximum timestamp: set seconds to zero to avoid rounding issues
        max_dt = self.theclass.max.replace(tzinfo=timezone.utc,
                                           second=0, microsecond=0)
        max_ts = max_dt.timestamp()
        # date 9999-12-31 23:59:00+00:00: timestamp 253402300740
        self.assertEqual(self.theclass.fromtimestamp(max_ts, tz=timezone.utc),
                         max_dt)

        # number of seconds greater than 1 year: make sure that the new date
        # is not valid in datetime.datetime limits
        delta = 3600 * 24 * 400

        # too small
        ts = min_ts - delta
        # converting a Python int to C time_t can raise a OverflowError,
        # especially on 32-bit platforms.
        self.assertRaises(ValueError, self.theclass.fromtimestamp, ts)
        self.assertRaises(ValueError, self.theclass.utcfromtimestamp, ts)

        # too big
        ts = max_dt.timestamp() + delta
        self.assertRaises(ValueError, self.theclass.fromtimestamp, ts)
        self.assertRaises(ValueError, self.theclass.utcfromtimestamp, ts)
    """

    def test_negative_float_fromtimestamp(self):
        # The result is tz-dependent; at least test that this doesn't
        # fail (like it did before bug 1646728 was fixed).
        self.theclass.fromtimestamp(-1.05)

    """
    def test_negative_float_utcfromtimestamp(self):
        d = self.theclass.utcfromtimestamp(-1.05)
        self.assertEqual(d, self.theclass(1969, 12, 31, 23, 59, 58, 950000))
    """

    def test_utcnow(self):
        import time

        # Call it a success if utcnow() and utcfromtimestamp() are within
        # a second of each other.
        from_timestamp = None
        from_now = None
        tolerance = timedelta(seconds=1)
        for dummy in range(3):
            from_now = self.theclass.utcnow()
            from_timestamp = self.theclass.utcfromtimestamp(time.time())
            if abs(from_timestamp.__val__() - from_now.__val__()) <= tolerance:
                break
            # Else try again a few times.
        self.assertLessEqual(abs(from_timestamp.__val__() - from_now.__val__()), tolerance)

    def test_extract(self):
        dt = self.theclass(2002, 3, 4, 18, 45, 3, 1234)
        self.assertEqual(dt.date(), date(2002, 3, 4))
        self.assertEqual(dt.time(), time(18, 45, 3, 1234))

    def test_combine(self):
        d = date(2002, 3, 4)
        t = time(18, 45, 3, 1234)
        expected = self.theclass(2002, 3, 4, 18, 45, 3, 1234)
        combine = self.theclass.combine
        dt = combine(d, t)
        self.assertEqual(dt, expected)

        dt = combine(time=t, date=d)
        self.assertEqual(dt, expected)

        self.assertEqual(d, dt.date())
        self.assertEqual(t, dt.time())
        self.assertEqual(dt, combine(dt.date(), dt.time()))

        """
        self.assertRaises(TypeError, combine) # need an arg
        self.assertRaises(TypeError, combine, d) # need two args
        self.assertRaises(TypeError, combine, t, d) # args reversed
        self.assertRaises(TypeError, combine, d, t, 1) # wrong tzinfo type
        self.assertRaises(TypeError, combine, d, t, 1, 2)  # too many args
        self.assertRaises(TypeError, combine, "date", "time") # wrong types
        self.assertRaises(TypeError, combine, d, "time") # wrong type
        self.assertRaises(TypeError, combine, "date", t) # wrong type

        # tzinfo= argument
        dt = combine(d, t, timezone.utc)
        self.assertIs(dt.tzinfo, timezone.utc)
        dt = combine(d, t, tzinfo=timezone.utc)
        self.assertIs(dt.tzinfo, timezone.utc)
        t = time()
        dt = combine(dt, t)
        self.assertEqual(dt.date(), d)
        self.assertEqual(dt.time(), t)
        """

    def test_replace(self):
        cls = self.theclass
        args = (1, 2, 3, 4, 5, 6, 7)
        base = cls(*args)
        self.assertEqual(base, base.replace())

        self.assertEqual(base.replace(year=2), cls(2, 2, 3, 4, 5, 6, 7))
        self.assertEqual(base.replace(month=3), cls(1, 3, 3, 4, 5, 6, 7))
        self.assertEqual(base.replace(day=4), cls(1, 2, 4, 4, 5, 6, 7))
        self.assertEqual(base.replace(hour=5), cls(1, 2, 3, 5, 5, 6, 7))
        self.assertEqual(base.replace(minute=6), cls(1, 2, 3, 4, 6, 6, 7))
        self.assertEqual(base.replace(second=7), cls(1, 2, 3, 4, 5, 7, 7))
        self.assertEqual(base.replace(microsecond=8), cls(1, 2, 3, 4, 5, 6, 8))
        # Out of bounds.
        base = cls(2000, 2, 29)
        self.assertRaises(ValueError, lambda: base.replace(year=2001))

    """
    def test_astimezone(self):
        dt = self.theclass.now()
        f = FixedOffset(44, "0044")
        dt_utc = dt.replace(tzinfo=timezone(timedelta(hours=-4), 'EDT'))
        self.assertEqual(dt.astimezone(), dt_utc) # naive
        self.assertRaises(TypeError, dt.astimezone, f, f) # too many args
        self.assertRaises(TypeError, dt.astimezone, dt) # arg wrong type
        dt_f = dt.replace(tzinfo=f) + timedelta(hours=4, minutes=44)
        self.assertEqual(dt.astimezone(f), dt_f) # naive
        self.assertEqual(dt.astimezone(tz=f), dt_f) # naive

        class Bogus(tzinfo):
            def utcoffset(self, dt): return None
            def dst(self, dt): return timedelta(0)
        bog = Bogus()
        self.assertRaises(ValueError, dt.astimezone, bog)   # naive
        self.assertEqual(dt.replace(tzinfo=bog).astimezone(f), dt_f)

        class AlsoBogus(tzinfo):
            def utcoffset(self, dt): return timedelta(0)
            def dst(self, dt): return None
        alsobog = AlsoBogus()
        self.assertRaises(ValueError, dt.astimezone, alsobog) # also naive

        class Broken(tzinfo):
            def utcoffset(self, dt): return 1
            def dst(self, dt): return 1
        broken = Broken()
        dt_broken = dt.replace(tzinfo=broken)
        with self.assertRaises(TypeError):
            dt_broken.astimezone()
    """

    def test_fromisoformat_datetime(self):
        # Test that isoformat() is reversible
        base_dates = [(1, 1, 1), (1900, 1, 1), (2004, 11, 12), (2017, 5, 30)]

        base_times = [
            (0, 0, 0, 0),
            (0, 0, 0, 241000),
            (0, 0, 0, 234567),
            (12, 30, 45, 234567),
        ]

        separators = [" ", "T"]

        dts = [
            self.theclass(*date_tuple, *time_tuple)
            for date_tuple in base_dates
            for time_tuple in base_times
        ]

        for dt in dts:
            for sep in separators:
                dtstr = dt.isoformat(sep=sep)
                dt_rt = self.theclass.fromisoformat(dtstr)
                self.assertEqual(dt, dt_rt)

    def test_fromisoformat_separators(self):
        separators = [" ", "T"]

        for sep in separators:
            dt = self.theclass(2018, 1, 31, 23, 59, 47, 124789)
            dtstr = dt.isoformat(sep=sep)

            dt_rt = self.theclass.fromisoformat(dtstr)
            self.assertEqual(dt, dt_rt)

    def test_fromisoformat_ambiguous(self):
        # Test strings like 2018-01-31+12:15 (where +12:15 is not a time zone)
        separators = ["+", "-"]
        for sep in separators:
            dt = self.theclass(2018, 1, 31, 12, 15)
            dtstr = dt.isoformat(sep=sep)

            dt_rt = self.theclass.fromisoformat(dtstr)
            self.assertEqual(dt, dt_rt)

    def test_fromisoformat_timespecs(self):
        datetime_bases = [(2009, 12, 4, 8, 17, 45, 123456), (2009, 12, 4, 8, 17, 45, 0)]

        timespecs = ["hours", "minutes", "seconds", "milliseconds", "microseconds"]

        for dt_tuple in datetime_bases:
            dt = self.theclass(*(dt_tuple[0:4]))
            dtstr = dt.isoformat(timespec="hours")
            dt_rt = self.theclass.fromisoformat(dtstr)
            self.assertEqual(dt, dt_rt)

            dt = self.theclass(*(dt_tuple[0:5]))
            dtstr = dt.isoformat(timespec="minutes")
            dt_rt = self.theclass.fromisoformat(dtstr)
            self.assertEqual(dt, dt_rt)

            dt = self.theclass(*(dt_tuple[0:6]))
            dtstr = dt.isoformat(timespec="seconds")
            dt_rt = self.theclass.fromisoformat(dtstr)
            self.assertEqual(dt, dt_rt)

            new_microseconds = 1000 * (dt_tuple[6] // 1000)
            dt_tuple2 = (*dt_tuple[0:6], new_microseconds)
            dt = self.theclass(*(dt_tuple2[0:7]))
            dtstr = dt.isoformat(timespec="milliseconds")
            dt_rt = self.theclass.fromisoformat(dtstr)
            self.assertEqual(dt, dt_rt)

            dt = self.theclass(*(dt_tuple[0:8]))
            dtstr = dt.isoformat(timespec="microseconds")
            dt_rt = self.theclass.fromisoformat(dtstr)
            self.assertEqual(dt, dt_rt)

    def test_fromisoformat_fails_datetime(self):
        # Test that fromisoformat() fails on invalid values
        bad_strs = [
            "",  # Empty string
            "\ud800",  # bpo-34454: Surrogate code point
            "2009.04-19T03",  # Wrong first separator
            "2009-04.19T03",  # Wrong second separator
            "2009-04-19T0a",  # Invalid hours
            "2009-04-19T03:1a:45",  # Invalid minutes
            "2009-04-19T03:15:4a",  # Invalid seconds
            "2009-04-19T03;15:45",  # Bad first time separator
            "2009-04-19T03:15;45",  # Bad second time separator
            "2009-04-19T03:15:4500:00",  # Bad time zone separator
            "2009-04-19T03:15:45.2345",  # Too many digits for milliseconds
            "2009-04-19T03:15:45.1234567",  # Too many digits for microseconds
            "2009-04-19T03:15:45.123456+24:30",  # Invalid time zone offset
            "2009-04-19T03:15:45.123456-24:30",  # Invalid negative offset
            "2009-04-10ᛇᛇᛇᛇᛇ12:15",  # Too many unicode separators
            "2009-04\ud80010T12:15",  # Surrogate char in date
            "2009-04-10T12\ud80015",  # Surrogate char in time
            "2009-04-19T1",  # Incomplete hours
            "2009-04-19T12:3",  # Incomplete minutes
            "2009-04-19T12:30:4",  # Incomplete seconds
            "2009-04-19T12:",  # Ends with time separator
            "2009-04-19T12:30:",  # Ends with time separator
            "2009-04-19T12:30:45.",  # Ends with time separator
            "2009-04-19T12:30:45.123456+",  # Ends with timzone separator
            "2009-04-19T12:30:45.123456-",  # Ends with timzone separator
            "2009-04-19T12:30:45.123456-05:00a",  # Extra text
            "2009-04-19T12:30:45.123-05:00a",  # Extra text
            "2009-04-19T12:30:45-05:00a",  # Extra text
        ]

        for bad_str in bad_strs:
            self.assertRaises(ValueError, self.theclass.fromisoformat, bad_str)


class TestTime(Static[TestCase]):
    theclass: type = time

    def test_basic_attributes(self):
        t = self.theclass(12, 0)
        self.assertEqual(t.hour, 12)
        self.assertEqual(t.minute, 0)
        self.assertEqual(t.second, 0)
        self.assertEqual(t.microsecond, 0)

    def test_basic_attributes_nonzero(self):
        # Make sure all attributes are non-zero so bugs in
        # bit-shifting access show up.
        t = self.theclass(12, 59, 59, 8000)
        self.assertEqual(t.hour, 12)
        self.assertEqual(t.minute, 59)
        self.assertEqual(t.second, 59)
        self.assertEqual(t.microsecond, 8000)

    def test_comparing(self):
        args = (1, 2, 3, 4)
        newargsx = [(2, 2, 3, 4), (1, 3, 3, 4), (1, 2, 4, 4), (1, 2, 3, 5)]
        t1 = self.theclass(*args)
        t2 = self.theclass(*args)
        self.assertEqual(t1, t2)
        self.assertTrue(t1 <= t2)
        self.assertTrue(t1 >= t2)
        self.assertFalse(t1 != t2)
        self.assertFalse(t1 < t2)
        self.assertFalse(t1 > t2)

        for i in range(len(args)):
            newargs = newargsx[i]
            t2 = self.theclass(*newargs)  # this is larger than t1
            self.assertTrue(t1 < t2)
            self.assertTrue(t2 > t1)
            self.assertTrue(t1 <= t2)
            self.assertTrue(t2 >= t1)
            self.assertTrue(t1 != t2)
            self.assertTrue(t2 != t1)
            self.assertFalse(t1 == t2)
            self.assertFalse(t2 == t1)
            self.assertFalse(t1 > t2)
            self.assertFalse(t2 < t1)
            self.assertFalse(t1 >= t2)
            self.assertFalse(t2 <= t1)

        """
        for badarg in OTHERSTUFF:
            self.assertEqual(t1 == badarg, False)
            self.assertEqual(t1 != badarg, True)
            self.assertEqual(badarg == t1, False)
            self.assertEqual(badarg != t1, True)

            self.assertRaises(TypeError, lambda: t1 <= badarg)
            self.assertRaises(TypeError, lambda: t1 < badarg)
            self.assertRaises(TypeError, lambda: t1 > badarg)
            self.assertRaises(TypeError, lambda: t1 >= badarg)
            self.assertRaises(TypeError, lambda: badarg <= t1)
            self.assertRaises(TypeError, lambda: badarg < t1)
            self.assertRaises(TypeError, lambda: badarg > t1)
            self.assertRaises(TypeError, lambda: badarg >= t1)
        """

    def test_bad_constructor_arguments(self):
        # bad hours
        self.theclass(0, 0)  # no exception
        self.theclass(23, 0)  # no exception

        make_time1 = lambda a, b: self.theclass(a, b)
        make_time2 = lambda a, b, c: self.theclass(a, b, c)
        make_time3 = lambda a, b, c, d: self.theclass(a, b, c, d)

        self.assertRaises(ValueError, make_time1, -1, 0)
        self.assertRaises(ValueError, make_time1, 24, 0)
        # bad minutes
        self.theclass(23, 0)  # no exception
        self.theclass(23, 59)  # no exception
        self.assertRaises(ValueError, make_time1, 23, -1)
        self.assertRaises(ValueError, make_time1, 23, 60)
        # bad seconds
        self.theclass(23, 59, 0)  # no exception
        self.theclass(23, 59, 59)  # no exception
        self.assertRaises(ValueError, make_time2, 23, 59, -1)
        self.assertRaises(ValueError, make_time2, 23, 59, 60)
        # bad microseconds
        self.theclass(23, 59, 59, 0)  # no exception
        self.theclass(23, 59, 59, 999999)  # no exception
        self.assertRaises(ValueError, make_time3, 23, 59, 59, -1)
        self.assertRaises(ValueError, make_time3, 23, 59, 59, 1000000)

    def test_hash_equality(self):
        d = self.theclass(23, 30, 17)
        e = self.theclass(23, 30, 17)
        self.assertEqual(d, e)
        self.assertEqual(hash(d), hash(e))

        dic = {d: 1}
        dic[e] = 2
        self.assertEqual(len(dic), 1)
        self.assertEqual(dic[d], 2)
        self.assertEqual(dic[e], 2)

        d = self.theclass(0, 5, 17)
        e = self.theclass(0, 5, 17)
        self.assertEqual(d, e)
        self.assertEqual(hash(d), hash(e))

        dic = {d: 1}
        dic[e] = 2
        self.assertEqual(len(dic), 1)
        self.assertEqual(dic[d], 2)
        self.assertEqual(dic[e], 2)

    def test_isoformat(self):
        t = self.theclass(4, 5, 1, 123)
        self.assertEqual(t.isoformat(), "04:05:01.000123")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass()
        self.assertEqual(t.isoformat(), "00:00:00")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass(microsecond=1)
        self.assertEqual(t.isoformat(), "00:00:00.000001")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass(microsecond=10)
        self.assertEqual(t.isoformat(), "00:00:00.000010")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass(microsecond=100)
        self.assertEqual(t.isoformat(), "00:00:00.000100")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass(microsecond=1000)
        self.assertEqual(t.isoformat(), "00:00:00.001000")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass(microsecond=10000)
        self.assertEqual(t.isoformat(), "00:00:00.010000")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass(microsecond=100000)
        self.assertEqual(t.isoformat(), "00:00:00.100000")
        self.assertEqual(t.isoformat(), str(t))

        t = self.theclass(hour=12, minute=34, second=56, microsecond=123456)
        self.assertEqual(t.isoformat(timespec="hours"), "12")
        self.assertEqual(t.isoformat(timespec="minutes"), "12:34")
        self.assertEqual(t.isoformat(timespec="seconds"), "12:34:56")
        self.assertEqual(t.isoformat(timespec="milliseconds"), "12:34:56.123")
        self.assertEqual(t.isoformat(timespec="microseconds"), "12:34:56.123456")
        self.assertEqual(t.isoformat(timespec="auto"), "12:34:56.123456")

        t = self.theclass(hour=12, minute=34, second=56, microsecond=999500)
        self.assertEqual(t.isoformat(timespec="milliseconds"), "12:34:56.999")

        t = self.theclass(hour=12, minute=34, second=56, microsecond=0)
        self.assertEqual(t.isoformat(timespec="milliseconds"), "12:34:56.000")
        self.assertEqual(t.isoformat(timespec="microseconds"), "12:34:56.000000")
        self.assertEqual(t.isoformat(timespec="auto"), "12:34:56")

    def test_str(self):
        self.assertEqual(str(self.theclass(1, 2, 3, 4)), "01:02:03.000004")
        self.assertEqual(str(self.theclass(10, 2, 3, 4000)), "10:02:03.004000")
        self.assertEqual(str(self.theclass(0, 2, 3, 400000)), "00:02:03.400000")
        self.assertEqual(str(self.theclass(12, 2, 3, 0)), "12:02:03")
        self.assertEqual(str(self.theclass(23, 15, 0, 0)), "23:15:00")

    def test_repr(self):
        self.assertEqual(
            repr(self.theclass(1, 2, 3, 4)),
            "time(hour=1, minute=2, second=3, microsecond=4)",
        )
        self.assertEqual(
            repr(self.theclass(10, 2, 3, 4000)),
            "time(hour=10, minute=2, second=3, microsecond=4000)",
        )
        self.assertEqual(
            repr(self.theclass(0, 2, 3, 400000)),
            "time(hour=0, minute=2, second=3, microsecond=400000)",
        )
        self.assertEqual(
            repr(self.theclass(12, 2, 3, 0)), "time(hour=12, minute=2, second=3)"
        )
        self.assertEqual(repr(self.theclass(23, 15, 0, 0)), "time(hour=23, minute=15)")

    def test_resolution_info(self):
        self.assertTrue(self.theclass.max > self.theclass.min)

    def test_bool(self):
        # time is always True.
        cls = self.theclass
        self.assertTrue(cls(1))
        self.assertTrue(cls(0, 1))
        self.assertTrue(cls(0, 0, 1))
        self.assertTrue(cls(0, 0, 0, 1))
        self.assertTrue(cls(0))
        self.assertTrue(cls())

    def test_replace(self):
        cls = self.theclass
        args = (1, 2, 3, 4)
        base = cls(*args)
        self.assertEqual(base, base.replace())

        self.assertEqual(base.replace(hour=5), cls(5, 2, 3, 4))
        self.assertEqual(base.replace(minute=6), cls(1, 6, 3, 4))
        self.assertEqual(base.replace(second=7), cls(1, 2, 7, 4))
        self.assertEqual(base.replace(microsecond=8), cls(1, 2, 3, 8))

        # Out of bounds.
        base = cls(1)
        self.assertRaises(ValueError, lambda: base.replace(hour=24))
        self.assertRaises(
            ValueError, lambda: base.replace(minute=-2)
        )  # minute=-1 indicates default; changed to -2 for test
        self.assertRaises(ValueError, lambda: base.replace(second=100))
        self.assertRaises(ValueError, lambda: base.replace(microsecond=1000000))


case_td = TestTimeDelta()
case_td.test_computations()
case_td.test_constructor()
case_td.test_resolution_info()
case_td.test_basic_attributes()
case_td.test_total_seconds()
case_td.test_carries()
case_td.test_hash_equality()
case_td.test_compare()
case_td.test_str()
case_td.test_repr()
case_td.test_microsecond_rounding()
case_td.test_massive_normalization()
case_td.test_bool()
case_td.test_division()
case_td.test_remainder()
case_td.test_divmod()

case_do = TestDateOnly()
case_do.test_delta_non_days_ignored()

case_dx = TestDate[date]()
case_dx.test_basic_attributes()
case_dx.test_ordinal_conversions()
# case_dx.test_extreme_ordinals()
case_dx.test_bad_constructor_arguments()
case_dx.test_hash_equality()
case_dx.test_computations()
case_dx.test_fromtimestamp()
case_dx.test_today()
case_dx.test_weekday()
case_dx.test_isocalendar()
case_dx.test_iso_long_years()
case_dx.test_isoformat()
case_dx.test_ctime()
case_dx.test_timetuple()
case_dx.test_compare()
case_dx.test_replace()
case_dx.test_fromisoformat()
case_dx.test_fromisoformat_fails()
case_dx.test_fromisocalendar()
case_dx.test_fromisocalendar_value_errors()

case_dt = TestDateTime[datetime]()
case_dt.test_ordinal_conversions()
case_dt.test_today()
case_dt.test_weekday()
case_dt.test_isocalendar()
case_dt.test_iso_long_years()
case_dt.test_ctime()
case_dt.test_timetuple()
case_dt.test_compare()
case_dt.test_fromisoformat()
case_dt.test_fromisoformat_fails()
case_dt.test_fromisocalendar_value_errors()
# ---
case_dt.test_basic_attributes()
case_dt.test_basic_attributes_nonzero()
case_dt.test_isoformat()
# case_dt.test_more_ctime()
case_dt.test_tz_independent_comparing()
case_dt.test_bad_constructor_arguments()
case_dt.test_hash_equality()
case_dt.test_computations()
case_dt.test_more_compare()
case_dt.test_fromtimestamp()
# case_dt.test_utcfromtimestamp()
case_dt.test_timestamp_naive()
# case_dt.test_microsecond_rounding()
# case_dt.test_timestamp_limits()
case_dt.test_negative_float_fromtimestamp()
# case_dt.test_negative_float_utcfromtimestamp()
case_dt.test_utcnow()
case_dt.test_extract()
case_dt.test_combine()
case_dt.test_replace()
case_dt.test_fromisoformat_datetime()
case_dt.test_fromisoformat_separators()
case_dt.test_fromisoformat_ambiguous()
case_dt.test_fromisoformat_timespecs()
case_dt.test_fromisoformat_fails_datetime()

case_tx = TestTime()
case_tx.test_basic_attributes()
case_tx.test_basic_attributes_nonzero()
case_tx.test_comparing()
case_tx.test_bad_constructor_arguments()
case_tx.test_hash_equality()
case_tx.test_isoformat()
case_tx.test_str()
case_tx.test_repr()
case_tx.test_resolution_info()
case_tx.test_bool()
case_tx.test_replace()

@test
def test_constants():
    assert str(timedelta.min) == '-106751992 days, 19:59:05.224192'  # note: diff w/ Python
    assert str(timedelta.max) == '106751991 days, 4:00:54.775807'    # note: diff w/ Python
    assert str(timedelta.resolution) == '0:00:00.000001'

    assert str(date.min) == '0001-01-01'
    assert str(date.max) == '9999-12-31'
    assert str(date.resolution) == '1 day, 0:00:00'

    assert str(time.min) == '00:00:00'
    assert str(time.max) == '23:59:59.999999'
    assert str(time.resolution) == '0:00:00.000001'

    assert str(datetime.min) == '0001-01-01 00:00:00'
    assert str(datetime.max) == '9999-12-31 23:59:59.999999'
    assert str(datetime.resolution) == '0:00:00.000001'
test_constants()
