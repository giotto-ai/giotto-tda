"""Base class for real intervals."""
# License: GNU AGPLv3

from numbers import Real
from operator import le, lt


def _interval_like(other):
    return (hasattr(other, 'left')
            and hasattr(other, 'right')
            and hasattr(other, 'closed'))


class Interval:
    """Immutable object implementing an interval.

    Parameters
    ----------
    left : real scalar, required
        Left bound for the interval.

    right : real scalar, required
        Right bound for the interval.

    closed : ``'right'`` | ``'left'`` | ``'both'`` | ``'neither'``, required
        Whether the interval is closed on the left-side, right-side, both or
        neither.

    """
    _VALID_CLOSED = frozenset(['left', 'right', 'both', 'neither'])

    def __init__(self, left, right, *, closed):
        self._validate_endpoint(left)
        self._validate_endpoint(right)
        if closed not in self._VALID_CLOSED:
            raise ValueError(
                f"Invalid option for `closed`: {closed}. Argument must be "
                f"one of {list(self._VALID_CLOSED)}.")
        if not left <= right:
            raise ValueError("Left side of interval must be <= right side")

        self.left = left
        self.right = right
        self.closed = closed

    @staticmethod
    def _validate_endpoint(endpoint):
        if not isinstance(endpoint, Real):
            raise ValueError(
                "Only real (finite or infinite) endpoints are allowed when "
                "constructing an Interval.")

    @property
    def closed_left(self):
        """Check if the interval is closed on the left side.

        """
        return self.closed in ('left', 'both')

    @property
    def closed_right(self):
        """Check if the interval is closed on the right side.

        """
        return self.closed in ('right', 'both')

    @property
    def open_left(self):
        """Check if the interval is open on the left side.

        """
        return not self.closed_left

    @property
    def open_right(self):
        """Check if the interval is closed on the left side.

        """
        return not self.closed_right

    @property
    def mid(self):
        """Return the midpoint of the interval. Take care when the left or
        right sides are infinite.

        """
        return 0.5 * (self.left + self.right)

    @property
    def length(self):
        """Return the length of the interval. Take care when the left or
        right sides are infinite.

        """
        return self.right - self.left

    @property
    def is_empty(self):
        """Indicates if an interval is empty, meaning it contains no points.

        """
        return (self.right == self.left) & (self.closed != 'both')

    def __hash__(self):
        return hash((self.left, self.right, self.closed))

    def __contains__(self, key):
        if _interval_like(key):
            raise TypeError("__contains__ not defined for two intervals")
        return ((self.left < key if self.open_left else self.left <= key) &
                (key < self.right if self.open_right else key <= self.right))

    def __reduce__(self):
        args = (self.left, self.right, self.closed)
        return type(self), args

    def __repr__(self):
        left, right = self.left, self.right
        name = type(self).__name__
        repr_str = f"{name}({repr(left)}, {repr(right)}, " \
                   f"closed={repr(self.closed)})"
        return repr_str

    def __str__(self):
        left, right = self.left, self.right
        start_symbol = '[' if self.closed_left else '('
        end_symbol = ']' if self.closed_right else ')'
        return f'{start_symbol}{left}, {right}{end_symbol}'

    def __add__(self, y):
        if isinstance(y, Real):
            return Interval(self.left + y, self.right + y, closed=self.closed)
        elif isinstance(y, Interval) and isinstance(self, Real):
            return Interval(y.left + self, y.right + self, closed=y.closed)
        return NotImplemented

    def __sub__(self, y):
        if isinstance(y, Real):
            return Interval(self.left - y, self.right - y, closed=self.closed)
        return NotImplemented

    def __mul__(self, y):
        if isinstance(y, Real):
            return Interval(self.left * y, self.right * y, closed=self.closed)
        elif isinstance(y, Interval) and isinstance(self, Real):
            return Interval(y.left * self, y.right * self, closed=y.closed)
        return NotImplemented

    def __div__(self, y):
        if isinstance(y, Real):
            return Interval(self.left / y, self.right / y, closed=self.closed)
        return NotImplemented

    def __truediv__(self, y):
        if isinstance(y, Real):
            return Interval(self.left / y, self.right / y, closed=self.closed)
        return NotImplemented

    def __floordiv__(self, y):
        if isinstance(y, Real):
            return Interval(
                self.left // y, self.right // y, closed=self.closed)
        return NotImplemented

    def intersects(self, other):
        """Check whether two :cls:`Interval` objects intersect. Two
        intervals intersect if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        intersect.

        Parameters
        ----------
        other : Interval object
            Interval to check against for an overlap.

        Returns
        -------
        bool
            ``True`` if the two intervals overlap.

        """
        if not isinstance(other, Interval):
            raise TypeError("`other` must be an Interval, "
                            f"got {type(other).__name__}")

        # equality is okay if both endpoints are closed (overlap at a point)
        op1 = le if (self.closed_left and other.closed_right) else lt
        op2 = le if (other.closed_left and self.closed_right) else lt

        # overlaps is equivalent negation of two interval being disjoint:
        # disjoint = (A.left > B.right) or (B.left > A.right)
        # (simplifying the negation allows this to be done in fewer operations)
        return op1(self.left, other.right) and op2(other.left, self.right)
