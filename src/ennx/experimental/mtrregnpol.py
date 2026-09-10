from enum import Enum


class SharingPolicy(str, Enum):
    SHARED = "shared"
    NEAREST_CENTER = "nearest_center"
    INDEPENDENT = "independent"
