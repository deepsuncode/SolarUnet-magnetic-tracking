# CHANGELOG (3rd Feb 2024)

## System configuration

* Hardware - Apple M2 Pro Chip
* OS - MacOS 14.3
* Python version - v3.8.18
* Pip - v23.0.1

## Package Versions

1. astropy==5.2.2
2. keras==2.13.1
3. matplotlib==3.1.0
4. numpy==1.24.3
5. opencv-python==4.9.0.80
6. scipy==1.10.1
7. scikit-image==0.21.0
8. scikit-learn==1.3.2
9. tensorflow==2.13.0
10. tensorflow-metal - Optional and only for Apple silicon

## Code Change

1. magnetic_tracking.py - Line 87

    **Old Code**

    `structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter`

    **New Code**

    `structure = np.ones((3, 3), dtype=np.int64)  # this defines the connection filter`

2. statistics_analysis.py - Line 80

    **Old Code**
    `solarNet_structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter`

    **New Code**
    `solarNet_structure = np.ones((3, 3), dtype=np.int64)  # this defines the connection filter`
