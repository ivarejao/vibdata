# MAFAULDA

## Dataset Description
This database is composed of 1951 multivariate time-series acquired by sensors on a SpectraQuest's Machinery Fault Simulator (MFS) Alignment-Balance-Vibration (ABVT). The 1951 comprises six different simulated states: normal function, imbalance fault, horizontal and vertical misalignment faults and, inner and outer bearing faults.

## Dataset Structure

- The database is composed by several CSV (Comma-Separated Values) files, each one with 8 columns, one column for each sensor, according to:

    Column 1: Tachometer signal that allows to estimate rotation frequency;

    Columns 2 to 4: Underhang bearing accelerometer (axial, radiale tangential direction);

    Columns 5 to 7: Overhang bearing accelerometer (axial, radiale tangential direction);

    Column 8: microphone.

- Rotation: Hz

- Classes: Normal, Horizontal Misalignment, Vertical Misalignment, Imbalance, Underhang Cage Fault, Underhang Outer Race, Underhang Ball Fault, Overhang Cage Fault, Overhang Outer Race and Overhang Ball Fault.

- Domain: Time

- Sample rate:  samples per second

## Summary

#### Label distribution
|       Label               | Number samples |
|:-------------------------:|:--------------:|
|  Normal                   |                |
|  Horizontal Misalignment  |                |
|  Vertical Misalignment    |                |
|  Imbalance                |                |
|  Underhang Cage           |                |
|  Underhang Outer Race     |                |
|  Underhang Ball           |                |
|  Overhang Cage            |                |
|  Overhang Outer Race      |                |
|  Overhang Ball            |                |
|      **Total**            |                |

<!-- ![image](../../images/MAFAULDA/label_dist.png) -->


#### Signal size distribution
|   Size    | Number samples |
|:---------:|:--------------:|
|           |                |
| **Total** |                |

<!-- ![image](../../images/MAFAULDA/signal_size_dist.png) -->


#### Rotatory frequency distribution
| Frequency (Hz) | Number samples |
|:--------------:|:--------------:|
|                |                |
|                |                |
|                |                |
|   **Total**    |                |

<!-- ![image](../../images/MAFAULDA/frequency_dist.png) -->
