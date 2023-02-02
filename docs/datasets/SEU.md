# SEU

## Dataset Description
This dataset contains mechanical data used in the paper "Highly-Accurate Machine Fault Diagnosis Using Deep Transfer Learning". All the data are the original vibration signals acquired by sensors.
This dataset is from Southeast University, China. These data are collected from Drivetrain Dynamic Simulator. This dataset contains 2 subdatasets, including bearing data and gear data, which are both acquired on Drivetrain Dynamics Simulator (DDS). There are two kinds of working conditions with rotating speed - load configuration set to be 20-0 and 30-2.

## Dataset Structure

- All data files are in Comma-separated values (*.csv) format.

- Within each file, there are 8 rows of signals which represent: 

    - 1 - Motor vibration

    - 2,3,4 - Vibration of planetary gearbox in three directions: x, y, and z

    - 5 - Motor torque

    - 6,7,8 - Vibration of parallel gear box in three directions: x, y, and z
    
    - Signals of rows 2,3,4 are all effective.

- Rotation Frequency: 20 ~ 30 Hz

- Classes: ??

- Domain: Time

## Summary

#### Label distribution
|       Label          | Number samples |
|:--------------------:|:--------------:|
|         0            |      32        |
|         1            |      16        |
|         2            |      16        |
|         3            |      16        |
|         4            |      16        |
|         5            |      16        |
|         6            |      16        |
|         7            |      16        |
|         8            |      16        |
|      **Total**       |      160       |

<!-- ![image](../../images/SEU/label_dist.png) -->


#### Signal size distribution
|   Size    | Number samples |
|:---------:|:--------------:|
| 1048560   |     160        |
| **Total** |     160        |

<!-- ![image](../../images/SEU/signal_size_dist.png) -->


#### Rotatory frequency distribution
| Frequency (Hz) | Number samples |
|:--------------:|:--------------:|
|      20        |      80        |
|      30        |      80        |
|   **Total**    |      160       |

<!-- ![image](../../images/SEU/frequency_dist.png) -->
