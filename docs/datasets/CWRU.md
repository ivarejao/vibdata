# CWRU

## Dataset Description
This dataset provides access to ball bearing test data for normal and faulty bearings.  
Experiments were conducted using a 2 hp Reliance Electric motor, and acceleration data was measured at locations near to and remote from the motor bearings.

## Dataset Structure

- All data files are in Matlab (*.mat) format. Each file contains fan and drive end vibration data as well as motor rotational speed.  For all files, the following item in the variable name indicates:

    - DE - drive end accelerometer data

    - FE - fan end accelerometer data

    - BA - base accelerometer data

    - time - time series data
    
    - RPM - rpm during testing

- Rotation Speed: 1720 ~ 1797 RPM 

- Rotation Frequency: 28.83 ~ 29.95 Hz

- Classes: Normal, Inner Race, Outer Race and Ball

- Domain: Time

- Sample Rate: 12000 and 48000 samples per second

## Summary

#### Label distribution
|   Label    | Number samples |
|:----------:|:--------------:|
|   Normal   |       8        |
| Inner Race |       94       |
| Outer Race |       96       |
|    Ball    |       96       |
| **Total**  |      294       |

![image](../../images/CWRU/label_dist.png)


#### Signal size distribution
|      Size       | Number samples |
|:---------------:|:--------------:|
| 120617 ~ 491446 |      294       |
|    **Total**    |      294       |

![image](../../images/CWRU/signal_size_dist.png)


#### Rotatory frequency distribution
| Frequency (Hz) | Number samples |
|:--------------:|:--------------:|
|     28.83      |       74       |
|     29.17      |       74       |
|     29.53      |       74       |
|     29.95      |       72       |
|   **Total**    |      294       |

![image](../../images/CWRU/frequency_dist.png)
