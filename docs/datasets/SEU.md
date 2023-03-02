# SEU

## Dataset Description
This dataset contains mechanical data used in the paper "Highly-Accurate Machine Fault Diagnosis Using Deep Transfer Learning". All the data are the original vibration signals acquired by sensors.
- Bearing dataset is from Case Western Reserve University bearing data center.
- Gearbox dataset is from Southeast University, China. These data are collected from Drivetrain Dynamic Simulator.

## Dataset Structure

- **Bearing dataset**

  - For the names of the data files, for example 'B007_0', the first letter represents fault position, 
  next three numbers represent fault diameters (0.007, 0.014, 0.021 inches) and the last number represents bearing loads(0,1,2,3).
  There are three kinds of fault positions here, B-bearing rolling element, IR-inner raceway, OR-outer raceway. 
  Data we used are all from fan end which is marked as 'FE' in the data files.

- **Gearbox dataset**

  - Within each file, there are 8 rows of signals which represent: 

      - 1 - Motor vibration

      - 2,3,4 - Vibration of planetary gearbox in three directions: x, y, and z

      - 5 - Motor torque

      - 6,7,8 - Vibration of parallel gear box in three directions: x, y, and z
    
      - Signals of rows 2,3,4 are all effective.

- Rotation Frequency: 20 ~ 30 Hz

- Classes: Ball, Inner Ring, Outer Ring, Inner Ring + Outer Ring, Chipped Tooth, Missing Tooth, Root Fault and Surface Fault

- Domain: Time

## Summary

#### Label distribution
|           Label           | Number samples |
|:-------------------------:|:--------------:|
|          Normal           |       32       |
|           Ball            |       16       |
|        Inner Ring         |       16       |
|        Outer Ring         |       16       |
| Inner Ring and Outer Ring |       16       |
|       Chipped Tooth       |       16       |
|       Missing Tooth       |       16       |
|        Root Fault         |       16       |
|       Surface Fault       |       16       |
|         **Total**         |      160       |

![image](../../images/SEU/label_dist.png)


#### Signal size distribution
|   Size    | Number samples |
|:---------:|:--------------:|
|  1048560  |      160       |
| **Total** |      160       |

![image](../../images/SEU/signal_size_dist.png)


#### Rotatory frequency distribution
| Frequency (Hz) | Number samples |
|:--------------:|:--------------:|
|       20       |       80       |
|       30       |       80       |
|   **Total**    |      160       |

![image](../../images/SEU/frequency_dist.png)
